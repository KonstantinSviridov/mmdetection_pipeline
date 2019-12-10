from mmcv import Config
from mmcv.runner import Runner
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.core import (DistOptimizerHook, DistEvalmAPHook,
                        CocoDistEvalRecallHook, CocoDistEvalmAPHook,
                        Fp16OptimizerHook)
import mmdet.datasets as mmdetDatasets
from mmdet.datasets import build_dataloader
from mmdet.models import RPN

import imgaug

import numpy as np
import tqdm
from segmentation_models.utils import set_trainable
import keras
from musket_core import configloader, datasets
from musket_core.generic_config import ExecutionConfig, ReporterCallback, KFoldCallback
import os
from musket_core.datasets import SubDataSet,DataSet, WriteableDataSet
from mmdetection_pipeline import checkpoint_registry
from mmdetection_pipeline.tools import isURL

from mmdet import __version__
from mmdet.apis.train import build_optimizer, batch_processor
from mmdet.apis import get_root_logger, inference_detector
from mmdet.models import build_detector
from mmdet.datasets.coco import CocoDataset
from mmcv.runner import load_checkpoint
import torch
import torch.distributed
from torch.utils import model_zoo

import musket_core.generic_config as generic
from musket_core.builtin_trainables import OutputMeta
from mmdetection_pipeline.metrics import mmdet_mAP_bbox, mAP_masks
from mmdetection_pipeline.data import MMdetWritableDS, DataSetAdapter, InstanceSegmentationPredictionBlend
from mmdetection_pipeline.callbacks import HookWrapper, CustomCheckpointHook, DrawSamplesHook, KerasCBWrapper
from mmdetection_pipeline.conversions import convertMMDETModelOutput

class MMDetWrapper:
    def __init__(self, cfg:Config, weightsPath:str, classes: [str]):
        self.cfg = cfg
        self.weightsPath = weightsPath
        self.output_dim = 4
        self.stop_training = False
        self.classes = classes
        self.model = None

    def __call__(self, *args, **kwargs):
        return OutputMeta(self.output_dim, self)

    def compile(self, *args, **kwargs):
        cfg = self.cfg
        self.model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)#init_detector(self.cfg, self.weightsPath, device='cuda:0')
        self.model.CLASSES = self.classes
        pass

    def eval_func(self, y_true, y_pred, f, session, mean=True):
        func = f[0]

        arg1 = f[1]
        arg2 = f[2]

        if mean:
            return np.mean(session.run(func, {arg1: y_true, arg2: y_pred}))

        return session.run(func, {arg1: y_true, arg2: y_pred})

    def eval_metrics(self, y_true, y_pred, session):
        # result = {}
        #
        # for item in self.custom_metrics.keys():
        #     preds = y_pred
        #
        #     if generic_config.need_threshold(item):
        #         preds = (preds > 0.5).astype(np.float32)
        #
        #     result[item] = self.eval_func(y_true, preds, self.custom_metrics[item], session)
        #
        # return result
        print("eval_metrics")
        pass

    def to_tensor(self, func):
        # i1 = keras.layers.Input((self.output_dim,))
        # i2 = keras.layers.Input((self.output_dim,))
        #
        # return func(i1, i2), i1, i2
        pass

    def convert_data(self, generator):
        result_x = []
        result_y = []

        for item in generator:
            result_x.append(item[0])
            result_y.append(item[1])

        result_x = np.concatenate(result_x)
        result_y = np.concatenate(result_y)

        result_x = np.reshape(result_x, (len(result_x), -1))
        result_y = np.reshape(result_y, (len(result_y), -1))

        if self.output_dim > 1:
            result_y = np.argmax(result_y, 1)
        else:
            result_y = (result_y > 0.5).flatten()

        return result_x.astype(np.float32), result_y.astype(np.int32)

    def setClasses(self, classes:[str]):
        self.model.CLASSES = classes

    def predict(self, *args, **kwargs):

        self.model.cfg = self.cfg
        self.model.to(torch.cuda.current_device())

        input = args[0]

        self.model.eval()
        predictions = inference_detector(self.model, input)

        wm = self.model.with_mask
        result = [convertMMDETModelOutput(x, wm) for x in predictions]
        return result

    def load_weights(self, path, val = None):

        if self.model is None:
            self.compile()

        self.cfg.resume_from = path
        checkpoint = load_checkpoint(self.model, path)

    def numbers_to_vectors(self, numbers):
        result = np.zeros((len(numbers), self.output_dim))

        count = 0

        if self.output_dim == 1:
            for item in numbers:
                result[count, 0] = item

                count += 1

            return result

        for item in numbers:
            result[count, item] = 1

            count += 1

        return result

    def groups_to_vectors(self, data, length):
        result = np.zeros((length, self.output_dim))

        if self.output_dim == 1:
            result[:, 0] = data

            return result

        if self.output_dim == 2:
            ids = np.array(range(length), np.int32)

            ids = [ids, (data > 0.5).astype(np.int32)]

            result[ids] = 1

            return result

        for item in range(self.output_dim):
            result[:, item] = data[length * item : length * (item + 1)]

        return result

    def to_tf(self, numbers, data):
        y_true = self.numbers_to_vectors(numbers)

        y_pred = self.groups_to_vectors(data, len(numbers))

        return y_true, y_pred

    def save(self, file_path, overwrite):
        if hasattr(self.model, "booster_"):
            self.model.booster_.save_model(file_path)

class PipelineConfig(generic.GenericImageTaskConfig):

    def evaluate(self, d, fold, stage, negatives="all", limit=16):
        mdl = self.load_model(fold, stage)
        ta = self.transformAugmentor()
        folds = self.kfold(d, range(0, len(d)))
        rs = folds.load(fold, False, negatives, limit)

        for z in ta.augment_batches([rs]):
            res = mdl.predict(np.array(z.images_aug))
            z.heatmaps_aug = [imgaug.HeatmapsOnImage(x, x.shape) for x in res];
            yield z
        pass

    def createStage(self,x):
        return DetectionStage(x, self)

    def  __init__(self,**atrs):
        self.configPath = None
        self.weightsPath = None
        self.nativeConfig = None
        self.imagesPerGpu = None
        self.resetHeads = True
        self.threshold = 0.5
        super().__init__(**atrs)
        if 'folds_count' not in atrs:
            self.folds_count = 1
        self.dataset_clazz = datasets.ImageKFoldedDataSet
        self.flipPred=False
        self.final_metrics.extend([mAP_masks.__name__, mmdet_mAP_bbox.__name__])
        self.needsSessionForPrediction = False

    def initNativeConfig(self):

        atrs = self.all
        self.nativeConfig = Config.fromfile(self.getNativeConfigPath())
        cfg = self.nativeConfig
        cfg.gpus = self.gpus

        wd = os.path.dirname(self.path)
        cfg.work_dir = wd

        if 'bbox_head' in cfg.model and 'classes' in atrs:
            setCfgAttr(cfg.model.bbox_head, 'num_classes', atrs['classes']+1)

        if 'mask_head' in cfg.model and 'classes' in atrs:
            setCfgAttr(cfg.model.mask_head, 'num_classes', atrs['classes']+1)

        weightsPath = self.getWeightsPath()
        if self.weightsPath is not None:
            cfg.load_from = weightsPath
            cfg.model.pretrained = None #prevent weights form being loaded during model init

        cfg.resetHeads = self.resetHeads

        cfg.total_epochs = None
        if self.imagesPerGpu is not None:
            cfg.data.imgs_per_gpu = self.imagesPerGpu
        cfg.data.workers_per_gpu = 1
        cfg.log_config.interval = 1
        modelCfg = cfg['model']

        self.setNumClasses(modelCfg, 'bbox_head')
        self.setNumClasses(modelCfg, 'mask_head')
        self.setNumClasses(modelCfg, 'mask_iou_head')

        if 'semantic_head' in modelCfg:
            del modelCfg['semantic_head']

        if 'semantic_roi_extractor' in modelCfg:
            del modelCfg['semantic_roi_extractor']

        self.setImageScale(cfg.data, 'train', True)
        self.setImageScale(cfg.data, 'test')
        self.setImageScale(cfg.data, 'val')

        self.disableSemanticHead(cfg.data, 'train')
        self.disableSemanticHead(cfg.data, 'test')
        self.disableSemanticHead(cfg.data, 'val')

        cfg.workflow = [ ('train',1),('val',1) ]

        # # set cudnn_benchmark
        # if cfg.get('cudnn_benchmark', False):
        #     torch.backends.cudnn.benchmark = True
        # # update configs according to CLI args
        #
        # if args_resume_from is not None:
        #     cfg.resume_from = args_resume_from
        #

    def setNumClasses(self, modelCfg, moduleTitle):
        if not moduleTitle in modelCfg:
            return

        m = modelCfg[moduleTitle]
        if isinstance(m,list):
            for x in m:
                x['num_classes'] = self.classes + 1
        else:
            m['num_classes'] = self.classes + 1

    def setImageScale(self, dataCfg, moduleTitle, multi=False):
        if not moduleTitle in dataCfg:
            return

        m = dataCfg[moduleTitle]
        shapesList = self.shape
        if not isinstance(shapesList[0], list):
            shapesList = [ shapesList ]

        shapesList = [ tuple(x[:2]) for x in shapesList ]
        value = shapesList
        if not multi or len(shapesList) == 1:
            value = shapesList[0]

        m['img_scale'] = value

    def disableSemanticHead(self, dataCfg, moduleTitle, multi=False):
        if not moduleTitle in dataCfg:
            return

        m = dataCfg[moduleTitle]
        if 'with_semantic_seg' in m:
            m['with_semantic_seg'] = False

    def __setattr__(self, key, value):
        hasAttr = hasattr(self,key)
        super().__setattr__(key,value)
        if key == 'gpus' and hasAttr:
            self.initNativeConfig()

    def getWeightsPath(self):
        wp = self.weightsPath
        if os.path.isabs(wp) or wp.startswith("open-mmlab://") or isURL(wp):
            return wp

        wd = os.path.dirname(self.path)
        joined = os.path.join(wd, wp)
        result = os.path.normpath(joined)
        return result

    def getWeightsOutPath(self):
        wd = os.path.dirname(self.path)
        joined = os.path.join(wd, 'weights')
        result = os.path.normpath(joined)
        return result

    def getNativeConfigPath(self):
        wd = os.path.dirname(self.path)
        joined = os.path.join(wd, self.configPath)
        result = os.path.normpath(joined)
        return result

    def update(self,z,res):
        z.segmentation_maps_aug = [imgaug.SegmentationMapOnImage(x, x.shape) for x in res];
        pass

    def createNet(self):
        classes = self.get_dataset().root().meta()['CLASSES']
        result = MMDetWrapper(self.nativeConfig, self.getWeightsPath(), classes)

        return result

    def compile(self, net: keras.Model, opt: keras.optimizers.Optimizer, loss:str=None)->keras.Model:
        net.compile(opt, None, None)
        return net

    def evaluateAll(self,ds, fold:int,stage=-1,negatives="real",ttflips=None):
        folds = self.kfold(ds, range(0, len(ds)))
        vl, vg, test_g = folds.generator(fold, False,negatives=negatives,returnBatch=True)
        indexes = folds.sampledIndexes(fold, False, negatives)
        m = self.load_model(fold, stage)
        num=0
        with tqdm.tqdm(total=len(indexes), unit="files", desc="segmentation of validation set from " + str(fold)) as pbar:
            try:
                for f in test_g():
                    if num>=len(indexes): break
                    x, y, b = f
                    z = self.predict_on_batch(m,ttflips,b)
                    ids=[]
                    augs=[]
                    for i in range(0,len(z)):
                        if num >= len(indexes): break
                        orig=b.images[i]
                        num = num + 1
                        ma=z[i]
                        id=b.data[i]
                        segmentation_maps_aug = [imgaug.SegmentationMapOnImage(ma, ma.shape)]
                        augmented = imgaug.augmenters.Scale(
                                    {"height": orig.shape[0], "width": orig.shape[1]}).augment_segmentation_maps(segmentation_maps_aug)
                        ids.append(id)
                        augs=augs+augmented

                    res=imgaug.Batch(images=b.images,data=ids,segmentation_maps=b.segmentation_maps)
                    res.predicted_maps_aug=augs
                    yield res
                    pbar.update(len(ids))
            finally:
                vl.terminate()
                vg.terminate()
        pass

    def get_eval_batch(self)->int:
        return self.inference_batch

    def load_writeable_dataset(self, ds, path)->DataSet:
        resName = (ds.name if hasattr(ds, "name") else "") + "_predictions"
        result = MMdetWritableDS(ds, resName, path, self.withMask(), self.threshold)
        return result

    def create_writeable_dataset(self, dataset:DataSet, dsPath:str)->WriteableDataSet:
        resName = (dataset.name if hasattr(dataset, "name") else "") + "_predictions"
        result = MMdetWritableDS(dataset, resName, dsPath, self.withMask(), self.threshold)
        return result

    def predict_to_directory(self, spath, tpath,fold=0, stage=0, limit=-1, batchSize=32,binaryArray=False,ttflips=False):
        generic.ensure(tpath)
        with tqdm.tqdm(total=len(generic.dir_list(spath)), unit="files", desc="segmentation of images from " + str(spath) + " to " + str(tpath)) as pbar:
            for v in self.predict_on_directory(spath, fold=fold, stage=stage, limit=limit, batch_size=batchSize, ttflips=ttflips):
                b:imgaug.Batch=v;
                for i in range(len(b.data)):
                    id=b.data[i];
                    entry = self.toEntry(b, i)
                    if isinstance(tpath, datasets.ConstrainedDirectory):
                        tp=tpath.path
                    else:
                        tp=tpath
                    p = os.path.join(tp, id[0:id.index('.')] + ".npy")
                    save(p,entry)

                pbar.update(batchSize)

    def toEntry(self, b, i):
        bboxes = b.bounding_boxes_unaug[i]
        if self.withMask():
            masks = b.segmentation_maps_unaug[i]
            entry = (bboxes, masks)
        else:
            entry = bboxes
        return entry

    def predict_in_directory(self, spath, fold, stage,cb, data,limit=-1, batchSize=32,ttflips=False):
        with tqdm.tqdm(total=len(generic.dir_list(spath)), unit="files", desc="segmentation of images from " + str(spath)) as pbar:
            for v in self.predict_on_directory(spath, fold=fold, stage=stage, limit=limit, batch_size=batchSize, ttflips=ttflips):
                b:imgaug.Batch=v;
                for i in range(len(b.data)):
                    id=b.data[i];
                    entry = self.toEntry(b, i)
                    cb(id,entry,data)
                pbar.update(batchSize)

    def createAnsambleModel(self, mdl):
        return mdl[0] #InstanceSegmentationAnsambleModel(mdl)

    def createPredictionsBlend(self, prs):
        return InstanceSegmentationPredictionBlend(prs)

    def predict_on_batch(self, mdl, ttflips, batch):
        #o1 = np.array(batch.images_unaug)
        res = mdl.predict(batch.images_unaug)
        if ttflips == "Horizontal":
            another = imgaug.augmenters.Fliplr(1.0).augment_images(batch.images_unaug)
            res1 = mdl.predict(np.array(another))
            if self.flipPred:
                res1 = imgaug.augmenters.Fliplr(1.0).augment_images(res1)
            res = (res + res1) / 2.0
        elif ttflips:
            res = self.predict_with_all_augs(mdl, ttflips, batch)
        return res

    def withMask(self)->bool:
        if 'data' in self.nativeConfig:
            data = self.nativeConfig.data
            if 'train' in data:
                return data.train.with_mask
            if 'val' in data:
                return data.val.with_mask
            if 'test' in data:
                return data.test.with_mask
        return False

    def update(self,z,res):

        # wm = self.withMask()
        # labels = []
        # bboxes = []
        # masks = [] if wm else None
        # for x in res:
        #     thresholded = applyTresholdToPrediction(x, wm, self.threshold)
        #     labels.append(thresholded[0].tolist())
        #     bboxes.append(thresholded[1].tolist())
        #     if wm:
        #         masks.append(thresholded[2].tolist())
        #
        # z.labels = labels
        # z.bounding_boxes_unaug = bboxes
        # z.segmentation_maps_unaug = masks
        pass

    def inject_task_specific_transforms(self, ds, transforms):
        return ds

#
#
# def parse(path) -> PipelineConfig:
#     cfg = configloader.parse("segmentation", path)
#     cfg.path = path
#     return cfg

class DetectionStage(generic.Stage):

    def add_visualization_callbacks(self, cb, ec, kf):
        # drawingFunction = ec.drawingFunction
        # if drawingFunction == None:
        #     drawingFunction = datasets.draw_test_batch
        # cb.append(DrawResults(self.cfg, kf, ec.fold, ec.stage, negatives=self.negatives, drawingFunction=drawingFunction))
        # if self.cfg.showDataExamples:
        #     cb.append(DrawResults(self.cfg, kf, ec.fold, ec.stage, negatives=self.negatives, train=True, drawingFunction=drawingFunction))
        print("add_visualization_callbacks")

    def unfreeze(self, model):
        set_trainable(model)

    def _doTrain(self, kf, model, ec, cb, kepoch):
        torch.cuda.set_device(0)
        negatives = self.negatives
        fold = ec.fold
        numEpochs = self.epochs
        callbacks = cb
        subsample = ec.subsample
        validation_negatives = self.validation_negatives
        verbose = self.cfg.verbose
        initial_epoch = kepoch

        for item in callbacks:
            if "CSVLogger" in str(item):
                item.set_model(model)
                item.on_train_begin()

                if "ModelCheckpoint" in str(item):
                    checkpoint_cb = item

        if validation_negatives == None:
            validation_negatives = negatives

        train_indexes = kf.sampledIndexes(fold, True, negatives)
        test_indexes = kf.sampledIndexes(fold, False, validation_negatives)

        train_indexes = train_indexes
        test_indexes = test_indexes

        batchSize = self.cfg.imagesPerGpu * self.cfg.gpus
        iterations = len(train_indexes) // (round(subsample * batchSize))
        if kf.maxEpochSize is not None and kf.maxEpochSize < iterations:
            iterations = min(iterations, kf.maxEpochSize)
            train_indexes = train_indexes[:iterations * batchSize]

        trainDS = SubDataSet(kf.ds, train_indexes)
        valDS = SubDataSet(kf.ds, test_indexes)

        v_steps = len(test_indexes) // (round(subsample * kf.batchSize))

        if v_steps < 1: v_steps = 1

        cfg = model.cfg
        train_dataset = DataSetAdapter(ds=trainDS, aug=kf.aug, transforms=kf.transforms, **cfg.data.train)
        CLASSES = model.classes
        train_dataset.CLASSES = CLASSES
        val_dataset = DataSetAdapter(ds=valDS, aug=kf.aug, transforms=kf.transforms, **cfg.data.val)
        val_dataset.CLASSES = CLASSES
        val_dataset1 = DataSetAdapter(ds=valDS, aug=kf.aug, transforms=kf.transforms, **cfg.data.val)
        val_dataset1.CLASSES = CLASSES
        val_dataset.test_mode = True
        if cfg.checkpoint_config is not None:
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__,
                config=cfg.text,
                CLASSES=train_dataset.CLASSES)
            cfg.checkpoint_config.out_dir = os.path.dirname(ec.weightsPath())
            cfg.checkpoint_config.filename_tmpl = os.path.basename(ec.weightsPath())
        logger = get_root_logger(cfg.log_level)
        model.setClasses(train_dataset.CLASSES)

        distributed = False
        # prepare data loaders
        data_loaders = [
            build_dataloader(
                train_dataset,
                cfg.data.imgs_per_gpu,
                cfg.data.workers_per_gpu,
                num_gpus=cfg.gpus,
                dist=distributed),

            build_dataloader(
                val_dataset1,
                1,
                1,
                num_gpus=cfg.gpus,
                dist=distributed)
        ]

        runner = train_detector(
            model.model,
            train_dataset,
            val_dataset,
            cfg,
            distributed=distributed,  # distributed,
            validate=True,  # args_validate,
            logger=logger)

        runner._epoch = initial_epoch
        runner.executionConfig = ec

        cpHooks = list(filter(lambda x: 'CheckpointHook' in str(x), runner.hooks))
        if len(cpHooks) == 0:
            raise ValueError("Checkpoint Hook is expected")

        cpHook = cpHooks[0]
        cpHookIndex = runner.hooks.index(cpHook)
        cpHook1 = CustomCheckpointHook(cpHook)
        runner.hooks[cpHookIndex] = cpHook1

        if self.cfg.resume:
            allBest = self.cfg.info('loss')
            filtered = list(filter(lambda x: x.stage == ec.stage and x.fold == ec.fold, allBest))
            if len(filtered) > 0:
                prevInfo = filtered[0]
                self.lr = prevInfo.lr
                cpHook1.setBest(prevInfo.best)

        dsh = DrawSamplesHook(val_dataset, list(range(min(len(test_indexes), 10))),
                              os.path.join(os.path.dirname(self.cfg.path), "examples"), ec, self.cfg.threshold)
        runner.register_hook(HookWrapper(dsh, toSingleGPUModeBefore, toSingleGPUModeAfter))
        for callback in callbacks:
            if "CSVLogger" in str(callback):
                runner.register_hook(KerasCBWrapper(callback))

        model.model.train()
        runner.run(data_loaders, cfg.workflow, numEpochs)

    def execute(self, kf: datasets.DefaultKFoldedDataSet, model: keras.Model, ec: ExecutionConfig, callbacks=None):
        if 'unfreeze_encoder' in self.dict and self.dict['unfreeze_encoder']:
            self.unfreeze(model)

        if 'unfreeze_encoder' in self.dict and not self.dict['unfreeze_encoder']:
            self.freeze(model)
        if callbacks is None:
            cb = [] + self.cfg.callbacks
        else:
            cb = callbacks
        if self.cfg._reporter is not None:
            if self.cfg._reporter.isCanceled():
                return
            cb.append(ReporterCallback(self.cfg._reporter))
            pass
        prevInfo = None
        if self.cfg.resume:
            allBest = self.cfg.info()
            filtered = list(filter(lambda x: x.stage == ec.stage and x.fold == ec.fold, allBest))
            if len(filtered) > 0:
                prevInfo = filtered[0]
                self.lr = prevInfo.lr

        if self.loss or self.lr:
            self.cfg.compile(model, self.cfg.createOptimizer(self.lr), self.loss)
        if self.initial_weights is not None:
            try:
                model.load_weights(self.initial_weights)
            except:
                z = model.layers[-1].name
                model.layers[-1].name = "tmpName12312"
                model.load_weights(self.initial_weights, by_name=True)
                model.layers[-1].name = z
        if 'callbacks' in self.dict:
            cb = configloader.parse("callbacks", self.dict['callbacks'])
        if 'extra_callbacks' in self.dict:
            cb = cb + configloader.parse("callbacks", self.dict['extra_callbacks'])
        kepoch = -1
        if "logAll" in self.dict and self.dict["logAll"]:
            cb = cb + [AllLogger(ec.metricsPath() + "all.csv")]
        cb.append(KFoldCallback(kf))
        kepoch = self._addLogger(model, ec, cb, kepoch)
        md = self.cfg.primary_metric_mode

        if self.cfg.gpus == 1:

            mcp = keras.callbacks.ModelCheckpoint(ec.weightsPath(), save_best_only=True,
                                                  monitor=self.cfg.primary_metric, mode=md, verbose=1)
            if prevInfo != None:
                mcp.best = prevInfo.best

            cb.append(mcp)

        self.add_visualization_callbacks(cb, ec, kf)
        if self.epochs - kepoch == 0:
            return
        self.loadBestWeightsFromPrevStageIfExists(ec, model)
        self._doTrain(kf, model, ec, cb, kepoch)

        print('saved')
        pass

def train_detector(model,
                   trainDataset,
                   valDataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   logger=None)->Runner:
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        return _dist_train_runner(model, trainDataset, valDataset, cfg, validate=validate)
    else:
        return _non_dist_train_runner(model, trainDataset, valDataset, cfg, validate=validate)

def _dist_train_runner(model, trainDataset, valDataset, cfg, validate=False)->Runner:

    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(model, batch_processor, optimizer, cfg.work_dir,
                    cfg.log_level)

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config,
                                             **fp16_cfg)
    else:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    if validate:
        val_dataset_cfg = valDataset
        eval_cfg = cfg.get('evaluation', {})
        if isinstance(model.module, RPN):
            # TODO: implement recall hooks for other datasets
            runner.register_hook(
                CocoDistEvalRecallHook(val_dataset_cfg, **eval_cfg))
        else:
            dataset_type = getattr(mmdetDatasets, val_dataset_cfg.type)
            if issubclass(dataset_type, CocoDataset):
                runner.register_hook(
                    CocoDistEvalmAPHook(val_dataset_cfg, **eval_cfg))
            else:
                runner.register_hook(
                    DistEvalmAPHook(val_dataset_cfg, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    return runner


def toSingleGPUModeBefore(runner):
    result = {'device_ids': runner.model.device_ids}
    runner.model.device_ids = [torch.cuda.current_device()]
    return result

def toSingleGPUModeAfter(runner, beforeRes):
    runner.model.device_ids = beforeRes['device_ids']

def _non_dist_train_runner(model, trainDataset, valDataset, cfg, validate=False)->Runner:

    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(model, batch_processor, optimizer, cfg.work_dir,
                    cfg.log_level)
    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=False)
    else:
        optimizer_config = cfg.optimizer_config

    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    before = toSingleGPUModeBefore
    after = toSingleGPUModeAfter

    # register eval hooks
    if validate:
        val_dataset_cfg = valDataset
        eval_cfg = cfg.get('evaluation', {})
        if isinstance(model.module, RPN):
            # TODO: implement recall hooks for other datasets
            runner.register_hook(
                HookWrapper(CocoDistEvalRecallHook(val_dataset_cfg, **eval_cfg),before,after))
        else:
            dataset_type = getattr(mmdetDatasets, val_dataset_cfg.type)
            if issubclass(dataset_type, mmdetDatasets.CocoDataset):
                runner.register_hook(
                    HookWrapper(CocoDistEvalmAPHook(val_dataset_cfg, **eval_cfg),before,after))
            else:
                runner.register_hook(
                    HookWrapper(DistEvalmAPHook(val_dataset_cfg, **eval_cfg),before,after))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        weightsPath = cfg.load_from
        if weightsPath is not None:
            if weightsPath.startswith("open-mmlab://"):
                cfgName = weightsPath[len("open-mmlab://"):]
                weightsPath = checkpoint_registry.getPath(cfgName)
            if cfg.resetHeads:
                torchHome = torch.hub._get_torch_home()
                chpName = os.path.basename(weightsPath)[0:(-1)*len(".pth")]
                noHeadWeightsPath = os.path.join(torchHome,f"checkpoints/nohead/{chpName}_nohead.pth")
                if not os.path.exists(noHeadWeightsPath):
                    if isURL(weightsPath):
                        weights = model_zoo.load_url(weightsPath)
                    else:
                        weights = torch.load(weightsPath)
                    weights['state_dict'] = {
                        k: v
                        for k, v in weights['state_dict'].items()
                        if not k.startswith('bbox_head') and not k.startswith('mask_head')
                    }
                    weightsDir = os.path.dirname(noHeadWeightsPath)
                    if not os.path.exists(weightsDir):
                        os.mkdir(weightsDir)
                    torch.save(weights, noHeadWeightsPath)
                weightsPath = noHeadWeightsPath

            runner.load_checkpoint(weightsPath)
    return runner




def setCfgAttr(obj, attrName, value):
    if isinstance(obj, list):
        for x in obj:
            setCfgAttr(x,attrName,value)
    else:
        setattr(obj,attrName,value)
