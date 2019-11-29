import imgaug
import torch
import numpy as np
from mmcv.parallel import DataContainer as DC
from musket_core.utils import save, load
import os
from musket_core.datasets import PredictionBlend, PredictionItem, DataSet, CompressibleWriteableDS
from musket_core.losses import SMOOTH

from mmdet.apis import show_result
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.utils import to_tensor, random_scale
from mmdet.core.post_processing.merge_augs import merge_aug_bboxes, merge_aug_masks
from typing import Callable
import networkx as nx
import imageio
from mmdetection_pipeline.callbacks import imdraw_det_bboxes


class MMdetWritableDS(CompressibleWriteableDS):

    def __init__(self,orig,name,dsPath, withMasks, threshold=0.5, count = 0,asUints=True,scale=255):
        super().__init__(orig,name,dsPath, count,False,scale)
        self.withMasks = withMasks
        self.threshold = threshold

    # def __getitem__(self, item):
    #     res = super().__getitem__(item)
    #     if isinstance(item, slice):
    #         for pi in res:
    #             self.processPredictionItem(pi)
    #     else:
    #         self.processPredictionItem(res)
    #     return res
    #
    # def processPredictionItem(self, pi):
    #     pred = pi.prediction
    #     tresholdedPrediction = applyTresholdToPrediction(pred,self.withMasks,self.threshold)
    #     pi.prediction = tresholdedPrediction

    def saveItem(self, path:str, item):
        wm = self.withMasks

        dire = os.path.dirname(path)
        if not os.path.exists(dire):
            os.mkdir(dire)

        labels = item[0]
        probabilities = item[1]
        bboxes = item[2]
        if wm:
            masks = item[3]
            if self.asUints:
                if self.scale <= 255:
                    masks = (masks * self.scale).astype(np.uint8)
                else:
                    masks = (masks * self.scale).astype(np.uint16)

            np.savez_compressed(file=path, labels=labels, probabilities=probabilities, bboxes=bboxes, masks=masks)
        else:
            np.savez_compressed(file=path, labels=labels, probabilities=probabilities, bboxes=bboxes)

    def loadItem(self, path:str):
        npzFile = np.load(path,allow_pickle=True)
        labels = npzFile['labels']
        probabilities = npzFile['probabilities']
        bboxes = npzFile['bboxes']
        if self.withMasks:
            masks = npzFile['masks']
            if self.asUints:
                masks=masks.astype(np.float32)/self.scale

            return (labels, probabilities, bboxes, masks)
        else:
            return (labels, probabilities, bboxes)

class MusketPredictionItemWrapper(object):

    def __init__(self, ind: int, ds: DataSet):
        self.ind = ind
        self.ds = ds
        self.callbacks: [Callable[[PredictionItem], None]] = []

    def getPredictionItem(self) -> PredictionItem:
        predictionItem = self.ds[self.ind]
        for x in self.callbacks:
            x(predictionItem)
        return predictionItem

    def addCallback(self, cb: Callable[[PredictionItem], None]):
        self.callbacks.append(cb)


class MusketInfo(object):

    def __init__(self, predictionItemWrapper: MusketPredictionItemWrapper):
        self.initialized = False
        self.predictionItemWrapper = predictionItemWrapper
        self.predictionItemWrapper.addCallback(self.initializer)

    def checkInit(self):
        if not self.initialized:
            self.getPredictionItem()

    def getPredictionItem(self) -> PredictionItem:
        result = self.predictionItemWrapper.getPredictionItem()
        return result

    def initializer(self, pi: PredictionItem):
        self._initializer(pi)
        self.initialized = True

    def _initializer(self, pi: PredictionItem):
        raise ValueError("Not implemented")

    def dispose(self):
        self._free()
        self.initialized = False

    def _free(self):
        raise ValueError("Not implemented")


class MusketImageInfo(MusketInfo):

    def __init__(self, piw: MusketPredictionItemWrapper):
        super().__init__(piw)
        self.ann = MusketAnnotationInfo(piw)
        self.img = None
        self.id = None

    def image(self) -> np.ndarray:
        pi = self.getPredictionItem()
        self.img = pi.x
        self.id = pi.id
        return self.img

    def __getitem__(self, key):
        if key == "height":
            self.checkInit()
            return self.height
        elif key == "width":
            self.checkInit()
            return self.width
        elif key == "ann":
            return self.ann
        elif key == "file_name" or key == "id":
            return self.id
        elif key == 'scale_factor':
            return 1.0
        elif key == 'flip':
            return False
        elif key == 'img_shape':
            return (self.height, self.width)
        return None

    def _initializer(self, pi: PredictionItem):
        img = pi.x
        self.width = img.shape[1]
        self.height = img.shape[0]

    def _free(self):
        self.img = None
        self.ann._free()


class MusketAnnotationInfo(MusketInfo):

    def _initializer(self, pi: PredictionItem):
        y = pi.y
        if y is not None:
            self.labels = y[0]
            self.bboxes = y[1]
            self.masks = y[2] if len(y) > 2 else None
        self.bboxes_ignore = np.zeros(shape=(0, 4), dtype=np.float32)
        self.labels_ignore = np.zeros((0), dtype=np.int64)

    def __getitem__(self, key):
        if key == "bboxes":
            self.checkInit()
            return self.bboxes
        elif key == "labels":
            self.checkInit()
            return self.labels
        elif key == "bboxes_ignore":
            self.checkInit()
            return self.bboxes_ignore
        elif key == 'labels_ignore':
            self.checkInit()
            return self.labels_ignore
        elif key == "masks":
            self.checkInit()
            return self.masks
        return None

    def _free(self):
        self.masks = None


class DataSetAdapter(CustomDataset):

    def __init__(self, ds: DataSet, aug=None, transforms=None, **kwargs):
        self.ds = ds
        self.aug = aug
        self.transforms = transforms
        args = kwargs.copy()
        if 'type' in args:
            args.pop('type')
        self.type = 'VOCDataset'
        self.img_infos = []
        super().__init__(**args)

        self.with_crowd = True

    def __len__(self):
        return len(self.ds)

    def augmentor(self, isTrain) -> imgaug.augmenters.Augmenter:
        allAug = []
        if isTrain:
            allAug = allAug + self.aug
        allAug = allAug + self.transforms
        aug = imgaug.augmenters.Sequential(allAug)
        return aug

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def load_annotations(self, ann_file):
        img_infos = []
        for idx in range(len(self.ds)):
            piw = MusketPredictionItemWrapper(idx, self.ds)
            img_info = MusketImageInfo(piw)
            img_infos.append(img_info)
        return img_infos

    def _filter_imgs(self, min_size=32):
        print("filter_images")
        return list(range(len(self)))

    def prepare_train_img(self, idx):

        try:
            img_info = self.img_infos[idx]
            # load image
            img = img_info.image()  # mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
            # load proposals if necessary
            if self.proposals is not None:
                proposals = self.proposals[idx][:self.num_max_proposals]
                # TODO: Handle empty proposals properly. Currently images with
                # no proposals are just ignored, but they can be used for
                # training in concept.
                if len(proposals) == 0:
                    return None
                if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                    raise AssertionError(
                        'proposals should have shapes (n, 4) or (n, 5), '
                        'but found {}'.format(proposals.shape))
                if proposals.shape[1] == 5:
                    scores = proposals[:, 4, None]
                    proposals = proposals[:, :4]
                else:
                    scores = None

            ann = self.get_ann_info(idx)
            gt_bboxes = ann['bboxes']
            gt_labels = ann['labels']
            gt_masks = None
            gt_bboxes_ignore = None
            if self.with_mask:
                gt_masks = ann['masks']
            if self.with_crowd:
                gt_bboxes_ignore = ann['bboxes_ignore']

            # dumpData(f"d:/ttt/{img_info.id}_tmp_bbox.jpg", f"d:/ttt/{img_info.id}_tmp_mask.jpg", img, gt_labels-1,
            #          gt_bboxes, gt_masks, self.CLASSES)

            img, gt_bboxes, gt_masks, gt_bboxes_ignore = self.applyAugmentations(img, gt_bboxes, gt_masks,
                                                                                 gt_bboxes_ignore, True)

            # dumpData(f"d:/ttt/{img_info.id}_tmp_bbox_aug.jpg", f"d:/ttt/{img_info.id}_tmp_mask_aug.jpg", img, gt_labels-1,
            #          gt_bboxes, gt_masks, self.CLASSES)

            # skip the image if there is no valid gt bbox
            if len(gt_bboxes) == 0:
                return None

            # extra augmentation
            if self.extra_aug is not None:
                # img = self.extra_aug(img)
                img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes,
                                                           gt_labels)

            # apply transforms
            flip = True if np.random.rand() < self.flip_ratio else False
            # randomly sample a scale

            img_scale = random_scale(self.img_scales, self.multiscale_mode)
            img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
            img = img.copy()
            if self.with_seg:
                # gt_seg = mmcv.imread(
                #     osp.join(self.seg_prefix, img_info['file_name'].replace(
                #         'jpg', 'png')),
                #     flag='unchanged')
                # gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
                # gt_seg = mmcv.imrescale(
                #     gt_seg, self.seg_scale_factor, interpolation='nearest')
                # gt_seg = gt_seg[None, ...]
                pass
            if self.proposals is not None:
                proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                                flip)
                proposals = np.hstack(
                    [proposals, scores]) if scores is not None else proposals
            gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                            flip)
            if self.with_crowd:
                gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                       scale_factor, flip)
            if self.with_mask:
                gt_masks = self.mask_transform(gt_masks, pad_shape,
                                               scale_factor, flip)

            ori_shape = (img_info['height'], img_info['width'], 3)
            img_meta = dict(
                id=img_info['id'],
                ori_shape=ori_shape,
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)

            # imgt = img.transpose(1, 2, 0)
            # imgt -= np.min(imgt)
            # imgt *= (255 / np.max(imgt))
            # imgt = imgt.astype(np.uint8)
            # dumpData(f"d:/ttt/{img_info.id}_tmp_bbox_aug1.jpg", f"d:/ttt/{img_info.id}_tmp_mask_aug1.jpg", imgt,
            #          gt_labels - 1,
            #          gt_bboxes, gt_masks, self.CLASSES)

            data = dict(
                img=DC(to_tensor(img), stack=True),
                img_meta=DC(img_meta, cpu_only=True),
                gt_bboxes=DC(to_tensor(gt_bboxes)))
            if self.proposals is not None:
                data['proposals'] = DC(to_tensor(proposals))
            if self.with_label:
                data['gt_labels'] = DC(to_tensor(gt_labels))
            if self.with_crowd:
                data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
            if self.with_mask:
                data['gt_masks'] = DC(gt_masks, cpu_only=True)
            # if self.with_seg:
            #     data['gt_semantic_seg'] = DC(to_tensor(gt_seg), stack=True)

            return data
        finally:
            img_info.dispose()

    def applyAugmentations(self, img, gt_bboxes, gt_masks, gt_bboxes_ignore, isTrain):

        bboxesDType = gt_bboxes.dtype
        masksDType = gt_masks.dtype

        bbox_split = len(gt_bboxes)
        all_bboxes = np.concatenate((gt_bboxes, gt_bboxes_ignore), axis=0)

        imgaugBBoxes = [imgaug.BoundingBox(x[0], x[1], x[2], x[3]) for x in all_bboxes]
        imgaugBBoxesOnImage = imgaug.BoundingBoxesOnImage(imgaugBBoxes, img.shape)

        imgaugSegmentationMapsOnImage = imgaug.SegmentationMapsOnImage(gt_masks.transpose(1, 2, 0),
                                                                       tuple(gt_masks.shape[1:]))

        batch = imgaug.Batch(images=[img], segmentation_maps=imgaugSegmentationMapsOnImage,
                             bounding_boxes=imgaugBBoxesOnImage)
        aug = self.augmentor(isTrain)
        augmentedBatch = aug.augment_batch(batch)

        img_aug = augmentedBatch.images_aug[0]
        all_bboxes_aug = [np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2], dtype=bboxesDType) for bbox in
                          augmentedBatch.bounding_boxes_aug.bounding_boxes]
        all_bboxes_aug = np.array(all_bboxes_aug, dtype=bboxesDType)
        gt_bboxes_aug = all_bboxes_aug[:bbox_split]
        gt_bboxes_ignore_aug = all_bboxes_aug[bbox_split:]

        masks_aug = augmentedBatch.segmentation_maps_aug.arr.transpose(2, 0, 1).astype(masksDType)

        return img_aug, gt_bboxes_aug, masks_aug, gt_bboxes_ignore_aug

    def augmentBoundingBoxes(self, aug, gt_bboxes, img):
        imgaugBBoxes = [imgaug.BoundingBox(x[0], x[1], x[2], x[3]) for x in gt_bboxes]
        imgaugBBoxesOnImage = imgaug.BoundingBoxesOnImage(imgaugBBoxes, img.shape)
        imgaugBBoxesOnImageAug = aug.augment_bounding_boxes(imgaugBBoxesOnImage)
        dtype = gt_bboxes.dtype
        shape = gt_bboxes.shape
        gt_bboxes = [np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2], dtype=dtype) for bbox in
                     imgaugBBoxesOnImageAug.bounding_boxes]
        gt_bboxes = np.array(gt_bboxes, dtype=dtype).reshape(shape)
        return gt_bboxes

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        try:
            img_info = self.img_infos[idx]
            img = img_info.image()  # mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
            if self.proposals is not None:
                proposal = self.proposals[idx][:self.num_max_proposals]
                if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                    raise AssertionError(
                        'proposals should have shapes (n, 4) or (n, 5), '
                        'but found {}'.format(proposal.shape))
            else:
                proposal = None

            def prepare_single(img, scale, flip, proposal=None):
                _img, img_shape, pad_shape, scale_factor = self.img_transform(
                    img, scale, flip, keep_ratio=self.resize_keep_ratio)
                _img = to_tensor(_img)
                _img_meta = dict(
                    ori_shape=(img_info['height'], img_info['width'], 3),
                    img_shape=img_shape,
                    pad_shape=pad_shape,
                    scale_factor=scale_factor,
                    flip=flip)
                if proposal is not None:
                    if proposal.shape[1] == 5:
                        score = proposal[:, 4, None]
                        proposal = proposal[:, :4]
                    else:
                        score = None
                    _proposal = self.bbox_transform(proposal, img_shape,
                                                    scale_factor, flip)
                    _proposal = np.hstack(
                        [_proposal, score]) if score is not None else _proposal
                    _proposal = to_tensor(_proposal)
                else:
                    _proposal = None
                return _img, _img_meta, _proposal

            imgs = []
            img_metas = []
            proposals = []
            for scale in self.img_scales:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, False, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
                if self.flip_ratio > 0:
                    _img, _img_meta, _proposal = prepare_single(
                        img, scale, True, proposal)
                    imgs.append(_img)
                    img_metas.append(DC(_img_meta, cpu_only=True))
                    proposals.append(_proposal)
            data = dict(img=imgs, img_meta=img_metas)
            if self.proposals is not None:
                data['proposals'] = proposals
            return data
        finally:
            img_info.dispose()

    def show(self, img, result):
        show_result(img, result, self.CLASSES)



class InstanceSegmentationPredictionBlend(PredictionBlend):

    def blend_predictions(self, item):

        LABEL_INDEX = 0
        CONF_INDEX = 1
        BBOX_INDEX = 2
        MASK_INDEX = 3
        byClazz = {}

        items = [ds[item] for ds in self.predictions]

        for dsInd in range(len(items)):
            pi = items[dsInd]
            pred = pi.prediction
            labels = pred[LABEL_INDEX]
            masks = pred[MASK_INDEX]
            for i in range(len(labels)):
                label = labels[i]
                mask = masks[i]
                if np.max(mask) == 0:
                    continue
                key = str(label)
                if not key in byClazz:
                    arr = []
                    byClazz[key] = arr
                else:
                    arr = byClazz[key]

                arr.append((dsInd,i,label))


        resultLabels = []
        resultConfidences = []
        resultBBoxes = []
        resultMasks = []
        for key in byClazz:
            arr = byClazz[key]
            l = len(arr)
            iouMatrix = np.eye(l,dtype=np.float)
            for i in range(l):
                for j in range(i+1,l):
                    dsInd_i = arr[i][0]
                    objInd_i = arr[i][1]
                    dsInd_j = arr[j][0]
                    objInd_j = arr[j][1]
                    # mask_i = items[dsInd_i].prediction[MASK_INDEX][objInd_i]
                    # mask_j = items[dsInd_j].prediction[MASK_INDEX][objInd_j]

                    bbox_i = items[dsInd_i].prediction[BBOX_INDEX][objInd_i]
                    bbox_j = items[dsInd_j].prediction[BBOX_INDEX][objInd_j]

                    intersection = np.zeros(bbox_i.shape,dtype = bbox_i.dtype)
                    intersection[:2] = np.maximum(bbox_i[:2], bbox_j[:2])
                    intersection[2:] = np.minimum(bbox_i[2:], bbox_j[2:])
                    union = np.zeros(bbox_i.shape, dtype=bbox_i.dtype)
                    union[:2] = np.minimum(bbox_i[:2], bbox_j[:2])
                    union[2:] = np.maximum(bbox_i[2:], bbox_j[2:])
                    intersectionArea = (intersection[2]-intersection[0])*(intersection[3]-intersection[1])
                    unionArea = (union[2] - union[0]) * (union[3] - union[1])
                    iou = (intersectionArea + SMOOTH) / (unionArea + SMOOTH)

                    # mask_i_bin = mask_i > 0
                    # mask_j_bin = mask_j > 0
                    # intersection = (mask_i_bin & mask_j_bin).sum()
                    # union = (mask_i_bin | mask_j_bin).sum()
                    # iou = (intersection + SMOOTH) / (union + SMOOTH)

                    iouMatrix[i, j] = iou
                    iouMatrix[j, i] = iou

            iouMatrix_bin = iouMatrix > 0.7
            graph = nx.Graph(iouMatrix_bin)
            components = nx.connected_components(graph)
            for componentAsSet in components:
                componentAsList = [arr[x] for x in componentAsSet]

                label = componentAsList[0][2]
                dsInd = componentAsList[0][0]
                objInd = componentAsList[0][1]
                pred = items[dsInd].prediction
                conf = pred[CONF_INDEX][objInd]
                mask = pred[MASK_INDEX][objInd]
                bbox = pred[BBOX_INDEX][objInd]

                mergedConf = conf
                mergedBBox = np.copy(bbox)
                mergedMask = np.copy(mask)
                piecesCount = len(componentAsList)
                if piecesCount > 1:
                    for i in range(1,piecesCount):
                        dsInd = componentAsList[i][0]
                        objInd = componentAsList[i][1]

                        pred = items[dsInd].prediction
                        conf = pred[CONF_INDEX][objInd]
                        mask = pred[MASK_INDEX][objInd]
                        bbox = pred[BBOX_INDEX][objInd]

                        mergedConf += conf
                        mergedBBox[:2] = np.minimum(mergedBBox[:2],bbox[:2])
                        mergedBBox[2:] = np.maximum(mergedBBox[2:], bbox[2:])
                        mergedMask += mask

                    mergedConf /= piecesCount
                    mergedMask = (mergedMask.astype(np.float) / piecesCount + 0.5).astype(np.int)

                resultLabels.append(label)
                resultConfidences.append(mergedConf)
                resultBBoxes.append(mergedBBox)
                resultMasks.append(mergedMask)

        resultLabels = np.array(resultLabels,dtype=np.int)
        resultConfidences = np.array(resultConfidences, dtype=np.float)
        resultBBoxes = np.array(resultBBoxes, dtype=np.float)
        resultMasks = np.array(resultMasks)

        result = (resultLabels, resultConfidences, resultBBoxes, resultMasks)
        return result


def applyMasksToImage(img, gt_masks, gt_labels, numColors):
    masksShape = list(img.shape[:2]) + [1]
    objColor = 1
    gtMasksArr = np.zeros(masksShape, dtype=np.int)
    for i in range(len(gt_labels)):
        l = gt_labels[i]
        gtm = gt_masks[i]
        gtMasksArr[gtm > 0] = objColor
        objColor = 1 + (objColor + 1) % (numColors - 1)
    gtMaskImg = imgaug.SegmentationMapOnImage(gtMasksArr, img.shape).draw_on_image(img)[0]
    return gtMaskImg


def dumpData(bboxesPath, masksPath, img, labels, bboxes, masks, classes):
    numColors = len(classes)
    gtMaskImg = applyMasksToImage(img, masks, labels, numColors)
    imageio.imwrite(masksPath, gtMaskImg)
    gtBBoxImg = imdraw_det_bboxes(img.copy(), bboxes, labels - 1, class_names=classes)
    imageio.imwrite(bboxesPath, gtBBoxImg)
