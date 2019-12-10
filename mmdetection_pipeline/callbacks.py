from collections import OrderedDict
import mmcv
import cv2
import imageio
from mmcv.visualization.color import color_val
from mmcv.runner import Runner, Hook
from mmcv.runner.hooks.checkpoint import CheckpointHook
from mmcv.parallel import scatter, collate
import imgaug

import numpy as np
import keras
from musket_core.generic_config import ExecutionConfig
import os
import torch
import torch.distributed
from mmdetection_pipeline.conversions import convertMMDETModelOutput, applyTresholdToPrediction


class DrawSamplesHook(Hook):

    def __init__(self, dataset, indices, dstFolder, ec:ExecutionConfig, threshold=0.5):
        self.dataset = dataset
        self.indices = indices
        self.dstFolder = dstFolder
        self.exampleWidth = 800
        self.threshold = threshold
        self.ec=ec

    def after_train_epoch(self, runner:Runner):
        ec = runner.executionConfig
        runner.model.eval()
        results = [None for _ in range(len(self.indices))]
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.indices))
        for idx in self.indices:
            pi = self.dataset.ds[idx]
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                pred = runner.model(
                    return_loss=False, rescale=True, **data_gpu)

            withMasks = isinstance(pred,tuple)
            result = convertMMDETModelOutput(pred,withMasks)
            result = applyTresholdToPrediction(result,withMasks,self.threshold)
            results[idx] = (result, pi)

            #batch_size = runner.world_size
            if runner.rank == 0:
                prog_bar.update()


        gtImages = []
        predImages = []
        gtMaskedImages=[]
        predMaskedImages=[]

        classNames = self.dataset.CLASSES
        for r in results:
            imgOrig = r[1].x
            scale = self.exampleWidth / imgOrig.shape[1]
            newY = self.exampleWidth
            newX = int(imgOrig.shape[0] * scale)
            img = imgaug.imresize_single_image(imgOrig,(newX, newY), 'cubic')

            gtLabels = r[1].y[0]-1
            gtBboxesRaw = r[1].y[1]
            gtBboxes = gtBboxesRaw * scale

            result = r[0]
            labels = result[0]
            bboxes = result[1]
            if len(result) == 3:
                segm_result = result[2]

            bboxes *= scale
            numColors = len(imgaug.SegmentationMapsOnImage.DEFAULT_SEGMENT_COLORS)
            if segm_result is not None:
                masksShape = list(imgOrig.shape[:2]) + [1]
                gtMasks = r[1].y[2]
                maskIndices = set()
                objColor = 1
                if gtMasks is not None:
                    gtMasksArr = np.zeros(masksShape, dtype=np.int)
                    for i in range(len(gtLabels)):
                        l = gtLabels[i]
                        gtm = gtMasks[i]
                        gtMasksArr[gtm > 0] = objColor
                        objColor = 1 + (objColor + 1) % (numColors-1)

                predMasksArr = np.zeros(masksShape, dtype=np.int)
                objColor = 1
                for x in segm_result:
                    predMasksArr[x > 0] = objColor
                    objColor = 1 + (objColor + 1) % (numColors - 1)

                #CustomSegmentationMapOnImage(np.transpose(gtMasks, axes=(1,2,0)), imgOrig.shape).draw_on_image(imgOrig)
                gtMaskImg = imgaug.SegmentationMapOnImage(gtMasksArr, imgOrig.shape).draw_on_image(imgOrig)[0]
                predMaskImg = imgaug.SegmentationMapOnImage(predMasksArr, imgOrig.shape).draw_on_image(imgOrig)[0]
                #predMaskImg = imgaug.HeatmapsOnImage(predMasksArr,imgOrig.shape).draw_on_image(imgOrig)
                gtMaskedImages.append(imgaug.imresize_single_image(gtMaskImg, (newX, newY), 'cubic'))
                predMaskedImages.append(imgaug.imresize_single_image(predMaskImg,(newX, newY), 'cubic'))

            predImg = imdraw_det_bboxes(img.copy(),bboxes,labels,class_names=classNames)
            gtImg = imdraw_det_bboxes(img.copy(), gtBboxes, gtLabels, class_names=classNames)
            gtImages.append(gtImg)
            predImages.append(predImg)

        gtImg = np.concatenate(gtImages,axis=0)
        predImg = np.concatenate(predImages, axis=0)
        exampleImg = np.concatenate([gtImg, predImg], axis=1)

        if len(gtMaskedImages) > 0:
            gtMaskImg = np.concatenate(gtMaskedImages, axis=0)
            exampleImg = np.concatenate([exampleImg, gtMaskImg], axis=1)

        if len(predMaskedImages) > 0:
            predMaskImg = np.concatenate(predMaskedImages, axis=0)
            exampleImg = np.concatenate([exampleImg, predMaskImg], axis=1)

        epoch = runner.epoch
        imgPath = f"{epoch}.jpg"
        imFolder = os.path.join(self.dstFolder,f"{ec.fold}/{ec.stage}")
        if not os.path.exists(imFolder):
            os.makedirs(imFolder)
        out_file = os.path.join(imFolder,imgPath)
        imageio.imwrite(out_file, exampleImg)



def imdraw_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      thickness=1,
                      font_scale=0.5,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5

    if score_thr > 0:
        assert bboxes.shape[1] == 4
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = str(class_names[
            label]) if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(bbox[-1])
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

    return img



class HookWrapper(Hook):

    def __init__(self, hook:Hook, before, after):
        self.hook = hook
        self.before = before
        self.after = after

    def before_run(self, runner):
        beforeRes = self.before(runner)
        self.hook.before_run(runner)
        self.after(runner, beforeRes)

    def after_run(self, runner):
        beforeRes = self.before(runner)
        self.hook.after_run(runner)
        self.after(runner, beforeRes)

    def before_epoch(self, runner):
        beforeRes = self.before(runner)
        self.hook.before_epoch(runner)
        self.after(runner, beforeRes)

    def after_epoch(self, runner):
        beforeRes = self.before(runner)
        self.hook.after_epoch(runner)
        self.after(runner, beforeRes)

    def before_iter(self, runner):
        beforeRes = self.before(runner)
        self.hook.before_iter(runner)
        self.after(runner, beforeRes)

    def after_iter(self, runner):
        beforeRes = self.before(runner)
        self.hook.after_iter(runner)
        self.after(runner, beforeRes)

    def before_train_epoch(self, runner):
        beforeRes = self.before(runner)
        self.hook.before_train_epoch(runner)
        self.after(runner, beforeRes)

    def before_val_epoch(self, runner):
        beforeRes = self.before(runner)
        self.hook.before_val_epoch(runner)
        self.after(runner, beforeRes)

    def after_train_epoch(self, runner):
        beforeRes = self.before(runner)
        self.hook.after_train_epoch(runner)
        self.after(runner, beforeRes)

    def after_val_epoch(self, runner):
        beforeRes = self.before(runner)
        self.hook.after_val_epoch(runner)
        self.after(runner, beforeRes)

    def before_train_iter(self, runner):
        beforeRes = self.before(runner)
        self.hook.before_train_iter(runner)
        self.after(runner, beforeRes)

    def before_val_iter(self, runner):
        beforeRes = self.before(runner)
        self.hook.before_val_iter(runner)
        self.after(runner, beforeRes)

    def after_train_iter(self, runner):
        beforeRes = self.before(runner)
        self.hook.after_train_iter(runner)
        self.after(runner, beforeRes)

    def after_val_iter(self, runner):
        beforeRes = self.before(runner)
        self.hook.after_val_iter(runner)
        self.after(runner, beforeRes)

    def every_n_epochs(self, runner, n):
        beforeRes = self.before(runner)
        result = self.hook.every_n_epochs(runner,n)
        self.after(runner, beforeRes)
        return result

    def every_n_inner_iters(self, runner, n):
        beforeRes = self.before(runner)
        result = self.hook.every_n_inner_iters(runner,n)
        self.after(runner, beforeRes)
        return result

    def every_n_iters(self, runner, n):
        beforeRes = self.before(runner)
        result = self.hook.every_n_iters(runner,n)
        self.after(runner, beforeRes)
        return result

    def end_of_epoch(self, runner):
        beforeRes = self.before(runner)
        result = self.hook.end_of_epoch(runner)
        self.after(runner, beforeRes)
        return result


class KerasCBWrapper(Hook):

    def __init__(self, cb:keras.callbacks.Callback):
        self.cb = cb

    def before_run(self, runner):
        self.cb.on_train_begin()

    def after_run(self, runner):
        self.cb.on_train_end()

    def before_epoch(self, runner):
        self.cb.on_epoch_begin(runner.epoch, None)

    def after_epoch(self, runner):
        self.cb.on_epoch_end(runner.epoch, None)

    def before_train_epoch(self, runner):
        self.cb.on_epoch_begin(runner.epoch, None)

    def before_val_epoch(self, runner):
        self.cb.on_epoch_begin(runner.epoch, None)

    def after_train_epoch(self, runner):
        self.train_logs = log(runner)
        #self.cb.on_epoch_end(runner.epoch, logs)

    def after_val_epoch(self, runner):
        self.val_logs = log(runner)
        if 'lr' in self.val_logs:
            del self.val_logs['lr']
        logs = {}
        for x in self.train_logs:
            x1 = x
            if x == 'mAP':
                x1 = 'val_mAP'
            logs[x1] = self.train_logs[x]
        for x in self.val_logs:
            logs[f"val_{x}"] = self.val_logs[x]
        self.cb.on_epoch_end(runner.epoch, logs)

    def end_of_epoch(self, runner):
        self.cb.on_epoch_end(runner.epoch, None)


def log(runner):
    runner.log_buffer.average()
    log_dict = OrderedDict()
    # training mode if the output contains the key "time"
    #log_dict['epoch'] = runner.epoch
    #mode = 'train' if 'time' in runner.log_buffer.output else 'val'
    #log_dict['mode'] = mode
    #log_dict['iter'] = runner.inner_iter + 1
    # only record lr of the first param group
    log_dict['lr'] = runner.current_lr()[0]
    # if mode == 'train':
    #     log_dict['time'] = runner.log_buffer.output['time']
    #     log_dict['data_time'] = runner.log_buffer.output['data_time']
    for name, val in runner.log_buffer.output.items():
        if name in ['time', 'data_time']:
            continue
        log_dict[name] = val
    log_dict['loss'] = get_loss(runner)
    return log_dict


def get_loss(runner):
    return 0.0 + runner.outputs['loss'].data.cpu().numpy()


class CustomCheckpointHook(Hook):

    def __init__(self, delegate: CheckpointHook):
        self.delegate = delegate
        self.best = 100.0

    def setBest(self, value):
        self.best = value

    def after_val_epoch(self, runner):
        loss = get_loss(runner)
        if loss < self.best:
            self.best = loss
            self.delegate.after_train_epoch(runner)

