from musket_core.metrics import final_metric
from musket_core.model import FoldsAndStages
from musket_core.datasets import DataSet
from mmdet.core.evaluation.mean_ap import eval_map
from mmdetection_pipeline.conversions import applyTresholdToPrediction, convertMMDETModelOutput
import numpy as np

@final_metric
def mmdet_mAP_bbox(fas:FoldsAndStages, ds:DataSet):
    classes = ds.root().meta()['CLASSES']
    numClasses = len(classes)

    gt_bboxes = []
    gt_labels = []
    gt_ignore = []
    results = []
    for pi in ds:
        item_gt_labels = pi.y[0]
        item_gt_bboxes = pi.y[1]
        gt_bboxes.append(item_gt_bboxes)
        gt_labels.append(item_gt_labels)

        pred = pi.prediction
        # withMasks = fas.wrapped.withMask()
        # threshold = fas.wrapped.threshold
        # pred = convertMMDETModelOutput(pred, withMasks)
        # pred = applyTresholdToPrediction(pred, withMasks, threshold)

        item_pred_labels = pred[0]
        item_pred_bboxes = pred[2]

        gt_ignore.append(np.zeros((len(item_gt_bboxes,)),dtype=np.bool))

        res_bboxes = np.concatenate([item_pred_bboxes,np.ones((len(item_pred_bboxes),1),dtype=np.float32)],axis=1)

        result = []
        for i in range(numClasses):
            result.append([])

        for i in range(len(item_pred_labels)):
            label = int(item_pred_labels[i]+0.5)
            bbox = res_bboxes[i]
            result[label].append(bbox)

        for i in range(numClasses):
            if len(result[i]) == 0:
                result[i] = np.zeros((0,5),dtype=np.float32)
            else:
                result[i] = np.concatenate([np.expand_dims(x,axis=0) for x in result[i]],axis=0)

        results.append(result)


    # If the dataset is VOC2007, then use 11 points mAP evaluation.
    mean_ap, eval_results = eval_map(
        results,
        gt_bboxes,
        gt_labels,
        gt_ignore=gt_ignore,
        scale_ranges=None,
        iou_thr=0.5,
        dataset=classes,
        print_summary=True)

    return mean_ap


@final_metric
def mAP_masks(fas:FoldsAndStages, ds:DataSet):
    classes = ds.root().meta()['CLASSES']
    numClasses = len(classes)

    clazz = {}
    clazz_counts = {}
    for pi in ds:

        item_gt_labels = pi.y[0]
        item_gt_masks = pi.y[2]
        gt = []
        for i in range(numClasses):
            gt.append([])

        for i in range(len(item_gt_labels)):
            label = int(item_gt_labels[i] + 0.5)
            mask = item_gt_masks[i]
            gt[label].append(mask)

        zeroShape = [0] + fas.wrapped.shape
        for i in range(numClasses):
            if len(gt[i]) == 0:
                gt[i] = np.zeros([0]+zeroShape, dtype=np.float32)
            else:
                gt[i] = np.concatenate([np.expand_dims(x,axis=0) for x in gt[i]],axis=0)

        pred = pi.prediction
        # withMasks = fas.wrapped.withMask()
        # threshold = fas.wrapped.threshold
        # pred = convertMMDETModelOutput(pred, withMasks)
        # pred = applyTresholdToPrediction(pred, withMasks, threshold)

        item_pred_labels = pred[0]
        item_pred_masks = pred[3]

        res_masks = item_pred_masks

        result = []
        for i in range(numClasses):
            result.append([])

        for i in range(len(item_pred_labels)):
            label = int(item_pred_labels[i] + 0.5)
            mask = res_masks[i]
            result[label].append(mask)

        for i in range(numClasses):
            if len(result[i]) == 0:
                result[i] = np.zeros(zeroShape, dtype=np.float32)
            else:
                result[i] = np.concatenate([np.expand_dims(x, axis=0) for x in result[i]], axis=0)

        for i in range(numClasses):

            pred_objects = result[i]
            true_objects = gt[i]
                # print(len(pred_objects),len(true_objects),i)
            if len(true_objects) == 0 and len(pred_objects) == 0:
                continue
            else:
                map_val = map(true_objects, pred_objects)
                print(map_val)
                if i in clazz:
                    clazz[i] = clazz[i] + map_val;
                    clazz_counts[i] = clazz_counts[i] + 1
                else:
                    clazz[i] = map_val;
                    clazz_counts[i] = 1

            cs = []
            for j in sorted(clazz.keys()):
                clazz_score = clazz[j] / clazz_counts[j]
                print(j, clazz_score)
                cs.append(clazz_score)
            print("Mean", np.mean(cs))

    score = 0
    for pi in clazz:
        score += clazz[pi]/clazz_counts[pi]

    return score

def iou(img_true, img_pred):
    i = np.sum((img_true*img_pred) >0)
    u = np.sum((img_true + img_pred) >0) + 0.0000000000000000001  # avoid division by zero
    return i/u

thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]



def map(masks_true, masks_pred):
    if np.sum(masks_true) == 0:
        return float(np.sum(masks_pred) == 0)
    ious = []
    mp_idx_found = []
    for mt in masks_true:
        for mp_idx, mp in enumerate(masks_pred):
            if mp_idx not in mp_idx_found:
                cur_iou = iou(mt,mp)
                if cur_iou > 0.5:
                    ious.append(cur_iou)
                    mp_idx_found.append(mp_idx)
                    break
    f2_total = 0
    for th in thresholds:
        tp = sum([iou > th for iou in ious])
        fn = len(masks_true) - tp
        fp = len(masks_pred) - tp
        f2_total += tp/(tp + fn + fp)

    return f2_total/len(thresholds)