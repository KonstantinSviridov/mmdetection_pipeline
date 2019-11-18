import numpy as np
from pycocotools import mask as mask_util

def convertMMDETModelOutput(pred, withMasks):
    labels = []
    probabilities = []
    resultBBoxes = []
    resultMasks = []
    picBBboxes = pred[0] if withMasks else pred
    if withMasks:
        picMasks = pred[1]
    for label in range(len(picBBboxes)):
        bboxes = picBBboxes[label]
        if len(bboxes) == 0:
            continue

        l = len(bboxes)
        if withMasks:
            masks = picMasks[label]
            if len(masks) != l:
                print("Mask and bounding boxes arrays have different lengths")
                l = min(l, len(masks))

        for i in range(l):
            predBB = bboxes[i]
            bb = predBB[:4]
            prob = predBB[4]
            labels.append(label)
            probabilities.append(prob)
            resultBBoxes.append(bb)
            if withMasks:
                decodedMask = mask_util.decode(masks[i])
                resultMasks.append(decodedMask)
    labels = np.array(labels, dtype=np.int16)
    probabilities = np.array(probabilities)
    resultBBoxes = np.array(resultBBoxes).reshape((-1,4))
    if withMasks:
        resultMasks = np.array(resultMasks)
        converted = (labels, probabilities, resultBBoxes, resultMasks)
    else:
        converted = (labels, probabilities, resultBBoxes)
    return converted

def applyTresholdToPrediction(pred, withMasks, threshold):
    probabilites = pred[1]
    inds = np.nonzero(probabilites >= threshold)
    labels = pred[0][inds]
    bboxes = pred[2][inds]
    if withMasks:
        masks = pred[3][inds]
        tresholdedPrediction = (labels, bboxes, masks)
    else:
        tresholdedPrediction = (labels, bboxes)
    return tresholdedPrediction