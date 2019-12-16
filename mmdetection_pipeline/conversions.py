import numpy as np
from pycocotools import mask as mask_util
# import time

def convertMMDETModelOutput(pred, withMasks, threshold):
    labels = []
    probabilities = []
    resultBBoxes = []
    resultMasks = []
    picBBboxes = pred[0] if withMasks else pred
    if withMasks:
        picMasks = pred[1]
    
    # t_decode = 0
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
            prob = predBB[4]
            if prob < threshold:
                continue
            bb = predBB[:4]
            labels.append(label)
            probabilities.append(prob)
            resultBBoxes.append(bb)
            if withMasks:
                # t0 = time.time()
                decodedMask = mask_util.decode(masks[i])
                # t1 = time.time()
                # t_decode += (t1-t0)
                resultMasks.append(decodedMask)
#    print(f"Masks decode: {t_decode}")
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