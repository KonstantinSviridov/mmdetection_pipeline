import numpy as np

def getBB(mask):
    a = np.where(mask != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    minX = max(0, bbox[0] - 10)
    maxX = min(mask.shape[0], bbox[1] + 1 + 10)
    minY = max(0, bbox[2] - 10)
    maxY = min(mask.shape[1], bbox[3] + 1 + 10)
    return np.array([maxX, maxY, minX, minY])



def isURL(s:str)->bool:
    if s is None:
        return False
    l = s.lower()
    return l.startswith("http://") or l.startswith("https://")