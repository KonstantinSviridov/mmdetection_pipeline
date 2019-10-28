import os
import xml.etree.ElementTree as ET
import requests
import mmdet
import threading
from mmcv import Config
import torch
__lock__ = threading.Lock()

mmdetDir = os.path.dirname(mmdet.__file__)
mmdetConfigsDir = os.path.dirname(mmdetDir) + "/configs"

amazonBucketPath = "https://s3.ap-northeast-2.amazonaws.com/open-mmlab"

def listModelWeightPaths(prefix:str = "mmdetection/models/", postfix:str = ".pth", bucketURL=amazonBucketPath)->[str]:

    allPaths = listBucket(bucketURL)
    filtered = [x for x in allPaths if (x.startswith(prefix) and x.endswith(postfix))]

    result = [{"key": x, "filename": os.path.basename(x), "path": f"{bucketURL}/{x}"} for x in filtered]
    return result

def listBucket(url:str)->[str]:

    tree = getXML(url)
    keys = listKeys(tree)
    return keys


def listKeys(tree):
    rootTag = tree.tag
    nsString = ""
    ind = rootTag.rindex("}")
    if rootTag[0] == "{":
        nsString = rootTag[0:ind + 1]
    contentsTag = nsString + "Contents"
    keyTag = nsString + "Key"
    contents = [x for x in tree if x.tag == contentsTag]
    keys = [[x for x in contentsElement if x.tag == keyTag][0].text for contentsElement in contents]
    return keys


def getXML(url):
    response = requests.get(url)
    text = response.text
    tree = ET.fromstring(text)
    return tree


def listConfigNames(p, result = []):
    if os.path.isdir(p):
        for x in os.listdir(p):
            xp = os.path.join(p,x)
            listConfigNames(xp, result)
    elif p.endswith(".py"):
        try:
            cfg = Config.fromfile(p)
            result.append(os.path.basename(p))
        except:
            pass
    return result

def buildCorrespondence(configsDir=mmdetConfigsDir, bucketPath=amazonBucketPath, prefix:str = "mmdetection/models/", postfix:str = ".pth"):
    configNames = set([x[0:x.rindex(".py")] for x in listConfigNames(configsDir)])
    weightPaths = listModelWeightPaths(prefix,postfix,bucketPath)
    result = {}
    for pth in weightPaths:
        name = pth["filename"]
        ind = name.rfind("_")
        if ind >= 0:
            n = name[0:ind]
            if n in configNames:
                result[n] = pth["path"]
    return result

storage = {}
alreadyTriedToContribute = False

def getPath(name:str):
    global storage
    contributeCheckpoins()
    if name is None or name not in storage:
        return None

    return storage[name]

def getCheckpoint(name:str, mapLocation=None):
    global storage
    contributeCheckpoins()
    if name is None or name not in storage:
        return None

    p = storage[name]
    result = torch.load(p,mapLocation)
    return result


def contributeCheckpoins(configsDir=mmdetConfigsDir, bucketPath=amazonBucketPath, prefix:str = "mmdetection/models/", postfix:str = ".pth"):

    global alreadyTriedToContribute, storage

    if alreadyTriedToContribute:
        return

    __lock__.acquire()
    try:
        correspondence = buildCorrespondence(configsDir,bucketPath,prefix,postfix)
        target = storage
        for x in correspondence:
            target[x] = correspondence[x]

        alreadyTriedToContribute = True
    except Exception as e:
        print("Unable to contribute mmdet-labs checkpoints")
        print(e)
    finally:
        __lock__.release()
