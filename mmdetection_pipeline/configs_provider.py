from mmcv import Config
import os

def getConfigs(p, result = []):
    if os.path.isdir(p):
        for x in os.listdir(p):
            xp = os.path.join(p,x)
            getConfigs(xp, result)
    elif p.endswith(".py"):
        try:
            cfg = Config.fromfile(p)
            result.append(cfg)
        except:
            pass
    return result