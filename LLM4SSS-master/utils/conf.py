import json

class Configuration:
    def __init__(self, confPath: str = ""):
        self.confPath = confPath
        self.confPath = confPath.strip().replace('\r', '')
        self.defaultConf = {}
        self.loadConf()
    
    
    def loadConf(self):
        with open(self.confPath, 'r') as fin:
            self.confLoaded = json.load(fin)
            
            
    def getEntry(self, key: str):
        if key in self.confLoaded:
            return self.confLoaded[key]
        elif key in self.defaultConf:
            return self.defaultConf[key]
        else:
            raise Exception(f"Key {key} not found in configuration.")