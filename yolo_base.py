class YOLOBase:
    def __init__(self, path):
        raise NotImplementedError
    
    def infer(self, img):
        raise NotImplementedError
