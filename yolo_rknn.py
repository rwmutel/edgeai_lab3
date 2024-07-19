from rknnlite.api import RKNNLite
from yolo_base import YOLOBase

class YOLORKNN(YOLOBase):
    def __init__(self, path):
        self.rknn = RKNNLite()
        self.rknn.load_rknn('/home/orangepi/yolo/models/yolov5s_i8.rknn')
        ret = self.rknn.init_runtime()
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print(f'Loaded RKNN YOLO from {path}')
    
    def infer(self, img):
        if not (isinstance(img, list) or isinstance(img, tuple)):
            if len(img.shape) == 3:
                img = img.reshape(1, *img.shape)
            img = [img]
        return self.rknn.inference(img)
    