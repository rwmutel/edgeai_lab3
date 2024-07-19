from rknnlite.api import RKNNLite

rknn = RKNNLite(verbose=True)
rknn.load_rknn('/home/orangepi/yolo/models/yolov5s_i8.rknn')
rknn.init_runtime()