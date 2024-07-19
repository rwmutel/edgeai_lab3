import onnxruntime as ort
from yolo_base import YOLOBase
import numpy as np

class YOLOONNX(YOLOBase):
    def __init__(self, path):
        self.ort_session = ort.InferenceSession(path,
                                                providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
        if not self.ort_session:
            print(f'Failed to load ONNX YOLO from {path}')
            exit(1)
        print(f'Loaded ONNX YOLO from {path}')

    def infer(self, img):
        input_data = img.transpose((2,0,1))
        input_data = input_data.reshape(1,*input_data.shape).astype(np.float32)
        input_data = input_data/255.0
        return self.ort_session.run(None, {'images': input_data})
    