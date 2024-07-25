import tflite_runtime.interpreter as tflite
from yolo_base import YOLOBase
import numpy as np

def convert_tflite_to_onnx(tflite_output):
    # Reshape TFLite output from (25200, 85) to (1, 3, 80, 80, 85),
    # ingoring the 40x40 and 20x20 detections
    tflite_output = (tflite_output[:, :19200, :]).reshape(1, 3, 80, 80, 85)
    # Transpose to (1, 3, 85, 80, 80)
    tflite_output = tflite_output.transpose(0, 1, 4, 2, 3)
    # Reshape to (1, 255, 80, 80)
    onnx_output = tflite_output.reshape(1, 255, 80, 80)
    return onnx_output


class YOLOTFLite(YOLOBase):
    def __init__(self, path):
        self.in_type = np.uint8 if 'int8' in path else np.float32
        self.interpreter = tflite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f'Loaded TFLite YOLO from {path}')
    
    def infer(self, img):
        if len(img.shape) == 3:
            img = img.reshape(1, *img.shape).astype(self.in_type)
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        return convert_tflite_to_onnx(
            [self.interpreter.get_tensor(out['index']) for out in self.output_details][0])
    
