import onnxruntime as ort
import numpy as np
from post_process import RetinaFacePostPostprocessor
import cv2




test_processor = RetinaFacePostPostprocessor((1920, 1080), (640, 640))

model = ort.InferenceSession('unquantized_retinaface.onnx', providers=['DmlExecutionProvider'])
#img = np.random.rand(1, 3, 640, 640).astype(np.float32)
img = cv2.imread("C:/Users/545391/Downloads/download.jpg")
img = np.expand_dims(cv2.resize(img, (640, 640)), axis=0).transpose(0, 3, 1, 2).astype(np.float32)

print(img.shape)

outputs = model.run(None, {model.get_inputs()[0].name: img})



converted = test_processor.process_output(outputs)

print(converted)