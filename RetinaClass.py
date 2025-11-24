import os
import argparse
import onnxruntime as ort
from utils import *

CFG = {
    "name": "mobilenet0.25",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
}
INPUT_SIZE = [608, 640]   #resize scale
DEVICE = torch.device("cpu")

class RetinaFace():
    def __init__(self):
        self.run_ort = ort.InferenceSession('RetinaFace_int.onnx', providers=['DmlExecutionProvider'])
        self.conf_thresh = 0.4
        self.nms_thresh = 0.5

    def run_inference(self, img_raw):
        """Infer an image with onnx seession
        Args:
            run_ort: Onnx session
            args: including image path and hyperparameters
        Returns: boxes_list, confidence_list, landm_list
            boxes_list = [[left, top, right, bottom]...]
            confidence_list = [[confidence]...]
            landm_list = [[landms(dim=10)]...]
        """
        # preprocess
        img, scale, resize = preprocess(img_raw, INPUT_SIZE, DEVICE)

        #print(img.shape)
        # to NHWC
        img = np.transpose(img, (0, 2, 3, 1))

        # forward 
        outputs = self.run_ort.run(None, {self.run_ort.get_inputs()[0].name: img})

        # postprocess
        dets = postprocess(CFG, img, outputs, scale, resize, self.conf_thresh, self.nms_thresh, DEVICE)

        # result list
        boxes = dets[:, :4]
        confidences = dets[:, 4:5]
        landms = dets[:, 5:]
        boxes_list = [box.tolist() for box in boxes]
        confidence_list = [confidence.tolist() for confidence in confidences]
        landm_list = [landm.tolist() for landm in landms]

        return boxes_list, confidence_list, landm_list

