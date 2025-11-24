import re
from itertools import product
from typing import List, Tuple
import onnxruntime as ort
import cv2

import numpy as np
import torch
from torch.nn.functional import softmax as torch_softmax


def nms(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray,
        scores: np.ndarray, thresh: float) -> List[int]:
    b = 1
    areas = (x2 - x1 + b) * (y2 - y1 + b)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + b)
        h = np.maximum(0.0, yy2 - yy1 + b)
        intersection = w * h

        union = (areas[i] + areas[order[1:]] - intersection)
        overlap = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)

        order = order[np.where(overlap <= thresh)[0] + 1]

    return keep


class RetinaFacePostPostprocessor:
    def __init__(self, origin_image_size: Tuple[int, int], input_image_size: Tuple[int, int]):
        self._origin_image_size = origin_image_size
        self._input_image_size = input_image_size[::-1] # need to reverse from WH to HW
        self.nms_threshold = 0.5
        self.face_prob_threshold = 0.8
        self.variance = [0.1, 0.2]
        self.prior_data = self.generate_prior_data()
        self.model = ort.InferenceSession('RetinaFace_int.onnx', providers=['DmlExecutionProvider'])

    def preprocess(self, image):
        resized_image, res_ratio = self.resize_image(image, self._input_image_size, keep_ratio=True) # Height, Width, Channel
        resized_image -= (104, 117, 123)
        #resized_image = resized_image.transpose(1, 0, 2)  # Width, Height, Channel
        #resized_image = np.transpose(resized_image, (0, 2, 1, 3))
        return np.expand_dims(resized_image, axis=0), res_ratio

    def process_output(self, raw_input: np.ndarray):
        img, res_ratio = self.preprocess(raw_input)
        #cv2.imwrite('test.jpg', img[0])
        raw_output = self.model.run(None, {self.model.get_inputs()[0].name: img})

        #bboxes_output = [raw_output[name][0] for name in raw_output if re.search('.bbox.', name)][0]
        bboxes_output = raw_output[0][0]
        scores_output = raw_output[1][0]
        #scores_output = [raw_output[name][0] for name in raw_output if re.search('.cls.', name)][0]
        landm_output = raw_output[2][0]


        proposals = self._get_proposals(bboxes_output, self.prior_data, res_ratio)
        #scores_output = self._softmax(scores_output)
        scores_output = torch_softmax(torch.from_numpy(raw_output[1][0]), dim=-1).numpy()
        scores = scores_output[:, 1]

        landms = self.decode_landm(landm_output, self.prior_data, self.variance, res_ratio)

        filter_idx = np.where(scores > self.face_prob_threshold)[0]
        proposals = proposals[filter_idx]
        scores = scores[filter_idx]
        landms = landms[filter_idx]

        if np.size(scores) > 0:
            x_mins, y_mins, x_maxs, y_maxs = proposals.T
            keep = nms(x_mins, y_mins, x_maxs, y_maxs, scores, self.nms_threshold)

            proposals = proposals[keep]
            scores = scores[keep]
            landms = landms[keep]

        box_arr, score_arr, land_arr = [], [], []
        if np.size(scores) != 0:
            scores = np.reshape(scores, -1)
            x_mins, y_mins, x_maxs, y_maxs = np.array(proposals).T

            for x_min, y_min, x_max, y_max, score, landmark in zip(x_mins, y_mins, x_maxs, y_maxs, scores, landms):
                #x_min *= self.scale_x
                #y_min *= self.scale_y
                #x_max *= self.scale_x
                #y_max *= self.scale_y
                box_arr.append((x_min, y_min, x_max, y_max))
                score_arr.append(score)
                land_arr.append(landmark)

        return box_arr, score_arr, land_arr

    def generate_prior_data(self):
        global_min_sizes = [[16, 32], [64, 128], [256, 512]]
        steps = [8, 16, 32]
        anchors = []
        feature_maps = [[int(np.rint(self._input_image_size[0] / step)), int(np.rint(self._input_image_size[1] / step))]
                        for step in steps]
        for idx, feature_map in enumerate(feature_maps):
            min_sizes = global_min_sizes[idx]
            for i, j in product(range(feature_map[0]), range(feature_map[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self._input_image_size[1]
                    s_ky = min_size / self._input_image_size[0]
                    dense_cx = [x * steps[idx] / self._input_image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * steps[idx] / self._input_image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        priors = np.array(anchors).reshape((-1, 4))
        return priors

    def _get_proposals(self, raw_boxes, priors, res_ratio):
        h, w = self._input_image_size
        scale = np.array([w, h, w, h]) #0 = H, 1=W
        #print(res_ratio)
        #quit()
        proposals = self.decode_boxes(raw_boxes, priors, self.variance)
        #proposals[:, ::2] = proposals[:, ::2] * self._input_image_size[1]
        #proposals[:, 1::2] = proposals[:, 1::2] * self._input_image_size[0]
        return proposals * scale / res_ratio

    @staticmethod
    def decode_boxes(raw_boxes, priors, variance):
        boxes = np.concatenate((
            priors[:, :2] + raw_boxes[:, :2] * variance[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(raw_boxes[:, 2:] * variance[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def decode_landm(self, pre, priors, variances, res_ratio):
        """Decode landm from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            pre (tensor): landm predictions for loc layers,
                Shape: [num_priors,10]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded landm predictions
        """
        landms = np.concatenate(
            (
                priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
            ),
            axis=1,
        )
        h, w = self._input_image_size
        scale = np.array([w, h, w, h, w, h, w, h, w, h])
        return landms * scale / res_ratio

    @property
    def scale_x(self) -> float:
        return self._origin_image_size[0] / self._input_image_size[1]

    @property
    def scale_y(self) -> float:
        return self._origin_image_size[1] / self._input_image_size[0]


    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                        pad_w, cv2.BORDER_CONSTANT,
                                        value=padvalue)
        return pad_image


    def resize_image(self, image, re_size, keep_ratio=True):
        """Resize image
        Args: 
            image: origin image
            re_size: resize scale
            keep_ratio: keep aspect ratio. Default is set to true.
        Returns:
            re_image: resized image
            resize_ratio: resize ratio
        """
        if not keep_ratio:
            re_image = cv2.resize(image, (re_size[1], re_size[0])).astype('float32')                    
            return re_image, 0
        ratio = re_size[0] * 1.0 / re_size[1] 
        h, w = image.shape[0:2]
        if h * 1.0 / w <= ratio:
            resize_ratio = re_size[1] * 1.0 / w
            re_h, re_w = int(h * resize_ratio), re_size[1] 
        else:
            resize_ratio = re_size[0] * 1.0 / h
            re_h, re_w = re_size[0], int(w * resize_ratio)

        re_image = cv2.resize(image, (re_w, re_h)).astype('float32')              
        re_image = self.pad_image(re_image, re_h, re_w, re_size, (0.0, 0.0, 0.0))
        return re_image.astype(np.float32), resize_ratio

    def _softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        #e_x = np.exp(x - np.max(x))
        #return e_x / e_x.sum()
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)