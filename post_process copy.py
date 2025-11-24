import re
from itertools import product
from typing import List, Tuple
import onnxruntime as ort
import cv2

import numpy as np


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
        self._input_image_size = input_image_size
        self.nms_threshold = 0.3
        self.face_prob_threshold = 0.7
        self.variance = [0.1, 0.2]
        self.prior_data = self.generate_prior_data()
        self.model = ort.InferenceSession('RetinaFace_int.onnx', providers=['DmlExecutionProvider'])


    def process_output(self, raw_input: np.ndarray):
        img = np.expand_dims(cv2.resize(raw_input, self._input_image_size), axis=0).astype(np.float32)#.transpose(0, 3, 1, 2)
        print(img.shape)
        raw_output = self.model.run(None, {self.model.get_inputs()[0].name: img})

        #bboxes_output = [raw_output[name][0] for name in raw_output if re.search('.bbox.', name)][0]
        bboxes_output = raw_output[0][0]
        scores_output = raw_output[1][0]
        #scores_output = [raw_output[name][0] for name in raw_output if re.search('.cls.', name)][0]

        proposals = self._get_proposals(bboxes_output, self.prior_data)
        scores = scores_output[:, 1]
        filter_idx = np.where(scores > self.face_prob_threshold)[0]
        proposals = proposals[filter_idx]
        scores = scores[filter_idx]

        if np.size(scores) > 0:
            x_mins, y_mins, x_maxs, y_maxs = proposals.T
            keep = nms(x_mins, y_mins, x_maxs, y_maxs, scores, self.nms_threshold)

            proposals = proposals[keep]
            scores = scores[keep]

        box_arr, score_arr = [], []
        if np.size(scores) != 0:
            scores = np.reshape(scores, -1)
            x_mins, y_mins, x_maxs, y_maxs = np.array(proposals).T

            for x_min, y_min, x_max, y_max, score in zip(x_mins, y_mins, x_maxs, y_maxs, scores):
                x_min *= self.scale_x
                y_min *= self.scale_y
                x_max *= self.scale_x
                y_max *= self.scale_y
                box_arr.append((x_min, y_min, x_max, y_max))
                score_arr.append(score)

        return box_arr, score_arr

    def generate_prior_data(self):
        global_min_sizes = [[16, 32], [64, 128], [256, 512]]
        steps = [8, 16, 32]
        anchors = []
        feature_maps = [[int(np.rint(self._input_image_size[1] / step)), int(np.rint(self._input_image_size[0] / step))]
                        for step in steps]
        for idx, feature_map in enumerate(feature_maps):
            min_sizes = global_min_sizes[idx]
            for i, j in product(range(feature_map[0]), range(feature_map[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self._input_image_size[0]
                    s_ky = min_size / self._input_image_size[1]
                    dense_cx = [x * steps[idx] / self._input_image_size[0] for x in [j + 0.5]]
                    dense_cy = [y * steps[idx] / self._input_image_size[1] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        priors = np.array(anchors).reshape((-1, 4))
        return priors

    def _get_proposals(self, raw_boxes, priors):
        proposals = self.decode_boxes(raw_boxes, priors, self.variance)
        proposals[:, ::2] = proposals[:, ::2] * self._input_image_size[0]
        proposals[:, 1::2] = proposals[:, 1::2] * self._input_image_size[1]
        return proposals

    @staticmethod
    def decode_boxes(raw_boxes, priors, variance):
        boxes = np.concatenate((
            priors[:, :2] + raw_boxes[:, :2] * variance[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(raw_boxes[:, 2:] * variance[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    @property
    def scale_x(self) -> float:
        return self._origin_image_size[0] / self._input_image_size[0]

    @property
    def scale_y(self) -> float:
        return self._origin_image_size[1] / self._input_image_size[1]