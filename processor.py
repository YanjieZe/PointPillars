from typing import List
import numpy as np
import torch

from config import Parameters
from point_pillars import createPillars, createPillarsTarget
from reader import DataReader, Label3D, KittiDataReader
from sklearn.utils import shuffle
import sys


def select_best_anchors(arr):
    dims = np.indices(arr.shape[1:])
    # arr[..., 0:1] gets the occupancy value from occ in {-1, 0, 1}, i.e. {bad match, neg box, pos box}
    ind = (np.argmax(arr[..., 0:1], axis=0),) + tuple(dims)
    return arr[ind]


def to_categorical(y, num_classes,dtype='float64'):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype=dtype)[y]


class DataProcessor(Parameters):

    def __init__(self):
        super(DataProcessor, self).__init__()
        anchor_dims = np.array(self.anchor_dims, dtype=np.float32)
        self.anchor_dims = anchor_dims[:, 0:3]
        self.anchor_z = anchor_dims[:, 3]
        self.anchor_yaw = anchor_dims[:, 4]
        # Counts may be used to make statistic about how well the anchor boxes fit the objects
        self.pos_cnt, self.neg_cnt = 0, 0

    @staticmethod
    def transform_labels_into_lidar_coordinates(labels: List[Label3D], R: np.ndarray, t: np.ndarray):
        """
        Input: label3D, Rotation Matrix, Translation Vector

        Output: lidar coordinates
        """
        transformed = []
        for label in labels:
            label.centroid = label.centroid @ np.linalg.inv(R).T - t
            label.dimension = label.dimension[[2, 1, 0]]
            label.yaw -= np.pi / 2
            while label.yaw < -np.pi:
                label.yaw += (np.pi * 2)
            while label.yaw > np.pi:
                label.yaw -= (np.pi * 2)
            transformed.append(label)
        return labels

    def make_point_pillars(self, points: np.ndarray):

        assert points.ndim == 2
        assert points.shape[1] == 4
        assert points.dtype == np.float32

        pillars, indices = createPillars(points,
                                         self.max_points_per_pillar,
                                         self.max_pillars,
                                         self.x_step,
                                         self.y_step,
                                         self.x_min,
                                         self.x_max,
                                         self.y_min,
                                         self.y_max,
                                         self.z_min,
                                         self.z_max,
                                         False)

        return pillars, indices

    def make_ground_truth(self, labels: List[Label3D]):

        # filter labels by classes (cars, pedestrians and Trams)
        # Label has 4 properties (Classification (0th index of labels file),
        # centroid coordinates, dimensions, yaw)
        labels = list(filter(lambda x: x.classification in self.classes, labels))

        if len(labels) == 0:
            pX, pY = int(self.Xn / self.downscaling_factor), int(self.Yn / self.downscaling_factor)
            a = int(self.anchor_dims.shape[0])
            return np.zeros((pX, pY, a), dtype='float32'), np.zeros((pX, pY, a, self.nb_dims), dtype='float32'), \
                   np.zeros((pX, pY, a, self.nb_dims), dtype='float32'), np.zeros((pX, pY, a), dtype='float32'), \
                   np.zeros((pX, pY, a, self.nb_classes), dtype='float64')

        # For each label file, generate these properties except for the Don't care class
        target_positions = np.array([label.centroid for label in labels], dtype=np.float32)
        target_dimension = np.array([label.dimension for label in labels], dtype=np.float32)
        target_yaw = np.array([label.yaw for label in labels], dtype=np.float32)
        target_class = np.array([self.classes[label.classification] for label in labels], dtype=np.int32)

        assert np.all(target_yaw >= -np.pi) & np.all(target_yaw <= np.pi)
        assert len(target_positions) == len(target_dimension) == len(target_yaw) == len(target_class)

        target, pos, neg = createPillarsTarget(target_positions,
                                               target_dimension,
                                               target_yaw,
                                               target_class,
                                               self.anchor_dims,
                                               self.anchor_z,
                                               self.anchor_yaw,
                                               self.positive_iou_threshold,
                                               self.negative_iou_threshold,
                                               self.nb_classes,
                                               self.downscaling_factor,
                                               self.x_step,
                                               self.y_step,
                                               self.x_min,
                                               self.x_max,
                                               self.y_min,
                                               self.y_max,
                                               self.z_min,
                                               self.z_max,
                                               False)
        self.pos_cnt += pos
        self.neg_cnt += neg

        # return a merged target view for all objects in the ground truth and get categorical labels
        sel = select_best_anchors(target)
        
        
        # class_label = np.array(sel[...,9],dtype='uint8').tolist()
        # print(class_label)
        # one_hot_encode = to_categorical(sel[..., 9], num_classes=self.nb_classes)

        return sel[..., 0], sel[..., 1:4], sel[..., 4:7], sel[..., 7], sel[..., 8], sel[...,9]


if __name__=='__main__':
    filename = '000000.txt'
    list_label = KittiDataReader.read_label(filename)

    processor = DataProcessor()
    gt = processor.make_ground_truth(list_label)
    print(gt)
    # print(to_categorical([1,2,3,4],num_classes=5))

