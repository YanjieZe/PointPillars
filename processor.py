from typing import List
import numpy as np
import torch

from config import Parameters
from point_pillars import createPillars, createPillarsTarget
from reader import DataReader, Label3D, KittiDataReader
from sklearn.utils import shuffle
import sys
import tensorflow as tf
from torch.utils.data import Dataset


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
        return transformed

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
        """
        return shape: 252*252*4, 252*252*4*3, 252*252*4*3, 252*252*4, 252*252*4, 252*252*4*4
        """

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
        one_hot_encode = tf.keras.utils.to_categorical(sel[..., 9], num_classes=self.nb_classes, dtype='float64')
        #      252*252*4,  252*252*4*3,   252*252*4*3       252*252*4   252*252*4,  252*252*4*4
        return sel[..., 0], sel[..., 1:4], sel[..., 4:7], sel[..., 7], sel[..., 8], one_hot_encode


class kitti_dataset(Dataset, DataProcessor):
    
    def __init__(self, root_path=Parameters.kitti_path):
        super(kitti_dataset, self).__init__()
        self.training_pc_root = root_path+'/training/velodyne/'
        self.traning_label_root = root_path+'/training/label_2/'
        # print(self.training_pc_root)
        self.file_num = 7481


    def get_filename(self, idx:int, suffix='.bin'):
        digit = 0
        tmp = int(idx)
        while tmp>0:
            tmp = int(tmp/10)
            digit += 1
        zeros = 6-digit
        if idx ==0:
            zeros = 5
        filename = '0'*zeros + str(idx)+suffix
        return filename

        
    
    def __getitem__(self, idx):
        pc_file = self.training_pc_root + self.get_filename(idx, suffix='.bin')
        label_file = self.traning_label_root + self.get_filename(idx, suffix='.txt')

        # get input
        pointcloud = KittiDataReader.read_lidar(pc_file)
        pillar_points, indices = self.make_point_pillars(pointcloud)
        pillar_points = torch.from_numpy(pillar_points).squeeze().float()
        indices = torch.from_numpy(indices).squeeze().float()


        # get label
        label = KittiDataReader.read_label(label_file)
        R, t = KittiDataReader.read_calibration()
        label_transformed = self.transform_labels_into_lidar_coordinates(label, R, t) # 这里为什么要做一个变换，还是没怎么懂
        occ0, loc0, size0, angle0, heading0, clf0 = self.make_ground_truth(label_transformed)

        occ0 = torch.from_numpy(occ0).float()
        loc0 = torch.from_numpy(loc0).float()
        size0 = torch.from_numpy(size0).float()
        angle0 = torch.from_numpy(angle0).float()
        heading0 = torch.from_numpy(heading0).float()
        clf0 = torch.from_numpy(clf0).float()

        return [pillar_points, indices], [occ0, loc0, size0, angle0, heading0, clf0]


    def __len__(self):
        return self.file_num
        


if __name__=='__main__':
    # filename = '000000.txt'
    # list_label = KittiDataReader.read_label(filename)
        
    # processor = DataProcessor()
    # a,b,c,d,e,f = processor.make_ground_truth(list_label)
    # print(f.shape)
    # print(to_categorical([1,2,3,4],num_classes=5))
    ds = kitti_dataset()
    inp, out = ds[1]
    print(out[1])
    # print(ds.get_filename(788,suffix='.txt'))


