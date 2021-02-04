import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

def bin2pointcloud(filename):
    """
    bin->point cloud
    return: point cloud, size(n, 4)
    """
    pointcloud = np.fromfile(filename,dtype=np.float64).reshape(-1,4)
    return pointcloud


        


if __name__=="__main__":
    filename = '000000.bin'
    points = bin2pointcloud(filename)
    np.savetxt("pointcloud.txt", points)
    print(points[1111])