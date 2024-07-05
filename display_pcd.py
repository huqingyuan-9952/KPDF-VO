# import pcl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def display_pcd(filename):
    # 从文件加载点云数据
    # cloud = pcl.load(filename)
    points = np.array(cloud)

    # 提取坐标
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=z, cmap='viridis', marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud')
    plt.show()

# 调用函数来显示点云数据
display_pcd('point_cloud.pcd')

