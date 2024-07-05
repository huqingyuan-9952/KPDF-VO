''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2019-01-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-07-09
@LastEditors: Huangying Zhan
@Description: DF-VO core program
'''

import cv2
import copy
from glob import glob
import math
from matplotlib import pyplot as plt
import numpy as np
import os
from time import time
from tqdm import tqdm

#kp3d
import sys
import torch
# import pcl
# import open3d as o3d
import argparse
import copy
import random
import subprocess
from sklearn.cluster import DBSCAN
from scipy.optimize import minimize
from PIL import Image
from termcolor import colored

from libs.geometry.camera_modules import SE3
import libs.datasets as Dataset
from libs.deep_models.deep_models import DeepModel
from libs.general.frame_drawer import FrameDrawer
from libs.general.timer import Timer
from libs.matching.keypoint_sampler import KeypointSampler
from libs.matching.depth_consistency import DepthConsistency
from libs.tracker import EssTracker, PnpTracker
from libs.general.utils import *

#kp3d
from libs.geometry.camera import Camera
from libs.geometry.pose import Pose
from libs.deep_models.disp_resnet import DispResnet
from libs.deep_models.keypoint_resnet import KeypointResnet
from libs.deep_models.superglue import SuperGlue
from libs.utils.image import to_color_normalized
from termcolor import colored

#add keyframe
from collections import deque
from libs.helpers import poseRt

#add backend
from libs.backend.display import Display2D, Display3D
from libs.backend.frame import Frame, match_frames
import numpy as np
import g2o
from libs.backend.pointmap import Map, Point
from libs.backend.helpers import triangulate, add_ones

class KeyFrame:
    def __init__(self, frame_number, data):
        self.frame_number = frame_number
        self.data = data


def plot_reproject(point2_2D, p_2D_array, source_Image, avg_e,window_name):
    """
    在图像上绘制两组坐标点并连线

    Parameters:
    point2_2D: numpy.ndarray
        第一组坐标点的数组，形状为 (N, 2)
    p_2D_array: numpy.ndarray
        第二组坐标点的数组，形状与 point2_2D 相同
    source_Image: numpy.ndarray
        原始图像
    avg_e: float
        平均误差值

    Returns:
    None
    """
    combined_image = source_Image.copy()
    
    # 绘制第一组坐标点和第二组坐标点
    for i in range(len(point2_2D)):
        cv2.circle(combined_image, tuple(point2_2D[i].astype(int)), 5, (0, 0, 255), -1)
        # cv2.circle(combined_image, tuple(p_2D_array[i].astype(int)), 5, (255, 0, 0), -1)
        
        # 绘制连线
        cv2.line(combined_image, tuple(point2_2D[i].astype(int)), tuple(p_2D_array[i].astype(int)), (0, 255, 0), 1)
        
        # 计算两点之间的距离
        e = point2_2D[i] - p_2D_array[i]
        distance = np.linalg.norm(e)
        
        # 在图像上显示距离信息
        text_position = ((point2_2D[i, 0] + p_2D_array[i, 0]) // 2, (point2_2D[i, 1] + p_2D_array[i, 1]) // 2)
        cv2.putText(combined_image, f'{distance:.2f}', (int(text_position[0]), int(text_position[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    
    # 在图像中添加 avg_e 文本
    cv2.putText(combined_image, f'Avg_e = {avg_e:.2f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 显示图像
    filename = f'{window_name.replace(" ", "_")}.png'
    cv2.imshow(filename, combined_image)
    cv2.waitKey(10)



def plot_reproject_con(p_2D1_unrob_array,p_2D2_unrob_array,p_2D1_array,p_2D2_array,source_Image,avg_e):
    """
    在两个源图像上绘制两组坐标点并连线

    Parameters:
    source_Image1: numpy.ndarray
        第一个原始图像
    source_Image2: numpy.ndarray
        第二个原始图像
    p_2D1_array: numpy.ndarray
        第一组第一个点的像素坐标数组，形状为 (N, 2)
    p_2D2_array: numpy.ndarray
        第二组第一个点的像素坐标数组，形状为 (N, 2)
    p_2D1_unrob_array: numpy.ndarray
        第一组第二个点的像素坐标数组，形状与 p_2D1_array 相同
    p_2D2_unrob_array: numpy.ndarray
        第二组第二个点的像素坐标数组，形状与 p_2D2_array 相同
    avg_e: float
        两组点的平均误差

    Returns:
    None
    """
    # 将两个源图像拼接在一起
    combined_image = cv2.hconcat((source_Image, source_Image))

    for i in range(len(p_2D1_array)):
        cv2.circle(combined_image, tuple(p_2D1_unrob_array[i].astype(int)), 5, (0, 0, 255), -1)
        cv2.circle(combined_image, tuple((p_2D2_unrob_array[i] ).astype(int)), 5, (255, 0, 0), -1)
        cv2.line(combined_image, tuple(p_2D1_unrob_array[i].astype(int)), tuple((p_2D2_unrob_array[i] ).astype(int)), (0, 255, 0), 1)

    # 在拼接后的图像上绘制第二组坐标点和连线
    for i in range(len(p_2D2_array)):
        cv2.circle(combined_image, tuple((p_2D1_array[i]+ (source_Image.shape[1], 0)).astype(int)), 5, (0, 0, 255), -1)
        cv2.circle(combined_image, tuple((p_2D2_array[i] + (source_Image.shape[1], 0)).astype(int)), 5, (255, 0, 0), -1)
        cv2.line(combined_image, tuple((p_2D1_array[i] + (source_Image.shape[1], 0)).astype(int)), tuple((p_2D2_array[i] + (source_Image.shape[1], 0)).astype(int)), (0, 255, 0), 1)

    # 在拼接后的图像中显示平均误差值
    cv2.putText(combined_image, f'Avg_e = {avg_e:.2f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 显示拼接后的图像
    cv2.imshow('Reprojection Dual', combined_image)
    cv2.waitKey(10)
    # cv2.destroyAllWindows()


# def depth_to_point_cloud(depth, target_intrinsic):
#     # 获取图像尺寸
#     height, width = depth.shape

#     # 构建像素网格
#     u, v = np.meshgrid(np.arange(width), np.arange(height))
#     uv1 = np.stack([u, v, np.ones_like(u)], axis=-1)

#     # 将深度值扩展为三通道，以匹配相机坐标的形状
#     depth_3d = np.repeat(depth[:, :, np.newaxis], 3, axis=2)

#     # 将像素坐标转换为相机坐标系下的三维坐标
#     K_inv = np.linalg.inv(target_intrinsic)
#     points_camera = depth_3d * np.dot(uv1, K_inv.T)

#     # 将相机坐标系下的三维坐标转换为世界坐标系下的三维坐标
#     points_world = points_camera.reshape(height, width, 3)

    
#     # 将点云数据转换为PCL点云格式
#     cloud = pcl.PointCloud()
#     cloud.from_array(points_world.reshape(-1, 3).astype(np.float32))
#     cloud = downsample_point_cloud_voxel_grid(cloud, leaf_size=0.3)
#     # cloud = downsample_point_cloud_with_line_feature(cloud, max_dist=0.1) 
#     # cloud=downsample_point_cloud(cloud, 0.5)

    return cloud

# def downsample_point_cloud_voxel_grid(cloud, leaf_size):
#     """
#     对点云进行体素网格滤波下采样。

#     Parameters:
#         cloud (pcl.PointCloud): 输入的点云数据。
#         leaf_size (float): 体素网格的叶子大小。

#     Returns:
#         downsampled_cloud (pcl.PointCloud): 下采样后的点云数据。
#     """
#     sor = cloud.make_voxel_grid_filter()
#     sor.set_leaf_size(leaf_size, leaf_size, leaf_size)
#     downsampled_cloud = sor.filter()

#     return downsampled_cloud


def visualize_depth_cv2(depth_map):
    # 将深度图缩放到 0 到 255 范围内
    depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
    depth_map_normalized = depth_map_normalized.astype(np.uint8)

    # 使用灰度色彩图显示深度图
    depth_map_color = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_RAINBOW)

    # 显示深度图
    cv2.imshow('Depth Map', depth_map_color)
    cv2.waitKey(10)
    # cv2.destroyAllWindows()

def show_keypoints_on_image_cv2(image, keypoints, window_name="Image with Keypoints"):
    """
    在图像上使用 OpenCV 显示关键点。

    参数：
        image: 输入的图像（numpy数组）。
        keypoints: 关键点的坐标，应该是一个形状为 (N, 2) 的数组，
                   其中 N 是关键点的数量，每个关键点用 (x, y) 坐标表示。
    """
    # 将图像从 RGB 转换为 BGR（因为 OpenCV 使用的是 BGR 格式）
    image_bgr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_bgr = image_bgr.astype(np.uint8)
    # print(image_bgr.shape)
    # print(keypoints.shape)
    # cv2.imshow('Image with Keypoints1', image_bgr)
    # cv2.imshow('Image with Keypoints2', image)
    # 在图像上绘制关键点
    for kp in keypoints:
        x, y = kp
        cv2.circle(image_bgr, (int(x), int(y)), 3, (0, 0, 255), -1)  # 在关键点处绘制红色的实心圆

    # 显示图像
    cv2.imshow(window_name, image_bgr)
    

    # 保存图像
    image_bgr_save=image_bgr*255
    filename = f'{window_name.replace(" ", "_")}.png'
    cv2.imwrite(filename, image_bgr_save)
    cv2.waitKey(10)

def show_image_cv2(image, window_name="Image with Keypoints"):
    """
    在图像上使用 OpenCV 显示关键点。

    参数：
        image: 输入的图像（numpy数组）。
        keypoints: 关键点的坐标，应该是一个形状为 (N, 2) 的数组，
                   其中 N 是关键点的数量，每个关键点用 (x, y) 坐标表示。
    """
    # 将图像从 RGB 转换为 BGR（因为 OpenCV 使用的是 BGR 格式）
    image = image.cpu().numpy().squeeze() 
    image = np.transpose(image, (1, 2, 0))
    # image = image.astype(np.uint8)
    # image=image/255
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)#修改颜色
    # image_bgr = image_bgr.astype(np.uint8)
    # print(image_bgr.shape)
    # print(keypoints.shape)
    # cv2.imshow('Image with Keypoints1', image_bgr)
    # cv2.imshow('Image with Keypoints2', image)
    # 在图像上绘制关键点

    # 显示图像
    cv2.imshow(window_name, image_bgr)
    
    cv2.waitKey(10)

def need_to_add_new_keyframe(current_frame, prev_keyframe, motion_threshold=0.1, point_count_threshold=100):
    # 1. 检测相机运动量是否超过阈值
    motion = compute_motion(current_frame, prev_keyframe)
    if motion > motion_threshold:
        return True
    
    # 2. 检测地图点数量是否超过阈值
    if len(current_frame.landmarks) > point_count_threshold:
        return True
    
    # 其他条件检测...
    
    return False



def minimize_reprojection_error(rvecs, t, point1_3D,point2_2D, K,source_Image):
    # print("before_t=",t)
    # 相机位姿参数
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    lsat_cost=0
    # rvec, _ = cv2.Rodrigues(R)
    pose = np.concatenate((t.flatten(), rvecs.flatten()))
    # pose = np.array([0,0,0,0,0,0], dtype=np.float64) 
    for iter in range(10):
        cost = 0
        H = 0
        b = 0
        p_2D1_unrob_list=[]
        p_2D2_unrob_list=[]
        p_2D1_list=[]
        p_2D2_list=[]
        count = 0
        #求平均的e
        print(point1_3D.shape)
        print(point2_2D.shape)
        for i in range(point1_3D.shape[0]):
            # print(point1_3D.shape[0])
            R,j = cv2.Rodrigues(pose[3:6])   # 旋转矩阵
            t = pose[0:3]                   # 平移矩阵
            # p_3D = np.dot(R,point1_3D[i])+np.squeeze(t) # 3D坐标点、
            p_3D = np.dot(R,point1_3D[i])+np.squeeze(t)
            print(point1_3D[i])
            print(p_3D)
            # p_3D = point1_3D[i]
            X = p_3D[0]                     #
            Y = p_3D[1]                     #
            inv_z = 1.0/p_3D[2]             # 深度距离
            inv_z2 = inv_z*inv_z            # 深度距离的平方
            p_2D = np.dot(K,p_3D)[0:2]*inv_z# 重投影   一开始忘记归一化深度了
            e = point2_2D[i] - p_2D         # 误差向量 一开始写反了  cost一直增大
            # if np.linalg.norm(e)> 15: #初步筛选出离谱的点
            #     continue
            p_2D1_unrob_list.append(point2_2D[i])
            p_2D2_unrob_list.append(p_2D)
            cost += np.linalg.norm(e)       # 向量的模 累加
            count += 1
        avg_e=cost/count
        for i in range(point1_3D.shape[0]):
            # print(point1_3D.shape[0])
            R,j = cv2.Rodrigues(pose[3:6])   # 旋转矩阵
            t = pose[0:3]                   # 平移矩阵
            # p_3D = np.dot(R,point1_3D[i])+np.squeeze(t) # 3D坐标点、
            p_3D = np.dot(point1_3D[i],R)+np.squeeze(t)
            # p_3D = point1_3D[i]
            X = p_3D[0]                     #
            Y = p_3D[1]                     #
            inv_z = 1.0/p_3D[2]             # 深度距离
            inv_z2 = inv_z*inv_z            # 深度距离的平方
            p_2D = np.dot(K,p_3D)[0:2]*inv_z# 重投影   一开始忘记归一化深度了
            e = point2_2D[i] - p_2D         # 误差向量 一开始写反了  cost一直增大
            # if np.linalg.norm(e)>(avg_e+2) : 
            #     continue
            cost += np.linalg.norm(e)       # 向量的模 累加
            # print("e:",np.linalg.norm(e))
            J = [[-fx * inv_z,           0, fx * X * inv_z2,      fx * X * Y * inv_z2, -fx - fx * X * X * inv_z2,  fx * Y * inv_z],
                [          0, -fy * inv_z, fy * Y * inv_z2, fy + fy * Y * Y * inv_z2,      -fy * X * Y * inv_z2, -fy * X * inv_z]]
            J = np.array(J)          # 以上建立雅克比矩阵
            H += np.dot(J.T,J)       # 得到H
            b += np.dot(-J.T,e)      # 得到b 
            p_2D1_list.append(point2_2D[i])
            p_2D2_list.append(p_2D)
        
        
        p_2D1_unrob_array = np.array(p_2D1_unrob_list) 
        p_2D2_unrob_array = np.array(p_2D2_unrob_list) 
        p_2D1_array = np.array(p_2D1_list) 
        p_2D2_array = np.array(p_2D2_list) 
        plot_reproject(p_2D1_unrob_array,p_2D2_unrob_array,source_Image,avg_e,"unrob")
        plot_reproject(p_2D1_array,p_2D2_array,source_Image,avg_e,"rob")
        plot_reproject_con(p_2D1_unrob_array,p_2D2_unrob_array,p_2D1_array,p_2D2_array,source_Image,avg_e)

        if iter>0 and cost>=last_cost:# 误差变大则退出
            print('cost = ', cost,' last_cost = ',last_cost)
            break
        last_cost = cost
        if np.linalg.det(H) == 0:     # 奇异矩阵 
            # print('Singular matrix')
            break
        x = np.linalg.solve(H,b)     # 求解方程组 得到增量
        pose += x                    # 更新位姿
        # print(np.linalg.norm(x))
        if np.linalg.norm(x) < 1e-6: # 增量很小 
            # print(x)
            break
        # print('%d: %f'%(iter,cost))
    # print("匹配点数：%d"%(point1_3D.shape[0]))

    R,j = cv2.Rodrigues(pose[3:6])
    ta=pose[0:3].reshape(3,1)
    # print('R = ', R)
    return R,ta
    

class SLAM(object):
  def __init__(self, W, H, K):
    # main classes
    self.mapp = Map()

    # params
    self.W, self.H = W, H
    self.K = K

  def process_frame(self, keyframe, pose=None, verts=None):
    start_time = time()
    img = keyframe.data['img']
    assert img.shape[0:2] == (self.H, self.W)
    #定义frame的时候需要把global_pose赋值进去
    frame = Frame(self.mapp, keyframe, self.K, verts=verts)


    if frame.id == 0:
      return

    f1 = self.mapp.frames[-1]
    f2 = self.mapp.frames[-2]

    #这里面的东西全部需要替换：
    #idx1和idx2是list
    #Rt是相对位姿：前端得到
    idx1, idx2, Rt = match_frames(f1, f2)

    # add new observations if the point is already observed in the previous frame
    # TODO: consider tradeoff doing this before/after search by projection
    for i,idx in enumerate(idx2):
      if f2.pts[idx] is not None and f1.pts[idx1[i]] is None:
        f2.pts[idx].add_observation(f1, idx1[i])


    #这里的poss应该是改为直接赋值
    if frame.id < 5 or True:
      # get initial positions from fundamental matrix
    #   print("f1.pose:",f1.pose)
    #   f1.pose = np.dot(f2.pose,Rt)#其实算的就是f1位姿的逆
      f1.pose = np.dot(Rt,f2.pose)
    #   f1.pose = np.linalg.inv(f1.pose)
        # f1.pose = f1.pose 
    #   print("f1.pose:",f1.pose)
    else:
      # kinematic model (not used)
      velocity = np.dot(f2.pose, np.linalg.inv(self.mapp.frames[-3].pose))
      f1.pose = np.dot(velocity, f2.pose)

    # pose optimization
    if pose is None:
    #   print(np.linalg.inv(f1.pose))
      #print(f1.pose)
    #   pose_opt = self.mapp.optimize(local_window=1, fix_points=True)
    #   print("Pose:     %f" % pose_opt)
      print(np.linalg.inv(f1.pose))
    #   print(f1.pose)
    else:
      # have ground truth for pose
      f1.pose = pose
        
    sbp_pts_count = 0

    # search by projection
    if len(self.mapp.points) > 0:
      # project *all* the map points into the current frame
      map_points = np.array([p.homogeneous() for p in self.mapp.points])
      projs = np.dot(np.dot(self.K, f1.pose[:3]), map_points.T).T#每一行对应一个地图点的投影坐标
      projs = projs[:, 0:2] / projs[:, 2:]#齐次坐标归一化为2D图像坐标

      # only the points that fit in the frame
      good_pts = (projs[:, 0] > 0) & (projs[:, 0] < self.W) & \
                 (projs[:, 1] > 0) & (projs[:, 1] < self.H)

      for i, p in enumerate(self.mapp.points):
        if not good_pts[i]:
          # point not visible in frame
          continue
        if f1 in p.frames:
          # we already matched this map point to this frame
          # TODO: understand this better
          continue
        for m_idx in f1.kd.query_ball_point(projs[i], 2):
          # if point unmatched
          if f1.pts[m_idx] is None:
            b_dist = p.orb_distance(f1.des[m_idx])
            # if any descriptors within 64
            if b_dist < 64.0:
              p.add_observation(f1, m_idx)
              sbp_pts_count += 1
              break


    # triangulate the points we don't have matches for
    good_pts4d = np.array([f1.pts[i] is None for i in idx1])

    # print("f1.kps[idx1]:",f1.kps[idx1])
    # do triangulation in global frame
    pts4d = triangulate(f1.pose, f2.pose, f1.kps[idx1], f2.kps[idx2])
    good_pts4d &= np.abs(pts4d[:, 3]) != 0
    pts4d /= pts4d[:, 3:]       # homogeneous 3-D coords
    # print("pts4d:",pts4d)

    #这个地方应该需要用到depth图，得到对应的深度值
    # good_pts4d = np.array([f1.pts[i] is None for i in idx1])
    # pts4d = f1.raw_depth,f1.kpus[idx1]


    # adding new points to the map from pairwise matches
    new_pts_count = 0
    for i,p in enumerate(pts4d):
      if not good_pts4d[i]:
        continue

      # check parallax is large enough
      # TODO: learn what parallax means
      """
      r1 = np.dot(f1.pose[:3, :3], add_ones(f1.kps[idx1[i]]))
      r2 = np.dot(f2.pose[:3, :3], add_ones(f2.kps[idx2[i]]))
      parallax = r1.dot(r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
      if parallax >= 0.9998:
        continue
      """

      # check points are in front of both cameras
      pl1 = np.dot(f1.pose, p)
      pl2 = np.dot(f2.pose, p)
      if pl1[2] < 0 or pl2[2] < 0:
        continue

      # reproject
      pp1 = np.dot(self.K, pl1[:3])
      pp2 = np.dot(self.K, pl2[:3])

      # check reprojection error
      pp1 = (pp1[0:2] / pp1[2]) - f1.kpus[idx1[i]]
      pp2 = (pp2[0:2] / pp2[2]) - f2.kpus[idx2[i]]
      pp1 = np.sum(pp1**2)
      pp2 = np.sum(pp2**2)
      if pp1 > 2 or pp2 > 2:
        continue

      # add the point
      try:
        color = img[int(round(f1.kpus[idx1[i],1])), int(round(f1.kpus[idx1[i],0]))]
      except IndexError:
        color = (255,0,0)
      pt = Point(self.mapp, p[0:3], color)
      pt.add_observation(f2, idx2[i])
      pt.add_observation(f1, idx1[i])
      new_pts_count += 1

    print("Adding:   %d new points, %d search by projection" % (new_pts_count, sbp_pts_count))

    # optimize the map
    print(np.linalg.inv(f1.pose))
    if frame.id >= 4 and frame.id%5 == 0:
    #   err = self.mapp.optimize() #verbose=True)
    #   print("Optimize: %f units of error" % err)
        print(np.linalg.inv(f1.pose))

    print("Map:      %d points, %d frames" % (len(self.mapp.points), len(self.mapp.frames)))
    print("Time:     %.2f ms" % ((time()-start_time)*1000.0))
    print(np.linalg.inv(f1.pose))





    

class DFVO():
    def __init__(self, cfg):
        """
        Args:
            cfg (edict): configuration reading from yaml file
        """
        # configuration
        self.cfg = cfg

        # tracking stage
        self.tracking_stage = 0

        # predicted global poses
        self.global_poses = {0: SE3()}

        # reference data and current data
        self.initialize_data()

        self.keyframe_list = []
        self.ref_keyframe = KeyFrame(0,{})
        self.setup()

    def setup(self):
        """Reading configuration and setup, including

            - Timer
            - Dataset
            - Tracking method
            - Keypoint Sampler
            - Deep networks
            - Deep layers
            - Visualizer
        """
        # timer
        self.timers = Timer()

        # intialize dataset
        self.dataset = Dataset.datasets[self.cfg.dataset](self.cfg)
        
        # get tracking method
        self.tracking_method = self.cfg.tracking_method
        self.initialize_tracker()

        # initialize keypoint sampler
        self.kp_sampler = KeypointSampler(self.cfg)
        
        # Deep networks
        self.deep_models = DeepModel(self.cfg)
        self.deep_models.initialize_models()
        if self.cfg.online_finetune.enable:
            self.deep_models.setup_train()
        


        # Depth consistency
        if self.cfg.kp_selection.depth_consistency.enable:
            self.depth_consistency_computer = DepthConsistency(self.cfg, self.dataset.cam_intrinsics)




        # visualization interface
        self.drawer = FrameDrawer(self.cfg.visualization)
        
    def initialize_data(self):
        """initialize data of current view and reference view
        """
        self.ref_data = {}
        self.cur_data = {}

    def initialize_tracker(self):
        """Initialize tracker
        """
        if self.tracking_method == 'hybrid':
            self.e_tracker = EssTracker(self.cfg, self.dataset.cam_intrinsics, self.timers)
            self.pnp_tracker = PnpTracker(self.cfg, self.dataset.cam_intrinsics)
        elif self.tracking_method == 'PnP':
            self.pnp_tracker = PnpTracker(self.cfg, self.dataset.cam_intrinsics)
        elif self.tracking_method == 'deep_pose':
            return
        else:
            assert False, "Wrong tracker is selected, choose from [hybrid, PnP, deep_pose]"

    def update_global_pose(self, new_pose, scale=1.):
        """update estimated poses w.r.t global coordinate system

        Args:
            new_pose (SE3): new pose
            scale (float): scaling factor
        """
        self.cur_data['pose'].t = self.cur_data['pose'].R @ new_pose.t * scale \
                            + self.cur_data['pose'].t
        self.cur_data['pose'].R = self.cur_data['pose'].R @ new_pose.R
        self.global_poses[self.cur_data['id']] = copy.deepcopy(self.cur_data['pose'])
        #关键帧根据id就可以得到global_poses

    def update_keyframe_global_pose(self, nR,nt, scale=1.):
        self.cur_data['pose'].t= self.keyframe_list[-1].data['pose'].R @ nt * scale \
                            +  self.keyframe_list[-1].data['pose'].t
        self.cur_data['pose'].R = self.keyframe_list[-1].data['pose'].R @ nR
        self.global_poses[self.cur_data['id']] = copy.deepcopy(self.cur_data['pose'])
    
    def kp_pnp_track(self,keypoint_net, depth_net,superglue_net, target_image, source_image, target_intrinsic,top_k=500):
        def keep_top_k(score, uv, feat, top_k):
            B, C, Hc, Wc = feat.shape
            # print("feat.shape:"+str(feat.shape))
            # Get top_k keypoint mask
            top_k_score, top_k_indices = score.view(B,Hc*Wc).topk(top_k, dim=1, largest=True)
            top_k_mask = torch.zeros(B, Hc * Wc).to(score.device)
            top_k_mask.scatter_(1, top_k_indices, value=1)
            top_k_mask = top_k_mask.gt(1e-3).view(Hc,Wc)
            uv    = uv.squeeze().permute(1,2,0)
            feat  = feat.squeeze().permute(1,2,0)
            top_k_uv    = uv[top_k_mask].view(top_k, 2)
            top_k_feat  = feat[top_k_mask].view(top_k, C)
            return top_k_score, top_k_uv, top_k_feat, top_k_mask

        # Set models to eval
        keypoint_net.eval()
        depth_net.eval()
        superglue_net.eval()

        # print("!!!!!!!!!!!!!!!!!!"+str(target_intrinsic))
        # Get dimensions
        B, _, H, W = target_image.shape
        # print("|||||||||||||||||||||||||||||||||||||||target_image.shape:"+str(target_image.shape))
        target_image_display=target_image.clone()
        source_image_display=source_image.clone()
        # Extract target and source keypoints, descriptors and score
        target_image = to_color_normalized(target_image.clone())
        target_score, target_uv, target_feat = keypoint_net(target_image)
        

        source_image = to_color_normalized(source_image.clone())
        ref_score, ref_uv, ref_feat = keypoint_net(source_image)
        ref_uv_pnp = ref_uv.clone()
        # ref_feat_pnp = ref_feat.clone()
        intrinsic_matrix=target_intrinsic.cpu().numpy().squeeze()
        
        # Sample (sparse) target keypoint depth
        target_uv_norm = target_uv.clone()
        target_uv_norm[:,0] = (target_uv_norm[:,0] / (float(W-1)/2.)) - 1.
        target_uv_norm[:,1] = (target_uv_norm[:,1] / (float(H-1)/2.)) - 1.
        target_uv_norm = target_uv_norm.permute(0, 2, 3, 1)

        ref_uv_norm = ref_uv.clone()
        ref_uv_norm[:,0] = (ref_uv_norm[:,0] / (float(W-1)/2.)) - 1.
        ref_uv_norm[:,1] = (ref_uv_norm[:,1] / (float(H-1)/2.)) - 1.
        ref_uv_norm = ref_uv_norm.permute(0, 2, 3, 1)


        img_list = [self.ref_data['img']]
        # print("self.ref_data['img']:",self.ref_data['img'].shape)
        self.ref_data['raw_depth'] = self.deep_models.forward_depth(imgs=img_list)
        self.ref_data['raw_depth'] = cv2.resize(self.ref_data['raw_depth'],
                                                    (self.cfg.image.width, self.cfg.image.height),
                                                    interpolation=cv2.INTER_NEAREST
                                                    )
        # print("self.ref_data['raw_depth']",self.ref_data['raw_depth'].shape)
        target_d = self.ref_data['raw_depth']
        target_i = self.ref_data['img']
        ref_d = self.cur_data['raw_depth']
        ref_i = self.cur_data['img']
        # print("target_d:",target_d.shape)
        # print("ref_d:",ref_d.shape)

        # Compute target depth
        # target_inv_depth = depth_net(target_image)
        # target_depth_np = target_inv_depth.squeeze().cpu().detach().numpy()

        # ref_inv_depth = depth_net(source_image)
        # ref_depth_np = ref_inv_depth.squeeze().cpu().detach().numpy()

        # 将深度图转换为点云，保存点云数据
        # point_cloud = depth_to_point_cloud(target_d, intrinsic_matrix)
        # pcl.save(point_cloud, 'point_cloud.pcd')

        visualize_depth_cv2(target_d)

        target_d = torch.tensor(target_d)  # 转换为 PyTorch 张量
        ref_d = torch.tensor(ref_d)  # 转换为 PyTorch 张量

        # 在第一个和第二个维度上增加维度，变成四维张量
        target_d = torch.unsqueeze(torch.unsqueeze(target_d, 0), 0)
        ref_d = torch.unsqueeze(torch.unsqueeze(ref_d, 0), 0)

        # print("target_d shape:", target_d.shape)  # 输出 torch.Size([1, 1, 192, 640])
        # print("ref_d shape:", ref_d.shape)  # 输出 torch.Size([1, 1, 192, 640])
        
        # target_depth_all = 1 / target_inv_depth.clamp(min=1e-6)
        target_depth_all = target_d.cuda()
        target_depth = torch.nn.functional.grid_sample(target_depth_all, target_uv_norm.detach(), mode='bilinear')

        # ref_depth_all = 1 / ref_inv_depth.clamp(min=1e-6)
        ref_depth_all = ref_d.cuda()
        ref_depth = torch.nn.functional.grid_sample(ref_depth_all, ref_uv_norm.detach(), mode='bilinear')

        # Get top_k source keypoints
        ref_score, ref_uv, ref_feat, ref_top_k_mask = keep_top_k(ref_score, ref_uv, ref_feat, top_k=top_k)
        # print("source_image.shape:"+str(source_image.shape))
        # print("source_image.shape:"+str(ref_uv.shape))
        # cv2.imshow('Image with Keypoints3', source_image_display[0].permute(1, 2, 0).cpu().numpy())
        show_keypoints_on_image_cv2(source_image_display[0].permute(1, 2, 0).cpu().numpy(), ref_uv.squeeze().cpu().detach().numpy(),"source_Image with Keypoints")
        # Get top_k target keypoints
        target_score, target_uv, target_feat, target_top_k_mask = keep_top_k(target_score, target_uv, target_feat, top_k=top_k)
        show_keypoints_on_image_cv2(target_image_display[0].permute(1, 2, 0).cpu().numpy(), ref_uv.squeeze().cpu().detach().numpy(),"target_Image with Keypoints")
        # Get corresponding target top_k depth
        target_depth = target_depth.squeeze()
        target_depth = target_depth[target_top_k_mask]

        ref_depth = ref_depth.squeeze()
        ref_depth = ref_depth[ref_top_k_mask]
        
        # Create target sparse point cloud
        target_cam = Camera(K=target_intrinsic.float()).to(target_image.device)
        target_uvz = torch.cat([target_uv, target_depth.unsqueeze(1)], 1).t().unsqueeze(1)
        target_xyz = target_cam.reconstruct_sparse(target_uvz.view(B,3,-1), frame='c').squeeze().t()

        
        ref_cam = Camera(K=target_intrinsic.float()).to(source_image.device)
        ref_uvz = torch.cat([ref_uv, ref_depth.unsqueeze(1)], 1).t().unsqueeze(1)
        ref_xyz = ref_cam.reconstruct_sparse(ref_uvz.view(B,3,-1), frame='c').squeeze().t()


        ref_uv_backen = ref_uv.clone()
        ref_feat_backen = ref_feat.clone()

        #superglue进行match
        # 对目标特征、UV、得分和图像进行形状调整
        target_feat_reshaped = target_feat.unsqueeze(0).permute(0, 2, 1)  # 调整特征的形状为(B, C, N)
        target_feat_reshaped = target_feat_reshaped.repeat(1, 4, 1) 
        target_uv_reshaped = target_uv.unsqueeze(0).permute(0, 1, 2)  # 调整UV的形状为(B, 2, N)

        # 对参考特征、UV、得分和图像进行形状调整
        ref_feat_reshaped = ref_feat.unsqueeze(0).permute(0, 2, 1)  # 调整特征的形状为(B, C, N)
        ref_feat_reshaped = ref_feat_reshaped.repeat(1, 4, 1) 
        ref_uv_reshaped = ref_uv.unsqueeze(0).permute(0, 1, 2)  # 调整UV的形状为(B, 2, N)
        # ref_score_reshaped = ref_score.squeeze(0)  # 去除得分张量的第一个维度

        # print('1111111'+str(target_score.shape))
        target_image_size = torch.tensor([target_image.size(2), target_image.size(3)])
        ref_image_size = torch.tensor([source_image.size(2), source_image.size(3)])
        target_image_size=target_image_size.unsqueeze(0)
        ref_image_size=ref_image_size.unsqueeze(0)
        # print(target_image_size.shape)


        match_data={
            'descriptors0': target_feat_reshaped,
            'descriptors1': ref_feat_reshaped,
            'keypoints0': target_uv_reshaped,
            'keypoints1': ref_uv_reshaped,
            'scores0': target_score,
            'scores1': ref_score,
            'image_size0': (target_image.size(2), target_image.size(3)),  # 图像尺寸
            'image_size1': (source_image.size(2), source_image.size(3)),
        }
        
        matches = superglue_net(match_data)
        # for key, value in matches.items():
        #     print("键名:", key)
        #     print("形状:", value.shape)



        # 将 PyTorch 张量转换为 NumPy 数组
        source_image_match_display = source_image_display[0].permute(1, 2, 0).cpu().numpy()
        target_image_match_display = target_image_display[0].permute(1, 2, 0).cpu().numpy()
        
        # source_image_match_display = np.uint8(source_image_match_display)
        # target_image_match_display = np.uint8(target_image_match_display)
        
        source_image_display_bgr = cv2.cvtColor(source_image_match_display, cv2.COLOR_BGR2RGB)
        source_image_display_bgr= source_image_display_bgr.astype(np.uint8)
        target_image_display_bgr = cv2.cvtColor(target_image_match_display, cv2.COLOR_BGR2RGB)
        target_image_display_bgr = target_image_display_bgr.astype(np.uint8)
        
        
        # # 显示拼接后的图像
        # cv2.imshow('Combined Image', combined_image)
        # cv2.waitKey(10)
        # 获取图像尺寸
        image_size0 = match_data['image_size0']
        image_size1 = match_data['image_size1']

        
        combined_image = cv2.hconcat([target_image_display_bgr, source_image_display_bgr])
        # print("|||||||||||||||||||"+str(type(combined_image)))

        # 获取匹配关系和特征点
        matches0 = matches['matches0'].cpu().detach().numpy()
        matches1 = matches['matches1'].cpu().detach().numpy()
        keypoints0 = match_data['keypoints0'].cpu().detach().numpy()
        keypoints1 = match_data['keypoints1'].cpu().detach().numpy()
        score0=matches['matching_scores0' ].cpu().detach().numpy()
        # #打印匹配点索引
        # j=0
        # for i in range(matches0.shape[1]):
        #     if matches0[0, i] != -1 and score0[0,i]>0.7:
        #         j+=1
        # print("匹配点对有："+str(j))


        # 绘制匹配点和连线
        for i in range(matches0.shape[1]):
            if matches0[0, i] != -1 and score0[0,i]>0.6:  # 如果有匹配点
                pt1 = (int(keypoints0[0, i, 0]), int(keypoints0[0, i, 1]))  # 第一张图像中的关键点坐标
                pt2 = (int(keypoints1[0, matches0[0, i], 0]) + image_size0[1], int(keypoints1[0, matches0[0, i], 1]))  # 第二张图像中的关键点坐标
                cv2.circle(combined_image, pt1, 3, (0, 0, 255), -1)  # 在第一张图像上绘制关键点
                cv2.circle(combined_image, pt2, 3, (0, 0, 255), -1)  # 在第二张图像上绘制关键点
                cv2.line(combined_image, pt1, pt2, (0, 255, 0), 1)  # 在新图像上绘制匹配线
                # print(score0[0,i])
        # 显示可视化结果
        cv2.imshow('Matches', combined_image)
        # print(combined_image*255)
        combined_image_save=combined_image*255
        # print("999999999999"+str(combined_image.shape))
        cv2.waitKey(10)

        # 保存图像
        # Image.fromarray(combined_image).save('combined_image.png')
        cv2.imwrite('combined_image.jpg', combined_image_save)
        

        #Compute descriptor matches using superglue（新）
        matches0 = matches['matches0']
        matches0_np = matches0[0].cpu().detach().numpy()
        matches1 = matches['matches1']
        matches1_np = matches1[0].cpu().detach().numpy()
        # print()
        # 获取匹配点的索引
        target_idx = np.arange(480)
        # print(target_idx) 
        ref_idx = matches0_np

        # print(target_idx)
        # print(ref_idx)
        pt0 = np.empty((0, 2), dtype=np.float32)
        pt1 = np.empty((0, 3), dtype=np.float32)
        pt2 = np.empty((0, 2), dtype=np.float32)
        pt3 = np.empty((0, 3), dtype=np.float32)
        # pt3 = np.empty((0, 3), dtype=np.int32)
        target_xyz=target_xyz.cpu().detach().numpy()
        target_uv=target_uv.cpu().detach().numpy()
        ref_uv=ref_uv.cpu().detach().numpy()
        ref_xyz=ref_xyz.cpu().detach().numpy()
        # pt1 = target_xyz[target_idx].cpu().numpy()
        # pt2 = ref_uv[ref_idx].cpu().numpy()
        
        for i in range(matches0.shape[1]):
            if matches0[0, i] != -1 and score0[0,i]:  # 如果有匹配点
                pt0 = np.append(pt0, [[(keypoints0[0, i, 0]), (keypoints0[0, i, 1])]], axis=0)  # Keypoints coordinates of the first image
                pt1 = np.append(pt1, [target_xyz[i]], axis=0)
                pt2 = np.append(pt2, [[(keypoints1[0, matches0[0, i], 0]), (keypoints1[0, matches0[0, i], 1])]], axis=0)  # # 第二张图像中的关键点坐标
                pt3 = np.append(pt3, [ref_xyz[matches0[0, i]]], axis=0)
        
        # print(pt0.shape)
        # print(pt1.shape)

        
        combined_image = cv2.hconcat([target_image_display_bgr, source_image_display_bgr])
        # 绘制匹配的关键点并连线
        for i in range(len(pt0)):
            pt0_x, pt0_y = int(pt0[i][0]), int(pt0[i][1])
            pt2_x, pt2_y = int(pt2[i][0] + image_size0[1]), int(pt2[i][1])
            cv2.circle(combined_image, (pt0_x, pt0_y), 5, (0, 0, 255), -1)  # 绘制目标图像的关键点
            cv2.circle(combined_image, (pt2_x, pt2_y), 5, (255, 0, 0), -1)  # 绘制参考图像的关键点
            cv2.line(combined_image, (pt0_x, pt0_y), (pt2_x, pt2_y), (0, 255, 0), 1)  # 绘制连接线
            

        # 显示拼接后的图像
        cv2.imshow('Combined Image', combined_image)
        cv2.waitKey(10)

        # Run PnP and get pose
        # K = target_intrinsic.float().cpu().detach().numpy().squeeze()
        # 2D->2D
        # em, mask = cv2.findEssentialMat(pt0, pt2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        # retval, R, t, mask = cv2.recoverPose(em, pt0, pt2, K, mask=mask) 
        # 2D->3D
        # _, rvecs, t, inliers = cv2.solvePnPRansac(pt1, pt2, K, None, reprojectionError=0.85, iterationsCount=15000)
        # print("pt1:",pt1)
        # print("pt1:",pt1.shape)
        # print(inliers)
        # R = cv2.Rodrigues(rvecs)[0]
        # outputs = {}
        # pose=SE3()
        # pose.R = cv2.Rodrigues(rvecs)[0]
        # pose.t = t
        # pose.pose=pose.inv_pose
        # outputs['pose'] = pose
        return pt0,pt1,pt2,pt3,ref_uv_pnp,ref_uv_backen,ref_feat_backen

    def super_extract(self,keypoint_net, depth_net,superglue_net, source_image, target_intrinsic,top_k=500):
        def keep_top_k(score, uv, feat, top_k):
            B, C, Hc, Wc = feat.shape
            # print("feat.shape:"+str(feat.shape))
            # Get top_k keypoint mask
            top_k_score, top_k_indices = score.view(B,Hc*Wc).topk(top_k, dim=1, largest=True)
            top_k_mask = torch.zeros(B, Hc * Wc).to(score.device)
            top_k_mask.scatter_(1, top_k_indices, value=1)
            top_k_mask = top_k_mask.gt(1e-3).view(Hc,Wc)
            uv    = uv.squeeze().permute(1,2,0)
            feat  = feat.squeeze().permute(1,2,0)
            top_k_uv    = uv[top_k_mask].view(top_k, 2)
            top_k_feat  = feat[top_k_mask].view(top_k, C)
            return top_k_score, top_k_uv, top_k_feat, top_k_mask

        # Set models to eval
        keypoint_net.eval()
        depth_net.eval()
        superglue_net.eval()

        # print("!!!!!!!!!!!!!!!!!!"+str(target_intrinsic))
        # Get dimensions
        B, _, H, W = source_image.shape
        # print("|||||||||||||||||||||||||||||||||||||||target_image.shape:"+str(target_image.shape))
        # target_image_display=target_image.clone()
        source_image_display=source_image.clone()
        # Extract target and source keypoints, descriptors and score
        # target_image = to_color_normalized(target_image.clone())
        # target_score, target_uv, target_feat = keypoint_net(target_image)
        

        source_image = to_color_normalized(source_image.clone())
        ref_score, ref_uv, ref_feat = keypoint_net(source_image)
        ref_uv_pnp = ref_uv.detach().clone()
        # ref_feat_pnp = ref_feat.clone()
        intrinsic_matrix=target_intrinsic.cpu().numpy().squeeze()
        
        # Sample (sparse) target keypoint depth
        # target_uv_norm = target_uv.clone()
        # target_uv_norm[:,0] = (target_uv_norm[:,0] / (float(W-1)/2.)) - 1.
        # target_uv_norm[:,1] = (target_uv_norm[:,1] / (float(H-1)/2.)) - 1.
        # target_uv_norm = target_uv_norm.permute(0, 2, 3, 1)

        ref_uv_norm = ref_uv.clone()
        ref_uv_norm[:,0] = (ref_uv_norm[:,0] / (float(W-1)/2.)) - 1.
        ref_uv_norm[:,1] = (ref_uv_norm[:,1] / (float(H-1)/2.)) - 1.
        ref_uv_norm = ref_uv_norm.permute(0, 2, 3, 1)


        # img_list = [self.ref_data['img']]
        # print("self.ref_data['img']:",self.ref_data['img'].shape)
        # self.ref_data['raw_depth'] = self.deep_models.forward_depth(imgs=img_list)
        # self.ref_data['raw_depth'] = cv2.resize(self.ref_data['raw_depth'],
        #                                             (self.cfg.image.width, self.cfg.image.height),
        #                                             interpolation=cv2.INTER_NEAREST
        #                                             )
        # # print("self.ref_data['raw_depth']",self.ref_data['raw_depth'].shape)
        # target_d = self.ref_data['raw_depth']
        # target_i = self.ref_data['img']
        ref_d = self.cur_data['raw_depth']
        ref_i = self.cur_data['img']
        # print("target_d:",target_d.shape)
        # print("ref_d:",ref_d.shape)

        # Compute target depth
        # target_inv_depth = depth_net(target_image)
        # target_depth_np = target_inv_depth.squeeze().cpu().detach().numpy()

        # ref_inv_depth = depth_net(source_image)
        # ref_depth_np = ref_inv_depth.squeeze().cpu().detach().numpy()

        # 将深度图转换为点云，保存点云数据
        # point_cloud = depth_to_point_cloud(target_d, intrinsic_matrix)
        # pcl.save(point_cloud, 'point_cloud.pcd')

        visualize_depth_cv2(ref_d)

        # target_d = torch.tensor(target_d)  # 转换为 PyTorch 张量
        ref_d = torch.tensor(ref_d)  # 转换为 PyTorch 张量

        # 在第一个和第二个维度上增加维度，变成四维张量
        # target_d = torch.unsqueeze(torch.unsqueeze(target_d, 0), 0)
        ref_d = torch.unsqueeze(torch.unsqueeze(ref_d, 0), 0)

        # print("target_d shape:", target_d.shape)  # 输出 torch.Size([1, 1, 192, 640])
        # print("ref_d shape:", ref_d.shape)  # 输出 torch.Size([1, 1, 192, 640])
        
        # target_depth_all = 1 / target_inv_depth.clamp(min=1e-6)
        # target_depth_all = target_d.cuda()
        # target_depth = torch.nn.functional.grid_sample(target_depth_all, target_uv_norm.detach(), mode='bilinear')

        # ref_depth_all = 1 / ref_inv_depth.clamp(min=1e-6)
        ref_depth_all = ref_d.cuda()
        ref_depth = torch.nn.functional.grid_sample(ref_depth_all, ref_uv_norm.detach(), mode='bilinear')

        # Get top_k source keypoints
        ref_score, ref_uv, ref_feat, ref_top_k_mask = keep_top_k(ref_score, ref_uv, ref_feat, top_k=top_k)
        # print("source_image.shape:"+str(source_image.shape))
        # print("source_image.shape:"+str(ref_uv.shape))
        # cv2.imshow('Image with Keypoints3', source_image_display[0].permute(1, 2, 0).cpu().numpy())
        show_keypoints_on_image_cv2(source_image_display[0].permute(1, 2, 0).cpu().numpy(), ref_uv.squeeze().cpu().detach().numpy(),"source_Image with Keypoints")
        # Get top_k target keypoints
        # target_score, target_uv, target_feat, target_top_k_mask = keep_top_k(target_score, target_uv, target_feat, top_k=top_k)
        # show_keypoints_on_image_cv2(target_image_display[0].permute(1, 2, 0).cpu().numpy(), ref_uv.squeeze().cpu().detach().numpy(),"target_Image with Keypoints")
        # Get corresponding target top_k depth
        # target_depth = target_depth.squeeze()
        # target_depth = target_depth[target_top_k_mask]

        ref_depth = ref_depth.squeeze()
        ref_depth = ref_depth[ref_top_k_mask]
        
        # Create target sparse point cloud
        # target_cam = Camera(K=target_intrinsic.float()).to(target_image.device)
        # target_uvz = torch.cat([target_uv, target_depth.unsqueeze(1)], 1).t().unsqueeze(1)
        # target_xyz = target_cam.reconstruct_sparse(target_uvz.view(B,3,-1), frame='c').squeeze().t()

        
        ref_cam = Camera(K=target_intrinsic.float()).to(source_image.device)
        ref_uvz = torch.cat([ref_uv, ref_depth.unsqueeze(1)], 1).t().unsqueeze(1)
        ref_xyz = ref_cam.reconstruct_sparse(ref_uvz.view(B,3,-1), frame='c').squeeze().t()

        ref_uv = ref_uv.detach().clone()
        ref_feat = ref_feat.detach().clone()
        ref_uv_backen = ref_uv.clone()
        ref_feat_backen = ref_feat.clone()

        return ref_uv_pnp,ref_uv_backen,ref_feat_backen
    
    def match_features(self,keypoint_net, depth_net,superglue_net, target_image, source_image, target_intrinsic,rR,rt,top_k=1000):
        def keep_top_k(score, uv, feat, top_k):
            B, C, Hc, Wc = feat.shape
            # print("feat.shape:"+str(feat.shape))
            # Get top_k keypoint mask
            top_k_score, top_k_indices = score.view(B,Hc*Wc).topk(top_k, dim=1, largest=True)
            top_k_mask = torch.zeros(B, Hc * Wc).to(score.device)
            top_k_mask.scatter_(1, top_k_indices, value=1)
            top_k_mask = top_k_mask.gt(1e-3).view(Hc,Wc)
            uv    = uv.squeeze().permute(1,2,0)
            feat  = feat.squeeze().permute(1,2,0)
            top_k_uv    = uv[top_k_mask].view(top_k, 2)
            top_k_feat  = feat[top_k_mask].view(top_k, C)
            # print("top_k_mask",top_k_mask.shape)
            return top_k_score, top_k_uv, top_k_feat, top_k_mask

        # Set models to eval
        keypoint_net.eval()
        depth_net.eval()
        superglue_net.eval()

        # print("!!!!!!!!!!!!!!!!!!"+str(target_intrinsic))
        # Get dimensions
        B, _, H, W = target_image.shape
        # print("|||||||||||||||||||||||||||||||||||||||target_image.shape:"+str(target_image.shape))
        target_image_display=target_image.clone()
        source_image_display=source_image.clone()
        # print(source_image)
        show_image_cv2(target_image,'tar')
        show_image_cv2(source_image,'sur')
        # Extract target and source keypoints, descriptors and score
        target_image = to_color_normalized(target_image.clone())
        target_score, target_uv, target_feat = keypoint_net(target_image)
        

        source_image = to_color_normalized(source_image.clone())
        ref_score, ref_uv, ref_feat = keypoint_net(source_image)

        intrinsic_matrix=target_intrinsic.cpu().numpy().squeeze()
        
        # Sample (sparse) target keypoint depth
        target_uv_norm = target_uv.clone()
        print("target_uv_norm:",target_uv_norm.shape)
        target_uv_norm[:,0] = (target_uv_norm[:,0] / (float(W-1)/2.)) - 1.
        target_uv_norm[:,1] = (target_uv_norm[:,1] / (float(H-1)/2.)) - 1.
        target_uv_norm = target_uv_norm.permute(0, 2, 3, 1)
        print("target_uv_norm:",target_uv_norm.shape)
        ref_uv_norm = ref_uv.clone()
        ref_uv_norm[:,0] = (ref_uv_norm[:,0] / (float(W-1)/2.)) - 1.
        ref_uv_norm[:,1] = (ref_uv_norm[:,1] / (float(H-1)/2.)) - 1.
        ref_uv_norm = ref_uv_norm.permute(0, 2, 3, 1)


        img_list = [self.ref_keyframe.data['img']]
        # print("self.ref_keyframe.data['img']:",self.ref_keyframe.data['img'].shape)
        self.ref_keyframe.data['raw_depth'] = self.deep_models.forward_depth(imgs=img_list)
        self.ref_keyframe.data['raw_depth'] = cv2.resize(self.ref_keyframe.data['raw_depth'],
                                                    (self.cfg.image.width, self.cfg.image.height),
                                                    interpolation=cv2.INTER_NEAREST
                                                    )
        # print("self.ref_keyframe.data['raw_depth']",self.ref_keyframe.data['raw_depth'].shape)
        target_d = self.ref_keyframe.data['raw_depth']
        target_i = self.ref_keyframe.data['img']
        ref_d = self.cur_data['raw_depth']
        ref_i = self.cur_data['img']
        # print("target_d:",target_d.shape)
        # print("ref_d:",ref_d.shape)

        # Compute target depth
        # target_inv_depth = depth_net(target_image)
        # print("target_inv_depth:",target_inv_depth.shape)
        # target_depth_np = target_inv_depth.squeeze().cpu().detach().numpy()
        # print("target_depth_np:",target_depth_np.shape)
        # ref_inv_depth = depth_net(source_image)
        # ref_depth_np = ref_inv_depth.squeeze().cpu().detach().numpy()

        # 将深度图转换为点云，保存点云数据
        # point_cloud = depth_to_point_cloud(target_d, intrinsic_matrix)
        # pcl.save(point_cloud, 'point_cloud.pcd')

        visualize_depth_cv2(target_d)

        target_d = torch.tensor(target_d)  # 转换为 PyTorch 张量
        ref_d = torch.tensor(ref_d)  # 转换为 PyTorch 张量

        # 在第一个和第二个维度上增加维度，变成四维张量
        target_d = torch.unsqueeze(torch.unsqueeze(target_d, 0), 0)
        ref_d = torch.unsqueeze(torch.unsqueeze(ref_d, 0), 0)

        # print("target_d shape:", target_d.shape)  # 输出 torch.Size([1, 1, 192, 640])
        # print("ref_d shape:", ref_d.shape)  # 输出 torch.Size([1, 1, 192, 640])
        # 
        # target_depth_all = 1 / target_inv_depth.clamp(min=1e-6)

        target_depth_all = target_d.cuda()
        # print("target_depth_all:",target_depth_all.shape)
        target_depth = torch.nn.functional.grid_sample(target_depth_all, target_uv_norm.detach(), mode='bilinear')
        # print("target_depth:", target_depth.shape)
        # ref_depth_all = 1 / ref_inv_depth.clamp(min=1e-6)
        ref_depth_all = ref_d.cuda()
        ref_depth = torch.nn.functional.grid_sample(ref_depth_all, ref_uv_norm.detach(), mode='bilinear')

        # Get top_k source keypoints
        ref_score, ref_uv, ref_feat, ref_top_k_mask = keep_top_k(ref_score, ref_uv, ref_feat, top_k=top_k)
        # print("source_image.shape:"+str(source_image.shape))
        # print("source_image.shape:"+str(ref_uv.shape))
        # cv2.imshow('Image with Keypoints3', source_image_display[0].permute(1, 2, 0).cpu().numpy())
        show_keypoints_on_image_cv2(source_image_display[0].permute(1, 2, 0).cpu().numpy(), ref_uv.squeeze().cpu().detach().numpy(),"source_Image with Keypoints")
        show_keypoints_on_image_cv2(ref_i, ref_uv.squeeze().cpu().detach().numpy(),"ref_i with Keypoints")
        # Get top_k target keypoints
        target_score, target_uv, target_feat, target_top_k_mask = keep_top_k(target_score, target_uv, target_feat, top_k=top_k)
        show_keypoints_on_image_cv2(target_image_display[0].permute(1, 2, 0).cpu().numpy(), target_uv.squeeze().cpu().detach().numpy(),"target_Image with Keypoints")
        show_keypoints_on_image_cv2(target_i, target_uv.squeeze().cpu().detach().numpy(),"target_i with Keypoints2")
        # Get corresponding target top_k depth
        target_depth = target_depth.squeeze()
        # print("target_depth:",target_depth.shape)
        target_depth = target_depth[target_top_k_mask]
        # print("target_depth:",target_depth.shape)
        ref_depth = ref_depth.squeeze()
        ref_depth = ref_depth[ref_top_k_mask]
        
        # Create target sparse point cloud
        target_cam = Camera(K=target_intrinsic.float()).to(target_image.device)
        target_uvz = torch.cat([target_uv, target_depth.unsqueeze(1)], 1).t().unsqueeze(1)
        target_xyz = target_cam.reconstruct_sparse(target_uvz.view(B,3,-1), frame='c').squeeze().t()

        
        ref_cam = Camera(K=target_intrinsic.float()).to(source_image.device)
        ref_uvz = torch.cat([ref_uv, ref_depth.unsqueeze(1)], 1).t().unsqueeze(1)
        ref_xyz = ref_cam.reconstruct_sparse(ref_uvz.view(B,3,-1), frame='c').squeeze().t()

        

        # 对目标特征、UV、得分和图像进行形状调整
        target_feat_reshaped = target_feat.unsqueeze(0).permute(0, 2, 1)  # 调整特征的形状为(B, C, N)
        target_feat_reshaped = target_feat_reshaped.repeat(1, 4, 1) 
        target_uv_reshaped = target_uv.unsqueeze(0).permute(0, 1, 2)  # 调整UV的形状为(B, 2, N)

        # 对参考特征、UV、得分和图像进行形状调整
        ref_feat_reshaped = ref_feat.unsqueeze(0).permute(0, 2, 1)  # 调整特征的形状为(B, C, N)
        ref_feat_reshaped = ref_feat_reshaped.repeat(1, 4, 1) 
        ref_uv_reshaped = ref_uv.unsqueeze(0).permute(0, 1, 2)  # 调整UV的形状为(B, 2, N)
        # ref_score_reshaped = ref_score.squeeze(0)  # 去除得分张量的第一个维度

        # print('1111111'+str(target_score.shape))
        target_image_size = torch.tensor([target_image.size(2), target_image.size(3)])
        ref_image_size = torch.tensor([source_image.size(2), source_image.size(3)])
        target_image_size=target_image_size.unsqueeze(0)
        ref_image_size=ref_image_size.unsqueeze(0)
        # print(target_image_size.shape)


        match_data={
            'descriptors0': target_feat_reshaped,
            'descriptors1': ref_feat_reshaped,
            'keypoints0': target_uv_reshaped,
            'keypoints1': ref_uv_reshaped,
            'scores0': target_score,
            'scores1': ref_score,
            'image_size0': (target_image.size(2), target_image.size(3)),  # 图像尺寸
            'image_size1': (source_image.size(2), source_image.size(3)),
        }
        matches = superglue_net(match_data)
        # for key, value in matches.items():
        #     print("键名:", key)
        #     print("形状:", value.shape)



        # 将 PyTorch 张量转换为 NumPy 数组
        source_image_match_display = source_image_display[0].permute(1, 2, 0).cpu().numpy()
        target_image_match_display = target_image_display[0].permute(1, 2, 0).cpu().numpy()
        
        # source_image_match_display = np.uint8(source_image_match_display)
        # target_image_match_display = np.uint8(target_image_match_display)
        
        source_image_display_bgr = cv2.cvtColor(source_image_match_display, cv2.COLOR_BGR2RGB)
        # source_image_display_bgr= source_image_display_bgr.astype(np.uint8)
        target_image_display_bgr = cv2.cvtColor(target_image_match_display, cv2.COLOR_BGR2RGB)
        # target_image_display_bgr = target_image_display_bgr.astype(np.uint8)
        
        
        # # 显示拼接后的图像
        # cv2.imshow('Combined Image', combined_image)
        # cv2.waitKey(10)
        # 获取图像尺寸
        image_size0 = match_data['image_size0']
        image_size1 = match_data['image_size1']

        
        combined_image = cv2.hconcat([target_image_display_bgr, source_image_display_bgr])
        # print("|||||||||||||||||||"+str(type(combined_image)))

        # 获取匹配关系和特征点
        matches0 = matches['matches0'].cpu().detach().numpy()
        matches1 = matches['matches1'].cpu().detach().numpy()
        keypoints0 = match_data['keypoints0'].cpu().detach().numpy()
        keypoints1 = match_data['keypoints1'].cpu().detach().numpy()
        score0=matches['matching_scores0' ].cpu().detach().numpy()
        # #打印匹配点索引
        # j=0
        # for i in range(matches0.shape[1]):
        #     if matches0[0, i] != -1 and score0[0,i]>0.7:
        #         j+=1
        # print("匹配点对有："+str(j))


        # 绘制匹配点和连线
        for i in range(matches0.shape[1]):
            if matches0[0, i] != -1 and score0[0,i]>0.6:  # 如果有匹配点
                pt1 = (int(keypoints0[0, i, 0]), int(keypoints0[0, i, 1]))  # 第一张图像中的关键点坐标
                pt2 = (int(keypoints1[0, matches0[0, i], 0]) + image_size0[1], int(keypoints1[0, matches0[0, i], 1]))  # 第二张图像中的关键点坐标
                cv2.circle(combined_image, pt1, 3, (0, 0, 255), -1)  # 在第一张图像上绘制关键点
                cv2.circle(combined_image, pt2, 3, (0, 0, 255), -1)  # 在第二张图像上绘制关键点
                cv2.line(combined_image, pt1, pt2, (0, 255, 0), 1)  # 在新图像上绘制匹配线
                # print(score0[0,i])
        # 显示可视化结果
        cv2.imshow('Matches', combined_image)
        # print(combined_image*255)
        combined_image_save=combined_image*255
        # print("999999999999"+str(combined_image.shape))
        cv2.waitKey(10)

        # 保存图像
        # Image.fromarray(combined_image).save('combined_image.png')
        cv2.imwrite('combined_image.jpg', combined_image_save)
        

        #Compute descriptor matches using superglue（新）
        matches0 = matches['matches0']
        matches0_np = matches0[0].cpu().detach().numpy()
        matches1 = matches['matches1']
        matches1_np = matches1[0].cpu().detach().numpy()
        # print()
        # 获取匹配点的索引
        target_idx = np.arange(480)
        # print(target_idx) 
        ref_idx = matches0_np

        # print(target_idx)
        # print(ref_idx)
        pt0 = np.empty((0, 2), dtype=np.float32)
        pt1 = np.empty((0, 3), dtype=np.float32)
        pt2 = np.empty((0, 2), dtype=np.float32)
        pt3 = np.empty((0, 3), dtype=np.float32)
        # pt3 = np.empty((0, 3), dtype=np.int32)
        target_xyz=target_xyz.cpu().detach().numpy()
        target_uv=target_uv.cpu().detach().numpy()
        ref_uv=ref_uv.cpu().detach().numpy()
        ref_xyz=ref_xyz.cpu().detach().numpy()
        # pt1 = target_xyz[target_idx].cpu().numpy()
        # pt2 = ref_uv[ref_idx].cpu().numpy()
        
        for i in range(matches0.shape[1]):
            if matches0[0, i] != -1 and score0[0,i]>0.5:  # 如果有匹配点
                pt0 = np.append(pt0, [[(keypoints0[0, i, 0]), (keypoints0[0, i, 1])]], axis=0)  # Keypoints coordinates of the first image
                pt1 = np.append(pt1, [target_xyz[i]], axis=0)
                pt2 = np.append(pt2, [[(keypoints1[0, matches0[0, i], 0]), (keypoints1[0, matches0[0, i], 1])]], axis=0)  # # 第二张图像中的关键点坐标
                pt3 = np.append(pt3, [ref_xyz[matches0[0, i]]], axis=0)
                
        combined_image = cv2.hconcat([target_image_display_bgr, source_image_display_bgr])

        # 绘制匹配的关键点并连线
        for i in range(len(pt0)):
            pt0_x, pt0_y = int(pt0[i][0]), int(pt0[i][1])
            pt2_x, pt2_y = int(pt2[i][0] + image_size0[1]), int(pt2[i][1])
            cv2.circle(combined_image, (pt0_x, pt0_y), 5, (0, 0, 255), -1)  # 绘制目标图像的关键点
            cv2.circle(combined_image, (pt2_x, pt2_y), 5, (255, 0, 0), -1)  # 绘制参考图像的关键点
            cv2.line(combined_image, (pt0_x, pt0_y), (pt2_x, pt2_y), (0, 255, 0), 1)  # 绘制连接线
            

        # 显示拼接后的图像
        cv2.imshow('Combined Image', combined_image)
        cv2.waitKey(10)

        # Run PnP and get pose
        K = target_intrinsic.float().cpu().detach().numpy().squeeze()
        # # 2D->2D
        # # em, mask = cv2.findEssentialMat(pt0, pt2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        # # retval, R, t, mask = cv2.recoverPose(em, pt0, pt2, K, mask=mask) 
        # # 2D->3D
        _, rvecs, t, inliers = cv2.solvePnPRansac(pt1, pt2, K, None, reprojectionError=0.85, iterationsCount=5000)
        # print(inliers)
        R = cv2.Rodrigues(rvecs)[0]
        # R_before=R
        # t_before=t
        pose=SE3()
        pose.R = cv2.Rodrigues(rvecs)[0]
        pose.t = t
        pose.pose = pose.inv_pose

        # self.ref_keyframe=self.keyframe_list[-1]
        # inv_keyframe_global_pose=self.ref_keyframe.data["pose"].inv_pose
        # cur_pose = self.cur_data['pose'].pose
        # relative_pose = inv_keyframe_global_pose @ cur_pose

        # R = relative_pose.R
        # t = relative_pose.t

        print("R:",R)
        print("t:",t)
        print("rR:",rR)
        print("rt:",rt)
        # rvecs, _ = cv2.Rodrigues(rR)
        # optimal_pose
        pt0 = np.empty((0, 2), dtype=np.float32)
        pt1 = np.empty((0, 3), dtype=np.float32)
        pt2 = np.empty((0, 2), dtype=np.float32)
        pt3 = np.empty((0, 3), dtype=np.float32)
        match_count=0
        print(matches0)
        for i in range(matches0.shape[1]):
            if matches0[0, i] != -1 and score0[0,i]>0.7:  # 如果有匹配点
                match_count += 1
                pt0 = np.append(pt0, [[(keypoints0[0, i, 0]), (keypoints0[0, i, 1])]], axis=0)  # Keypoints coordinates of the first image
                pt1 = np.append(pt1, [target_xyz[i]], axis=0)
                pt2 = np.append(pt2, [[(keypoints1[0, matches0[0, i], 0]), (keypoints1[0, matches0[0, i], 1])]], axis=0)  # # 第二张图像中的关键点坐标
                pt3 = np.append(pt3, [ref_xyz[matches0[0, i]]], axis=0)

        print("匹配点数：",match_count)
        # R,t=minimize_reprojection_error(rvecs, t, pt1,pt2, intrinsic_matrix,source_image_display_bgr)

        # P = np.concatenate([R,t], 1)
        R = pose.R
        t = pose.t 
        return R,t

        # self.ref_keyframe=self.keyframe_list[-1]
        # inv_keyframe_global_pose=self.ref_keyframe.data["pose"].inv_pose
        # cur_pose = self.cur_data['pose'].pose
        # relative_pose = inv_keyframe_global_pose @ cur_pose

        # R = relative_pose.R
        # t = relative_pose.t
        # rvecs, _ = cv2.Rodrigues(R)
        # # # 最小化重投影误差来进行位姿的优化
        # R,t=minimize_reprojection_error(rvecs, t, pt1,pt2, self.dataset.cam_intrinsics.mat,this.cur_data[img])
    
    def tracking(self):
        """Tracking using both Essential matrix and PnP
        Essential matrix for rotation and translation direction;
            *** triangluate depth v.s. CNN-depth for translation scale ***
        PnP if Essential matrix fails
        """
        # 初始化网络
        depth_net = DispResnet(version='18_pt')
        depth_net.load_state_dict(torch.load(self.cfg.depth_model, map_location='cpu'))
        depth_net = depth_net.cuda()  # move to GPU

        keypoint_net = KeypointResnet()
        keypoint_net.load_state_dict(torch.load(self.cfg.keypoint_model, map_location='cpu'))
        keypoint_net.cuda()

        config = {
            "descriptor_dim": 256,
            "weights": "outdoor",
            "keypoint_encoder": [32, 64, 128, 256],
            "GNN_layers": ["self", "cross"] * 9,
            "sinkhorn_iterations": 100,
            "match_threshold": 0.2,
            "cuda": True
        }
        superglue_net = SuperGlue(config)
        superglue_net.load_state_dict(torch.load(self.cfg.featmatch_model, map_location='cpu'))
        superglue_net.cuda()
        depth_net.eval()
        keypoint_net.eval()
        superglue_net.eval()
        # 计算当前帧当前帧和参考关键帧之间的相对位姿
        # for i in self.cur_data:
        #     print("cur",i)
        # for i in self.ref_data:
        #     print("ref",i)
        # self.ref_keyframe=self.keyframe_list[-1]
        # self.ref_keyframe.data['img'] = self.dataset.get_image(self.ref_keyframe.data['timestamp'])
        # self.cur_data['img'] = self.dataset.get_image(self.cur_data['timestamp'])
        
        
        # target_image = Image.fromarray(self.ref_data['img'])
        source_image = Image.fromarray(self.cur_data['img'])
        # target_image = np.array(target_image)
        source_image = np.array(source_image)

        # target_image = torch.tensor(target_image).to('cuda')
        source_image = torch.tensor(source_image).to('cuda')
        # target_image = target_image.unsqueeze(0)
        source_image = source_image.unsqueeze(0)
        # target_image = target_image.permute(0, 3, 1, 2)
        source_image = source_image.permute(0, 3, 1, 2)
        # target_image = target_image.float()
        source_image = source_image.float()


        target_intrinsic=torch.tensor(self.dataset.cam_intrinsics.mat).to('cuda')
        target_intrinsic = target_intrinsic.unsqueeze(0)
        # target_image = target_image/255
        source_image = source_image/255
        # First frame
        if self.tracking_stage == 0:
            # initial pose
            if self.cfg.directory.gt_pose_dir is not None:
                self.cur_data['pose'] = SE3(self.dataset.gt_poses[self.cur_data['id']])
            else:
                self.cur_data['pose'] = SE3()
            self.ref_data['motion'] = SE3()
            # print("frame_number:",self.keyframe_list[-1].data['id'])
            pt4,pt5,pt5_feat=self.super_extract(keypoint_net, depth_net, superglue_net, source_image, target_intrinsic)
            self.cur_data['superpoint'] = pt5
            self.cur_data['superfeat'] = pt5_feat
            self.keyframe_list.append(KeyFrame(frame_number=self.tracking_stage,data=copy.deepcopy(self.cur_data)))
            self.disp2d, self.disp3d = None, None
            self.disp3d = Display3D()
            #把各种值赋值过去就好了
            W = self.cfg.image.width
            H = self.cfg.image.height
            K = self.dataset.cam_intrinsics.mat
            self.disp2d = Display2D(W, H)
            self.slam = SLAM(W, H, K)
            frame = self.keyframe_list[-1]
            self.slam.process_frame(frame, None)
            self.disp3d.paint(self.slam.mapp)
            img = self.keyframe_list[-1].data['img']
            self.disp2d.paint(img)
            if self.disp3d is not None:
                self.disp3d.paint(self.slam.mapp)
            # print(type(disp3d))
            if self.disp2d is not None:
                img = self.keyframe_list[-1].data['img']
                self.disp2d.paint(img)
            return
        
        #add key-frame
        #把需要的东西都存进来，其实也就需要存一个深度图和superpoint的结果好像,
        
        
        # Second to last frames
        elif self.tracking_stage >= 1:
            pt4,pt5,pt5_feat=self.super_extract(keypoint_net, depth_net, superglue_net, source_image, target_intrinsic)
            self.cur_data['superpoint'] = pt5
            self.cur_data['superfeat'] = pt5_feat
            # ''' keypoint selection '''
            self.ref_keyframe=self.keyframe_list[-1]
            # # print("start_id:",self.ref_keyframe.data)
            # # print("start_fn:",self.ref_keyframe.frame_number)
            # # print("start_fn:",self.tracking_stage)
            if self.tracking_method in ['hybrid', 'PnP']:
                # Depth consistency (CNN depths + CNN pose)
                if self.cfg.kp_selection.depth_consistency.enable:
                    self.depth_consistency_computer.compute(self.cur_data, self.ref_data)

                # kp_selection
                self.timers.start('kp_sel', 'tracking')
                kp_sel_outputs = self.kp_sampler.kp_selection1(self.cur_data, self.ref_data,pt4)
                if kp_sel_outputs['good_kp_found']:
                    self.kp_sampler.update_kp_data(self.cur_data, self.ref_data, kp_sel_outputs)
                self.timers.end('kp_sel')

            ''' Pose estimation '''
            # Initialize hybrid pose
            hybrid_pose = SE3()
            E_pose = SE3()

            if not(kp_sel_outputs['good_kp_found']):
                print("No enough good keypoints, constant motion will be used!")
                pose = self.ref_data['motion']
                self.update_global_pose(pose, 1)
                return 


            ''' E-tracker '''
            if self.tracking_method in ['hybrid']:
                # Essential matrix pose
                self.timers.start('E-tracker', 'tracking')
                e_tracker_outputs = self.e_tracker.compute_pose_2d2d(
                                self.ref_data[self.cfg.e_tracker.kp_src],
                                self.cur_data[self.cfg.e_tracker.kp_src],
                                not(self.cfg.e_tracker.iterative_kp.enable)) # pose: from cur->ref
                E_pose = e_tracker_outputs['pose']
                self.timers.end('E-tracker')

                # Rotation
                hybrid_pose.R = E_pose.R

                # save inliers
                self.ref_data['inliers'] = e_tracker_outputs['inliers']

                # print("np.linalg.norm(E_pose.t):",np.linalg.norm(E_pose.t))
                # print("E_pose.t:",E_pose.t)
                # print("E_pose.R:",E_pose.R)
                # scale recovery
                if np.linalg.norm(E_pose.t) != 0:
                    self.timers.start('scale_recovery', 'tracking')
                    scale_out = self.e_tracker.scale_recovery(self.cur_data, self.ref_data, E_pose, False)
                    scale = scale_out['scale']
                    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!scale:",scale)
                    if self.cfg.scale_recovery.kp_src == 'kp_depth':
                        self.cur_data['kp_depth'] = scale_out['cur_kp_depth']
                        self.ref_data['kp_depth'] = scale_out['ref_kp_depth']
                        self.cur_data['rigid_flow_mask'] = scale_out['rigid_flow_mask']
                    if scale != -1:
                        hybrid_pose.t = E_pose.t * scale
                    self.timers.end('scale_recovery')

                # Iterative keypoint refinement
                if np.linalg.norm(E_pose.t) != 0 and self.cfg.e_tracker.iterative_kp.enable:
                    self.timers.start('E-tracker iter.', 'tracking')
                    # Compute refined keypoint
                    self.e_tracker.compute_rigid_flow_kp(self.cur_data,
                                                         self.ref_data,
                                                         hybrid_pose)

                    e_tracker_outputs = self.e_tracker.compute_pose_2d2d(
                                self.ref_data[self.cfg.e_tracker.iterative_kp.kp_src],
                                self.cur_data[self.cfg.e_tracker.iterative_kp.kp_src],
                                True) # pose: from cur->ref
                    E_pose = e_tracker_outputs['pose']

                    # Rotation
                    hybrid_pose.R = E_pose.R

                    # save inliers
                    self.ref_data['inliers'] = e_tracker_outputs['inliers']

                    # scale recovery
                    if np.linalg.norm(E_pose.t) != 0 and self.cfg.scale_recovery.iterative_kp.enable:
                        scale_out = self.e_tracker.scale_recovery(self.cur_data, self.ref_data, E_pose, True)
                        scale = scale_out['scale']
                        # print("222222222222222222222222222222222222scale:",scale)
                        if scale != -1:
                            hybrid_pose.t = E_pose.t * scale
                    else:
                        hybrid_pose.t = E_pose.t * scale
                    self.timers.end('E-tracker iter.')

            ''' PnP-tracker '''
            if self.tracking_method in ['PnP', 'hybrid']:
                # PnP if Essential matrix fail
                if np.linalg.norm(E_pose.t) == 0 or scale ==-1:
                    self.timers.start('pnp', 'tracking')
                    # kp_selection
                    # # 初始化网络
                    # depth_net = DispResnet(version='18_pt')
                    # depth_net.load_state_dict(torch.load(self.cfg.depth_model, map_location='cpu'))
                    # depth_net = depth_net.cuda()  # move to GPU

                    # keypoint_net = KeypointResnet()
                    # keypoint_net.load_state_dict(torch.load(self.cfg.keypoint_model, map_location='cpu'))
                    # keypoint_net.cuda()

                    # config = {
                    #     "descriptor_dim": 256,
                    #     "weights": "outdoor",
                    #     "keypoint_encoder": [32, 64, 128, 256],
                    #     "GNN_layers": ["self", "cross"] * 9,
                    #     "sinkhorn_iterations": 100,
                    #     "match_threshold": 0.2,
                    #     "cuda": True
                    # }
                    # superglue_net = SuperGlue(config)
                    # superglue_net.load_state_dict(torch.load(self.cfg.featmatch_model, map_location='cpu'))
                    # superglue_net.cuda()
                    # depth_net.eval()
                    # keypoint_net.eval()
                    # superglue_net.eval()
                    # # 计算当前帧当前帧和参考关键帧之间的相对位姿
                    # # for i in self.cur_data:
                    # #     print("cur",i)
                    # # for i in self.ref_data:
                    # #     print("ref",i)
                    # # self.ref_keyframe=self.keyframe_list[-1]
                    # # self.ref_keyframe.data['img'] = self.dataset.get_image(self.ref_keyframe.data['timestamp'])
                    # # self.cur_data['img'] = self.dataset.get_image(self.cur_data['timestamp'])
                    # target_image = Image.fromarray(self.ref_keyframe.data['img'])
                    # source_image = Image.fromarray(self.cur_data['img'])
                    # target_image = np.array(target_image)
                    # source_image = np.array(source_image)

                    # target_image = torch.tensor(target_image).to('cuda')
                    # source_image = torch.tensor(source_image).to('cuda')
                    # target_image = target_image.unsqueeze(0)
                    # source_image = source_image.unsqueeze(0)
                    # target_image = target_image.permute(0, 3, 1, 2)
                    # source_image = source_image.permute(0, 3, 1, 2)
                    # target_image = target_image.float()
                    # source_image = source_image.float()


                    # target_intrinsic=torch.tensor(self.dataset.cam_intrinsics.mat).to('cuda')
                    # target_intrinsic = target_intrinsic.unsqueeze(0)
                    # target_image = target_image/255
                    # source_image = source_image/255

                    # pt0,pt1,pt2,pt3,pt4=self.kp_pnp_track(keypoint_net, depth_net, superglue_net,target_image, source_image, target_intrinsic)
                    ''' keypoint selection '''
                    # self.ref_keyframe=self.keyframe_list[-1]
                    # print("start_id:",self.ref_keyframe.data)
                    # print("start_fn:",self.ref_keyframe.frame_number)
                    # print("start_fn:",self.tracking_stage)
                    # kp_sel_outputs = self.kp_sampler.kp_selection1(self.cur_data, self.ref_data,pt4)
                    # if kp_sel_outputs['good_kp_found']:
                    #     self.kp_sampler.update_kp_data(self.cur_data, self.ref_data, kp_sel_outputs)

                    # if not(kp_sel_outputs['good_kp_found']):
                    #     print("No enough good keypoints, constant motion will be used!")
                    #     pose = self.ref_data['motion']
                    #     self.update_global_pose(pose, 1)
                    #     return 
                    
                    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!",self.tracking_stage)
                    pnp_outputs = self.pnp_tracker.compute_pose_3d2d(
                                    self.ref_data[self.cfg.pnp_tracker.kp_src],
                                    self.cur_data[self.cfg.pnp_tracker.kp_src],
                                    self.ref_data['raw_depth'],
                                    not(self.cfg.pnp_tracker.iterative_kp.enable)
                                    ) # pose: from cur->ref
                    # pnp_outputs = self.pnp_tracker.compute_pose_3d2d(
                    #                 pt0,pt2,
                    #                 self.ref_data['depth'],
                    #                 not(self.cfg.pnp_tracker.iterative_kp.enable)
                    #                 ) # pose: from cur->ref
                    # print("self.ref_data[self.cfg.pnp_tracker.kp_src]:",self.ref_data[self.cfg.pnp_tracker.kp_src].shape)
                    # print(pt0.shape)
                    ###现在的目标就是修改上面这个函数###
                    ##如果修改好以后没用的话，那就说明不是这个方法的问题而是别的问题
                    # print("self.ref_data['depth']:",self.ref_data['depth'].shape)
                    # print("self.ref_data[self.cfg.pnp_tracker.kp_src]:",self.ref_data[self.cfg.pnp_tracker.kp_src])
                    # print("self.ref_data[self.cfg.pnp_tracker.kp_src]:",self.ref_data[self.cfg.pnp_tracker.kp_src].shape)
                    # Iterative keypoint refinement
                    # if self.cfg.pnp_tracker.iterative_kp.enable:
                    #     self.pnp_tracker.compute_rigid_flow_kp(self.cur_data, self.ref_data, pnp_outputs['pose'])
                    #     pnp_outputs = self.pnp_tracker.compute_pose_3d2d(
                    #                 self.ref_data[self.cfg.pnp_tracker.iterative_kp.kp_src],
                    #                 self.cur_data[self.cfg.pnp_tracker.iterative_kp.kp_src],
                    #                 self.ref_data['depth'],
                    #                 True
                    #                 ) # pose: from cur->ref
                    # print("np.linalg.norm(pnp_pose.t):",np.linalg.norm(pnp_outputs["pose"].t))
                    # print("pnp_pose.t:",pnp_outputs["pose"].t)
                    # print("pnp_pose.t:",pnp_outputs["pose"].R)
                    self.timers.end('pnp')

                    # use PnP pose instead of E-pose
                    # hybrid_pose = SE3()
                    # self.tracking_mode = "kp-PnP"
                    hybrid_pose = pnp_outputs['pose']
                    self.tracking_mode = "PnP"
                    # hybrid_pose.t = pnp_outputs['pose'].t
                    # hybrid_pose.R = kpnp_outputs['pose'].R
                    # hybrid_pose = kpnp_outputs['pose']
                    # self.tracking_mode = "kp-PnP"

                    

            ''' Deep-tracker '''
            if self.tracking_method in ['deep_pose']:
                hybrid_pose = SE3(self.ref_data['deep_pose'])
                self.tracking_mode = "DeepPose"



            ''' Summarize data '''
            # update global poses
            self.ref_data['pose'] = copy.deepcopy(hybrid_pose)
            self.ref_data['motion'] = copy.deepcopy(hybrid_pose)
            pose = self.ref_data['pose']
            self.update_global_pose(pose, 1)
            # print(self.cur_data['pose'].pose)#存的是全局位姿
            # print(self.ref_data['motion'].pose)#这个里面存的是相对位姿
            
            ''' add key_frame '''
            #后续需要修改判断是否需要加入关键帧的条件
            if self.tracking_stage%3==2:
                self.keyframe_list.append(KeyFrame(frame_number=self.tracking_stage,data=copy.deepcopy(self.cur_data)))
                print(len(self.keyframe_list))
                # print("self.global_poses[self.cur_data['id']]:",self.global_poses[self.cur_data['id']].pose)
                # print("self.cur_data['poss']:",self.cur_data['pose'].pose)
                # print("data['id']:",self.keyframe_list[-1].data['id'])
                
                # print("frame_number:",self.keyframe_list[-1].frame_number)
                # print("frame_data:",self.keyframe_list[-1].data)
                # for i in self.keyframe_list[-1].data:
                #     print("key_name:",i)
                
                # #初始化网络
                # depth_net = DispResnet(version='18_pt')
                # depth_net.load_state_dict(torch.load(self.cfg.depth_model, map_location='cpu'))
                # depth_net = depth_net.cuda()  # move to GPU

                # keypoint_net = KeypointResnet()
                # keypoint_net.load_state_dict(torch.load(self.cfg.keypoint_model, map_location='cpu'))
                # keypoint_net.cuda()

                # config = {
                #     "descriptor_dim": 256,
                #     "weights": "outdoor",
                #     "keypoint_encoder": [32, 64, 128, 256],
                #     "GNN_layers": ["self", "cross"] * 9,
                #     "sinkhorn_iterations": 100,
                #     "match_threshold": 0.2,
                #     "cuda": True
                # }
                # superglue_net = SuperGlue(config)
                # superglue_net.load_state_dict(torch.load(self.cfg.featmatch_model, map_location='cpu'))
                # superglue_net.cuda()
                # depth_net.eval()
                # keypoint_net.eval()
                # superglue_net.eval()
                # # 计算当前帧当前帧和参考关键帧之间的相对位姿
                # # for i in self.cur_data:
                # #     print("cur",i)
                # # for i in self.ref_data:
                # #     print("ref",i)
                # self.ref_keyframe=self.keyframe_list[-1]
                # # self.ref_keyframe.data['img'] = self.dataset.get_image(self.ref_keyframe.data['timestamp'])
                # # self.cur_data['img'] = self.dataset.get_image(self.cur_data['timestamp'])
                # target_image = Image.fromarray(self.ref_keyframe.data['img'])
                # source_image = Image.fromarray(self.cur_data['img'])
                # target_image = np.array(target_image)
                # source_image = np.array(source_image)

                # target_image = torch.tensor(target_image).to('cuda')
                # source_image = torch.tensor(source_image).to('cuda')
                # target_image = target_image.unsqueeze(0)
                # source_image = source_image.unsqueeze(0)
                # target_image = target_image.permute(0, 3, 1, 2)
                # source_image = source_image.permute(0, 3, 1, 2)
                # target_image = target_image.float()
                # source_image = source_image.float()


                # target_intrinsic=torch.tensor(self.dataset.cam_intrinsics.mat).to('cuda')
                # target_intrinsic = target_intrinsic.unsqueeze(0)

                # target_image = target_image/255
                # source_image = source_image/255
            
                # keyframe_global_pose=self.ref_keyframe.data["pose"].pose
                # inv_keyframe_global_pose=self.ref_keyframe.data["pose"].inv_pose
                # cur_pose = self.cur_data['pose'].pose
                # inv_cur_pose = self.cur_data['pose'].inv_pose
                # relative_pose = SE3(inv_keyframe_global_pose @ cur_pose)
                # inv_keyframe_global_pose=SE3(inv_cur_pose @ keyframe_global_pose)
                # inv_relative_pose=SE3(relative_pose.inv_pose)
                # # R = inv_relative_pose.R
                # # t = inv_relative_pose.t
                # R = relative_pose.R
                # t = relative_pose.t
                # # R,t=self.match_features(keypoint_net, depth_net, superglue_net,target_image, source_image, target_intrinsic,R,t)
                # # 需要把R还原成global_pose
                # self.update_keyframe_global_pose(R,t, 1)
                # self.keyframe_list.append(KeyFrame(frame_number=self.tracking_stage,data=copy.deepcopy(self.cur_data)))


                # print(self.ref_data["motion"].pose)
                # print(self.ref_data["pose"].pose)
                '''对关键帧进行一些操作，比如投影到地图点之类的，设置一些操作:地图点之间的匹配使用superglue'''
                '''对每一个关键帧都进行观测的添加，有了观测才可以进行不管是全局优化还是局部优化'''
                frame = self.keyframe_list[-1]
                self.slam.process_frame(frame, None)
                self.disp3d.paint(self.slam.mapp)
                img = self.keyframe_list[-1].data['img']
                self.disp2d.paint(img)
                if self.disp3d is not None:
                    self.disp3d.paint(self.slam.mapp)
                # print(type(disp3d))
                if self.disp2d is not None:
                    img = self.keyframe_list[-1].data['img']
                    self.disp2d.paint(img)
                # print(type(self.disp2d))

                '''local optimize'''
                if self.tracking_stage > 7:
                    # err = optimize(self.keyframe_list, local_window=7, fix_points=True, verbose=False, rounds=10)


                    '''loop closure'''
                    '''
                    todo:
                    1:需要根据t,来判断是否要进行回环检测
                    2:这里需要通过回环来返回回环的约束信息：和那一帧判定为回环：需要知道的是回环的判断阈值
                    '''
                    id2,id1=100,10
                    loop_flag=True

                    '''global optimize'''
                    '''
                    修改参数：根据回环的信息算出local_window的长度
                    
                    '''
                    if loop_flag:
                        window_number=id2-id1
                        # err = optimize(self.keyframe_list, local_window=window_number, fix_points=False, verbose=False, rounds=50)



                        '''all frames finetune'''
                        '''
                        线性插值法
                        '''


                    '''
                    最后不确定要不要在最后一帧的时候再做一次全局优化——可以加入看看量化数据
                    
                    '''
                    # err = optimize(self.keyframe_list, local_window=None, fix_points=False, verbose=False, rounds=50)

            

    def update_data(self, ref_data, cur_data):
        """Update data
        
        Args:
            ref_data (dict): reference data
            cur_data (dict): current data
        
        Returns:
            ref_data (dict): updated reference data
            cur_data (dict): updated current data
        """
        for key in cur_data:
            if key == "id":
                ref_data['id'] = cur_data['id']
            else:
                if ref_data.get(key, -1) is -1:
                    ref_data[key] = {}
                ref_data[key] = cur_data[key]
        
        # Delete unused flow to avoid data leakage
        ref_data['flow'] = None
        cur_data['flow'] = None
        ref_data['flow_diff'] = None
        return ref_data, cur_data

    def load_raw_data(self):
        """load image data and (optional) GT/precomputed depth data
        """
        # Reading image
        self.cur_data['img'] = self.dataset.get_image(self.cur_data['timestamp'])

        # Reading/Predicting depth
        if self.dataset.data_dir['depth_src'] is not None:
            self.cur_data['raw_depth'] = self.dataset.get_depth(self.cur_data['timestamp'])
    
    def deep_model_inference(self):
        """deep model prediction
        """
        if self.tracking_method in ['hybrid', 'PnP']:
            # Single-view Depth prediction
            if self.dataset.data_dir['depth_src'] is None:
                self.timers.start('depth_cnn', 'deep inference')
                if self.tracking_stage > 0 and \
                    self.cfg.online_finetune.enable and self.cfg.online_finetune.depth.enable:
                        img_list = [self.cur_data['img'], self.ref_data['img']]
                else:
                    img_list = [self.cur_data['img']]

                self.cur_data['raw_depth'] = \
                    self.deep_models.forward_depth(imgs=img_list)
                self.cur_data['raw_depth'] = cv2.resize(self.cur_data['raw_depth'],
                                                    (self.cfg.image.width, self.cfg.image.height),
                                                    interpolation=cv2.INTER_NEAREST
                                                    )
                # print("self.cur_data['raw_depth']",self.cur_data['raw_depth'].shape)
                self.timers.end('depth_cnn')
            self.cur_data['depth'] = preprocess_depth(self.cur_data['raw_depth'], self.cfg.crop.depth_crop, [self.cfg.depth.min_depth, self.cfg.depth.max_depth])

            # Two-view flow
            if self.tracking_stage >= 1:
                self.timers.start('flow_cnn', 'deep inference')
                flows = self.deep_models.forward_flow(
                                        self.cur_data,
                                        self.ref_data,
                                        forward_backward=self.cfg.deep_flow.forward_backward)
                # print("flows",flows)
                # Store flow
                self.ref_data['flow'] = flows[(self.ref_data['id'], self.cur_data['id'])].copy()
                if self.cfg.deep_flow.forward_backward:
                    self.cur_data['flow'] = flows[(self.cur_data['id'], self.ref_data['id'])].copy()
                    self.ref_data['flow_diff'] = flows[(self.ref_data['id'], self.cur_data['id'], "diff")].copy()
                # print("self.cur_data['flow']",self.cur_data['flow'].shape)
                # print("self.ref_data['flow_diff']",self.ref_data['flow_diff'].shape)
                self.timers.end('flow_cnn')
            
        # Relative camera pose
        if self.tracking_stage >= 1 and self.cfg.deep_pose.enable:
            self.timers.start('pose_cnn', 'deep inference')
            # Deep pose prediction
            pose = self.deep_models.forward_pose(
                        [self.ref_data['img'], self.cur_data['img']] 
                        )
            self.ref_data['deep_pose'] = pose # from cur->ref
            self.timers.end('pose_cnn')

    def main(self):
        """Main program
        """
        print("==> Start DF-VO")
        print("==> Running sequence: {}".format(self.cfg.seq))

        if self.cfg.no_confirm:
            start_frame = 0
        else:
            start_frame = int(input("Start with frame: "))

        for img_id in tqdm(range(start_frame, len(self.dataset), self.cfg.frame_step)):
            self.timers.start('DF-VO')
            self.tracking_mode = "Ess. Mat."#这个参数应该是为了输出显示的，对代码本身没有影响

            """ Data reading """
            # Initialize ids and timestamps
            self.cur_data['id'] = img_id
            self.cur_data['timestamp'] = self.dataset.get_timestamp(img_id)

            # Read image data and (optional) precomputed depth data
            self.timers.start('data_loading')
            self.load_raw_data()
            self.timers.end('data_loading')

            # Deep model inferences
            self.timers.start('deep_inference')
            self.deep_model_inference()#这里需要加入superpoint和superglue的代码
            self.timers.end('deep_inference')

            """ Visual odometry """
            self.timers.start('tracking')
            self.tracking()#这里需要用superpoint和superglue的结果来进行修改，包括关键帧的代码
            self.timers.end('tracking')

            """ Online Finetuning """
            if self.tracking_stage >= 1 and self.cfg.online_finetune.enable:
                self.deep_models.finetune(self.ref_data['img'], self.cur_data['img'],
                                      self.ref_data['pose'].pose,
                                      self.dataset.cam_intrinsics.mat,
                                      self.dataset.cam_intrinsics.inv_mat)
            """ Visualization """
            if self.cfg.visualization.enable:
                self.timers.start('visualization')
                self.drawer.main(self)#这里可以看到各个得到的参数在可视化中是怎么用的
                self.timers.end('visualization')

            """ Update reference and current data """
            self.ref_data, self.cur_data = self.update_data(
                                    self.ref_data,
                                    self.cur_data,
            )

            self.tracking_stage += 1

            self.timers.end('DF-VO')

        print("=> Finish!")



        """ Display & Save result """
        print("The result is saved in [{}].".format(self.cfg.directory.result_dir))
        # Save trajectory map
        print("Save VO map.")
        map_png = "{}/map.png".format(self.cfg.directory.result_dir)
        cv2.imwrite(map_png, self.drawer.data['traj'])

        # Save trajectory txt
        traj_txt = "{}/{}.txt".format(self.cfg.directory.result_dir, self.cfg.seq)
        self.dataset.save_result_traj(traj_txt, self.global_poses)

        # save finetuned model
        if self.cfg.online_finetune.enable and self.cfg.online_finetune.save_model:
            self.deep_models.save_model()

        # Output experiement information
        self.timers.time_analysis()
