


import os
import cv2
import numpy as np


from scipy.spatial import cKDTree
import os
import cv2
import numpy as np
from scipy.spatial import cKDTree
from libs.backend.constants import RANSAC_RESIDUAL_THRES, RANSAC_MAX_TRIALS
np.set_printoptions(suppress=True)

from skimage.measure import ransac
from libs.backend.helpers import add_ones, poseRt, fundamentalToRt, normalize, EssentialMatrixTransform, myjet

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

#这里我感觉直接用之前的会比较好，所以在前面提取特征点的时候还需要把描述子给return出来
#看文档，感觉靠筛选出来的点，会减小感受野，所以需要把点全都提取出来
#关键点也可以尝试用光流加keypoints来得到，但是关键帧之间的距离隔得比较远，光流网络的性能可能不会好
#最后的策略是，使用superpoint+superglue以后使用RANSAC选择出内点去进行BA
#是不是还需要对深度值过大的点进行一个mask呢？感觉可以在选择点的时候进行一个判断项来进行筛选（还不知道需不需要剔除呢！我觉得是需要的但在01数据集就比较麻烦，没准后续为了最后的量化指标不能进行离群值的剔除呢！）
#其实没区别因为pnp的时候其实是进行了内点选取的，应该是01数据集的时候，剔除了离群点，导致点较少，因此就这样了
def extractFeatures(img):
  #改成superpoint提取特征点加选取

  orb = cv2.ORB_create()
  # detection
  pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)

  # extraction
  kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
  kps, des = orb.compute(img, kps)

  # return pts and des
  return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

def match_frames_super(f1, f2):
  target_feat = f2.des
  ref_feat = f1.des
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

def compute_reprojection_error(pose, points1, points2):
    """
    计算重投影误差
    pose: 4x4 齐次变换矩阵
    points1, points2: 对应点对
    """
    num_points = points1.shape[0]
    
    # 将 points1 转换为齐次坐标，假设深度为1
    points1_hom = np.hstack((points1, np.ones((num_points, 1))))
    points1_hom = np.hstack((points1_hom, np.ones((num_points, 1))))  # 形成4维齐次坐标

    # 使用 pose 矩阵将 points1 转换到第二个图像的坐标系
    transformed_points1_hom = points1_hom @ pose.T

    # 将转换后的点恢复为非齐次坐标
    transformed_points1 = transformed_points1_hom[:, :3] / transformed_points1_hom[:, 3][:, np.newaxis]
    
    # 计算重投影误差（这里只计算前两维的误差，因为 points2 是二维的）
    errors = np.linalg.norm(transformed_points1[:, :2] - points2, axis=1)
    return errors

def find_inliers(initial_pose, points1, points2, residual_threshold):
    errors = compute_reprojection_error(initial_pose, points1, points2)
    inliers = errors < residual_threshold
    return inliers, errors



def match_frames(f1, f2):
  #改为superglue进行配对+RANSAC进行内点的选择，poss使用前端得到的位姿，就不需要再返回poss
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  matches = bf.knnMatch(f1.des, f2.des, k=2)

  # Lowe's ratio test
  ret = []
  idx1, idx2 = [], []
  idx1s, idx2s = set(), set()

  for m,n in matches:
    if m.distance < 0.75*n.distance:
      p1 = f1.kps[m.queryIdx]
      p2 = f2.kps[m.trainIdx]

      # be within orb distance 32
      if m.distance < 32:
        # keep around indices
        # TODO: refactor this to not be O(N^2)
        if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
          idx1.append(m.queryIdx)
          idx2.append(m.trainIdx)
          idx1s.add(m.queryIdx)
          idx2s.add(m.trainIdx)
          ret.append((p1, p2))

  # no duplicates
  assert(len(set(idx1)) == len(idx1))
  assert(len(set(idx2)) == len(idx2))

  assert len(ret) >= 8
  ret = np.array(ret)
  idx1 = np.array(idx1)
  idx2 = np.array(idx2)

  # Rt = f1.pose @ f2.pose  
  # Rt = np.linalg.inv(Rt)
  inv_pose1 = np.linalg.inv(f1.pose)
  inv_pose2 = np.linalg.inv(f2.pose)
  Rt = inv_pose1 @ inv_pose2 
  # Rt = np.linalg.inv(Rt)
  print("Rt:",Rt)#这个地方解出Rt其实就是为了选出内点
  
  # global_pose_inv
  # return idx1[inliers], idx2[inliers], np.linalg.inv(Rt)
  # initial_pose = np.array(Rt)
  # points1 = ret[:, 0]
  # points2 = ret[:, 1]
  # print("points1:",points1.shape)
  # print("points1:",points1)
  # inliers, errors = find_inliers(initial_pose, points1, points2, RANSAC_RESIDUAL_THRES)
  # print("Matches:  %d -> %d -> %d -> %d" % (len(f1.des), len(matches), len(inliers), sum(inliers)))
  # fit matrix
  # print("ret:",ret.shape)
  # model, inliers = ransac((ret[:, 0], ret[:, 1]),
  #                         EssentialMatrixTransform,
  #                         min_samples=10,
  #                         residual_threshold=RANSAC_RESIDUAL_THRES,
  #                         max_trials=RANSAC_MAX_TRIALS)
  # print("Matches:  %d -> %d -> %d -> %d" % (len(f1.des), len(matches), len(inliers), sum(inliers)))
  # print("ERT:",fundamentalToRt(model.params))
  return idx1, idx2, Rt
  # return idx1[inliers], idx2[inliers], Rt
  # return idx1[inliers], idx2[inliers], fundamentalToRt(model.params)

class Frame(object):
  def __init__(self, mapp, keyframe, K, pose=np.eye(4), tid=None, verts=None):
    self.K = np.array(K)
    self.pose = np.array(pose)
    img = keyframe.data['img']
    if img is not None:
      self.h, self.w = img.shape[0:2]
      if verts is None:
        self.kpus, self.des = extractFeatures(img)
        # self.kpus, self.des = keyframe.data['superpoint'],keyframe.data['superfeat']
        self.pose = keyframe.data['pose'].pose
        self.raw_depth = keyframe.data['raw_depth']
      else:
        assert len(verts) < 256
        self.kpus, self.des = verts, np.array(list(range(len(verts)))*32, np.uint8).reshape(32, len(verts)).T
      self.pts = [None]*len(self.kpus)
    else:
      # fill in later
      self.h, self.w = 0, 0
      self.kpus, self.des, self.pts = None, None, None

    self.id = tid if tid is not None else mapp.add_frame(self)

  def annotate(self, img):
    # paint annotations on the image
    for i1 in range(len(self.kpus)):
      u1, v1 = int(round(self.kpus[i1][0])), int(round(self.kpus[i1][1]))
      if self.pts[i1] is not None:
        if len(self.pts[i1].frames) >= 5:
          cv2.circle(img, (u1, v1), color=(0,255,0), radius=3)
        else:
          cv2.circle(img, (u1, v1), color=(0,128,0), radius=3)
        # draw the trail
        pts = []
        lfid = None
        for f, idx in zip(self.pts[i1].frames[-9:][::-1], self.pts[i1].idxs[-9:][::-1]):
          if lfid is not None and lfid-1 != f.id:
            break
          pts.append(tuple(map(lambda x: int(round(x)), f.kpus[idx])))
          lfid = f.id
        if len(pts) >= 2:
          cv2.polylines(img, np.array([pts], dtype=np.int32), False, myjet[len(pts)]*255, thickness=1, lineType=16)
      else:
        cv2.circle(img, (u1, v1), color=(0,0,0), radius=3)
    return img


  # inverse of intrinsics matrix
  @property
  def Kinv(self):
    if not hasattr(self, '_Kinv'):
      self._Kinv = np.linalg.inv(self.K)
    return self._Kinv

  # normalized keypoints
  @property
  def kps(self):
    if not hasattr(self, '_kps'):
      self._kps = normalize(self.Kinv, self.kpus)
    return self._kps

  # KD tree of unnormalized keypoints
  @property
  def kd(self):
    if not hasattr(self, '_kd'):
      self._kd = cKDTree(self.kpus)
    return self._kd 
