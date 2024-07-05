''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2019-09-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-07-07
@LastEditors: Huangying Zhan
@Description: this file contains different correspondence selection methods
'''

import math
import numpy as np
import cv2

def visualize_keypoints_on_two_images(ref_image, cur_image, filtered_kp1, filtered_kp2):
    """
    在两张图片上可视化两组坐标点，并用线段连接对应的关键点

    参数:
    - ref_image: 第一张图片，用于显示第一组坐标点
    - cur_image: 第二张图片，用于显示第二组坐标点
    - filtered_kp1: 第一组坐标，形状为 (N, 2)
    - filtered_kp2: 第二组坐标，形状为 (N, 2)
    """
    
    # 复制图像以绘制关键点和连线
    ref_image_with_kp = ref_image.copy()
    cur_image_with_kp = cur_image.copy()
    
    for (x1, y1), (x2, y2) in zip(filtered_kp1, filtered_kp2):
        # 绘制关键点
        cv2.circle(ref_image_with_kp, (int(x1), int(y1)), 3, (255, 0, 0), -1)  # Blue keypoints on ref_image
        cv2.circle(cur_image_with_kp, (int(x2), int(y2)), 3, (0, 0, 255), -1)  # Red keypoints on cur_image
        
        # 绘制连线
        color = (0, 255, 0)  # Green line
        cv2.line(ref_image_with_kp, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
    
    # 显示图像
    cv2.imshow('Keypoints Visualization on Two Images', ref_image_with_kp)
    cv2.waitKey(10)
    # cv2.destroyAllWindows()

def visualize_keypoints_on_combined_image(ref_image, cur_image, filtered_kp1, filtered_kp2):
    """
    在拼接的图片上可视化两组坐标点，并将对应的蓝点和红点用直线连接起来

    参数:
    - ref_image: 第一张图片，用于显示第一组坐标点
    - cur_image: 第二张图片，用于显示第二组坐标点
    - filtered_kp1: 第一组坐标，形状为 (N, 2)
    - filtered_kp2: 第二组坐标，形状为 (N, 2)
    """
    
    # 创建一个空白画布，高度为两张图像的最大高度，宽度为两张图像宽度之和
    combined_image = np.hstack((ref_image, cur_image))

    # 在拼接的图像上绘制关键点和连线
    offset_x = ref_image.shape[1]
    for (x1, y1), (x2, y2) in zip(filtered_kp1, filtered_kp2):
        # 绘制蓝色关键点
        cv2.circle(combined_image, (int(x1), int(y1)), 3, (255, 0, 0), -1)
        # 绘制红色关键点
        cv2.circle(combined_image, (int(x2) + offset_x, int(y2)), 3, (0, 0, 255), -1)
        # 绘制连接线
        cv2.line(combined_image, (int(x1), int(y1)), (int(x2) + offset_x, int(y2)), (0, 255, 0), 1)

    # 显示拼接的图像
    cv2.imshow('Combined Image with Keypoints', combined_image)
    cv2.waitKey(10)
    # cv2.destroyAllWindows()

def convert_idx_to_global_coord(local_idx, local_kp_list, x0):
    """Convert pixel index on a local window to global pixel index

    Args: 
        local_idx (list): K indexs of selected pixels 
        local_kp_list (4xN): 
        x0 (list): top-left offset of the local window [y, x]

    Returns:
        coord (array, [4xK]): selected pixels on global image coordinate 
    """
    coord = [local_kp_list[0][local_idx], local_kp_list[1][local_idx], local_kp_list[2][local_idx], local_kp_list[3][local_idx]]
    coord = np.asarray(coord)
    coord[1] += x0[0] # h
    coord[2] += x0[1] # w
    return coord


def bestN_flow_kp(kp1, kp2, ref_data, cfg, outputs):
    """select best-N keypoints with least flow inconsistency
    
    Args:
        kp1 (array, [1xHxWx2]): keypoints on view-1
        kp2 (array, [1xHxWx2]): keypoints on view-2
        ref_data (dict): data of reference view, a dictionary containing

            - **id** (int): index
            - **flow_diff** (array, [HxWx1]): flow difference
        cfg (edict): configuration dictionary
        outputs (dict): output data dictionary
    
    Returns:
        outputs (dict): output data dictionary. New data added

            - **kp1_best** (array, [Nx2]): keypoints on view-1
            - **kp2_best** (array, [Nx2]): keypoints on view-
    """
    bestN_cfg = cfg.kp_selection.bestN

    # initialization
    N = bestN_cfg.num_bestN

    # get data
    flow_diff = np.expand_dims(ref_data['flow_diff'], 0)

    # kp selection
    tmp_kp_list = np.where(flow_diff >= 0) # select all points as intialization
    sel_list = np.argpartition(flow_diff[tmp_kp_list], N)[:N]
    sel_kps = convert_idx_to_global_coord(sel_list, tmp_kp_list, [0, 0])

    kp1_best = kp1[:, sel_kps[1], sel_kps[2]]
    kp2_best = kp2[:, sel_kps[1], sel_kps[2]]

    outputs['kp1_best'] = kp1_best
    outputs['kp2_best'] = kp2_best
    outputs['fb_flow_mask'] = flow_diff[0,:,:,0]
    return outputs


def local_bestN(kp1, kp2, ref_data, cfg, outputs):
    """select best-N filtered keypoints from uniformly divided regions
    
    Args:
        kp1 (array, [1xHxWx2]): keypoints on view-1
        kp2 (array, [1xHxWx2]): keypoints on view-2
        ref_data (dict): data of reference view, a dictionary containing

            - **id** (int): index
            - **flow_diff** (array, [HxWx1]): flow difference
            - **depth_diff** (array, [HxWx1]): depth difference
        cfg (edict): configuration dictionary
        outputs (dict): output data dictionary
    
    Returns:
        outputs (dict): output data dictionary. New data added

            - **kp1_best** (array, [Nx2]): keypoints on view-1
            - **kp2_best** (array, [Nx2]): keypoints on view-2

    """
    # configuration setup
    kp_cfg = cfg.kp_selection
    bestN_cfg = cfg.kp_selection.local_bestN

    # initialization
    num_row = bestN_cfg.num_row #10
    num_col = bestN_cfg.num_col #10
    N = bestN_cfg.num_bestN #这里选了2000个点
    score_method = bestN_cfg.score_method #flow
    flow_diff_thre = bestN_cfg.thre #0.1
    depth_diff_thre = kp_cfg.depth_consistency.thre #0.05
    good_region_cnt = 0

    h, w, _ = ref_data['flow_diff'].shape
    
    #目前就想知道这里面输出的是不是坐标，如果是坐标的话，我可以直接把kp-3d里的数据加进去
    outputs['kp1_best'] = {}
    outputs['kp2_best'] = {}

    #这个是计算每个区域的关键点的个数
    n_best = math.floor(N/(num_col*num_row))
    sel_kps = []

    # get data
    flow_diff = np.expand_dims(ref_data['flow_diff'], 0)
    if kp_cfg.depth_consistency.enable:
        depth_diff = ref_data['depth_diff'].reshape(1, h, w, 1)
    
    # Insufficent keypoint case 1   #如果没有足够多的鲁棒点
    if (flow_diff[0,:,:,0] < flow_diff_thre).sum() < N * 0.1 :
        print("Cannot find enough good keypoints!")
        outputs['good_kp_found'] = False
        return outputs

    for row in range(num_row):
        for col in range(num_col):
            x0 = [int(h/num_row*row), int(w/num_col*col)] # top_left
            x1 = [int(h/num_row*(row+1))-1, int(w/num_col*(col+1))-1] # bottom right

            # computing masks
            tmp_flow_diff = flow_diff[:, x0[0]:x1[0], x0[1]:x1[1]].copy()

            if score_method == "flow":
                flow_mask = tmp_flow_diff < flow_diff_thre
            elif score_method == "flow_ratio":
                tmp_flow = np.expand_dims(ref_data['flow'][:, x0[0]:x1[0], x0[1]:x1[1]], 0)
                tmp_flow = np.transpose(tmp_flow, (0, 2, 3, 1))
                tmp_flow_diff_ratio = tmp_flow_diff / np.linalg.norm(tmp_flow, axis=3, keepdims=True)
                flow_mask = tmp_flow_diff_ratio < flow_diff_thre

            valid_mask = flow_mask

            if kp_cfg.depth_consistency.enable:
                tmp_depth_diff = depth_diff[:, x0[0]:x1[0],x0[1]:x1[1]].copy()
                depth_mask = tmp_depth_diff < depth_diff_thre
                valid_mask *= depth_mask
            
            # computing scores
            if score_method == 'flow':
                score = tmp_flow_diff
            elif score_method == 'flow_depth':
                score = tmp_flow_diff * tmp_depth_diff
            elif score_method == 'flow_ratio':
                score = tmp_flow_diff_ratio

            # kp selection
            tmp_kp_list = np.where(valid_mask)

            num_to_pick = min(n_best, len(tmp_kp_list[0]))

            if num_to_pick != 0:
                good_region_cnt += 1
            
            if num_to_pick <= n_best:
                sel_list = np.argpartition(score[tmp_kp_list], num_to_pick-1)[:num_to_pick]
            else:
                sel_list = np.argpartition(score[tmp_kp_list], num_to_pick)[:num_to_pick]
            
            sel_global_coords = convert_idx_to_global_coord(sel_list, tmp_kp_list, x0)#sel_list选的点
            for i in range(sel_global_coords.shape[1]):
                sel_kps.append(sel_global_coords[:, i:i+1])
    
    # Insufficent keypoint case 2
    if good_region_cnt < (num_row * num_col) * 0.1:
        print("Cannot find enough good keypoints from diversed regions!")
        outputs['good_kp_found'] = False
        return outputs

    # reshape selected keypoints
    sel_kps = np.asarray(sel_kps)
    sel_kps = np.transpose(sel_kps, (1, 0, 2))
    sel_kps = np.reshape(sel_kps, (4, -1))

    kp1_best = kp1[:, sel_kps[1], sel_kps[2]]
    kp2_best = kp2[:, sel_kps[1], sel_kps[2]]

    #提供的是两个图像的关键点
    outputs['kp1_best'] = kp1_best
    print(kp1_best)
    outputs['kp2_best'] = kp2_best
    # mask generation
    if score_method == 'flow_ratio':
        flow = np.expand_dims(ref_data['flow'], 0)
        flow = np.transpose(flow, (0, 2, 3, 1))
        flow_diff_ratio = flow_diff / np.linalg.norm(flow, axis=3, keepdims=True)
        outputs['fb_flow_mask'] = flow_diff_ratio[0,:,:,0]
    elif score_method == 'flow':
        outputs['fb_flow_mask'] = flow_diff[0,:,:,0]
    return outputs

def local_bestN1(kp1, kp2, ref_data,cur_data,tmp_flow_diff, cfg, outputs):
    """select best-N filtered keypoints from uniformly divided regions
    
    Args:
        kp1 (array, [1xHxWx2]): keypoints on view-1
        kp2 (array, [1xHxWx2]): keypoints on view-2
        ref_data (dict): data of reference view, a dictionary containing

            - **id** (int): index
            - **flow_diff** (array, [HxWx1]): flow difference
            - **depth_diff** (array, [HxWx1]): depth difference
        cfg (edict): configuration dictionary
        outputs (dict): output data dictionary
    
    Returns:
        outputs (dict): output data dictionary. New data added

            - **kp1_best** (array, [Nx2]): keypoints on view-1
            - **kp2_best** (array, [Nx2]): keypoints on view-2

    """
    # configuration setup
    kp_cfg = cfg.kp_selection
    bestN_cfg = cfg.kp_selection.local_bestN

    # initialization
    num_row = bestN_cfg.num_row #10
    num_col = bestN_cfg.num_col #10
    N = 1000 #这里选了2000个点
    score_method = bestN_cfg.score_method #flow
    flow_diff_thre = bestN_cfg.thre #0.1
    depth_diff_thre = kp_cfg.depth_consistency.thre #0.05
    good_region_cnt = 0

    h, w, _ = ref_data['flow_diff'].shape
    
    #目前就想知道这里面输出的是不是坐标，如果是坐标的话，我可以直接把kp-3d里的数据加进去
    outputs['kp1_best'] = {}
    outputs['kp2_best'] = {}

    #这个是计算每个区域的关键点的个数
    n_best = math.floor(N/(num_col*num_row))
    sel_kps = []

    # get data
    flow_diff = np.expand_dims(ref_data['flow_diff'], 0)
    # print("？？？？？",flow_diff)
    if kp_cfg.depth_consistency.enable:
        depth_diff = ref_data['depth_diff'].reshape(1, h, w, 1)
    
    # Insufficent keypoint case 1   #如果没有足够多的鲁棒点
    if (flow_diff[0,:,:,0] < flow_diff_thre).sum() < N * 0.1 :
        print("Cannot find enough good keypoints!")
        outputs['good_kp_found'] = False
        return outputs
    
    
    # #这段代码的作用就是分块选点，而基于superpoint的方法不需要选点
    # for row in range(num_row):
    #     for col in range(num_col):
    #         x0 = [int(h/num_row*row), int(w/num_col*col)] # top_left
    #         x1 = [int(h/num_row*(row+1))-1, int(w/num_col*(col+1))-1] # bottom right

    #         # computing masks
    #         # 得到的是这个区域的点的光流变化
    #         tmp_flow_diff = flow_diff[:, x0[0]:x1[0], x0[1]:x1[1]].copy()
    #         print("@@@@@@",tmp_flow_diff.shape)
    #         if score_method == "flow":
    #             flow_mask = tmp_flow_diff < flow_diff_thre
    #         elif score_method == "flow_ratio":
    #             tmp_flow = np.expand_dims(ref_data['flow'][:, x0[0]:x1[0], x0[1]:x1[1]], 0)
    #             tmp_flow = np.transpose(tmp_flow, (0, 2, 3, 1))
    #             tmp_flow_diff_ratio = tmp_flow_diff / np.linalg.norm(tmp_flow, axis=3, keepdims=True)
    #             flow_mask = tmp_flow_diff_ratio < flow_diff_thre

    #         valid_mask = flow_mask

    #         if kp_cfg.depth_consistency.enable:
    #             tmp_depth_diff = depth_diff[:, x0[0]:x1[0],x0[1]:x1[1]].copy()
    #             depth_mask = tmp_depth_diff < depth_diff_thre
    #             valid_mask *= depth_mask
            
    #         # computing scores
    #         if score_method == 'flow':
    #             score = tmp_flow_diff
    #         elif score_method == 'flow_depth':
    #             score = tmp_flow_diff * tmp_depth_diff
    #         elif score_method == 'flow_ratio':
    #             score = tmp_flow_diff_ratio

    #         # kp selection
    #         tmp_kp_list = np.where(valid_mask)

    #         num_to_pick = min(n_best, len(tmp_kp_list[0]))

    #         if num_to_pick != 0:
    #             good_region_cnt += 1
            
    #         if num_to_pick <= n_best:
    #             sel_list = np.argpartition(score[tmp_kp_list], num_to_pick-1)[:num_to_pick]
    #         else:
    #             sel_list = np.argpartition(score[tmp_kp_list], num_to_pick)[:num_to_pick]
            
    #         sel_global_coords = convert_idx_to_global_coord(sel_list, tmp_kp_list, x0)#sel_list选的点
    #         for i in range(sel_global_coords.shape[1]):
    #             sel_kps.append(sel_global_coords[:, i:i+1])
    
    # # Insufficent keypoint case 2
    # if good_region_cnt < (num_row * num_col) * 0.1:
    #     print("Cannot find enough good keypoints from diversed regions!")
    #     outputs['good_kp_found'] = False
    #     return outputs

    flow_mask = tmp_flow_diff < flow_diff_thre
    # print(tmp_flow_diff)
    valid_mask = flow_mask
    # print(valid_mask)
    # print("valid_mask:",valid_mask.shape)
    tmp_kp_list = np.where(valid_mask)
    filtered_kp1 = kp1[tmp_kp_list[:-1]]  # 去除最后一个维度的索引，因为 valid_mask 有一个额外的维度
    filtered_kp2 = kp2[tmp_kp_list[:-1]]
    # print("filtered_kp1:",filtered_kp1.shape)
    # print("filtered_kp2:",filtered_kp2.shape)
    # print(filtered_kp1)
    # print(filtered_kp2)
    # visualize_keypoints_on_combined_image(ref_data["img"],cur_data["img"], filtered_kp1, filtered_kp2)




    filtered_kp1 = np.expand_dims(filtered_kp1, axis=0)
    filtered_kp2 = np.expand_dims(filtered_kp2, axis=0)
    # print(filtered_kp1)
    # print(filtered_kp2)
    # print(tmp_kp_list.shape)
    # # reshape selected keypoints
    # sel_kps = np.asarray(sel_kps)
    # sel_kps = np.transpose(sel_kps, (1, 0, 2))
    # sel_kps = np.reshape(sel_kps, (4, -1))

    # print("kp1 的形状:", kp1.shape)
    # print("kp2 的形状:", kp2.shape)
    # print("sel_kps 的形状:", sel_kps.shape)
    # print("sel_kps 的值:", sel_kps)
    
    

    #提供的是两个图像的关键点
    outputs['kp1_best'] = filtered_kp1
    outputs['kp2_best'] = filtered_kp2
    # mask generation
    if score_method == 'flow_ratio':
        flow = np.expand_dims(ref_data['flow'], 0)
        flow = np.transpose(flow, (0, 2, 3, 1))
        flow_diff_ratio = flow_diff / np.linalg.norm(flow, axis=3, keepdims=True)
        outputs['fb_flow_mask'] = flow_diff_ratio[0,:,:,0]
    elif score_method == 'flow':
        outputs['fb_flow_mask'] = flow_diff[0,:,:,0]
    return outputs

def opt_rigid_flow_kp(kp1, kp2, ref_data, cfg, outputs, score_method):
    """select best-N filtered keypoints from uniformly divided regions 
    with rigid-flow mask
    
    Args:
        kp1 (array, [1xHxWx2]): keypoints on view-1
        kp2 (array, [1xHxWx2]): keypoints on view-2
        ref_data (dict):

            - **rigid_flow_diff** (array, [HxWx1]): rigid-optical flow consistency
            - **flow_diff** (array, [HxWx1]): forward-backward flow consistency
        cfg (edict): configuration dictionary
        outputs (dict): output data 
        method (str): [uniform, best]
        score_method (str): [opt_flow, rigid_flow]
    
    Returns:
        outputs (dict): output data with the new data

            - **kp1_depth** (array, [Nx2]): keypoints in view-1, best in terms of score_method
            - **kp2_depth** (array, [Nx2]): keypoints in view-2, best in terms of score_method
            - **kp1_depth_uniform** (array, [Nx2]): keypoints in view-1, uniformly sampled
            - **kp2_depth_uniform** (array, [Nx2]): keypoints in view-2, uniformly sampled
            - **rigid_flow_mask** (array, [HxW]): rigid-optical flow consistency 
    """
    kp_cfg = cfg.kp_selection
    bestN_cfg = cfg.kp_selection.rigid_flow_kp

    # initialization
    num_row = bestN_cfg.num_row
    num_col = bestN_cfg.num_col
    N = bestN_cfg.num_bestN
    rigid_flow_diff_thre = kp_cfg.rigid_flow_kp.rigid_flow_thre
    opt_flow_diff_thre = kp_cfg.rigid_flow_kp.optical_flow_thre

    n_best = math.floor(N/(num_col*num_row))
    sel_kps = []
    sel_kps_uniform = []

    # get data
    # flow diff
    rigid_flow_diff = ref_data['rigid_flow_diff']
    rigid_flow_diff = np.expand_dims(rigid_flow_diff, 0)
    _, h, w, _ = rigid_flow_diff.shape

    # optical flow diff
    opt_flow_diff = np.expand_dims(ref_data['flow_diff'], 0)

    for row in range(num_row):
        for col in range(num_col):
            x0 = [int(h/num_row*row), int(w/num_col*col)] # top_left
            x1 = [int(h/num_row*(row+1))-1, int(w/num_col*(col+1))-1] # bottom right

            # computing masks
            tmp_opt_flow_diff = opt_flow_diff[:, x0[0]:x1[0], x0[1]:x1[1]].copy()

            tmp_rigid_flow_diff = rigid_flow_diff[:, x0[0]:x1[0], x0[1]:x1[1]].copy()
            flow_mask = (tmp_rigid_flow_diff < rigid_flow_diff_thre) 
            
            flow_mask = flow_mask * (tmp_opt_flow_diff < opt_flow_diff_thre)
            valid_mask = flow_mask

            # computing scores
            if score_method == "rigid_flow":
                score = tmp_rigid_flow_diff
            elif score_method == "opt_flow":
                score = tmp_opt_flow_diff

            # kp selection
            tmp_kp_list = np.where(valid_mask)
            num_to_pick = min(n_best, len(tmp_kp_list[0]))
            
            # Pick uniform kps
            # if method == 'uniform':
            if num_to_pick > 0:
                step = int(len(tmp_kp_list[0]) / (num_to_pick))
                sel_list = np.arange(0, len(tmp_kp_list[0]), step)[:num_to_pick]
            else:
                sel_list = []
            sel_global_coords = convert_idx_to_global_coord(sel_list, tmp_kp_list, x0)
            for i in range(sel_global_coords.shape[1]):
                sel_kps_uniform.append(sel_global_coords[:, i:i+1])

            # elif method == 'best':
            if num_to_pick <= n_best:
                sel_list = np.argpartition(score[tmp_kp_list], num_to_pick-1)[:num_to_pick]
            else:
                sel_list = np.argpartition(score[tmp_kp_list], num_to_pick)[:num_to_pick]

            sel_global_coords = convert_idx_to_global_coord(sel_list, tmp_kp_list, x0)
            for i in range(sel_global_coords.shape[1]):
                sel_kps.append(sel_global_coords[:, i:i+1])

    # best
    sel_kps = np.asarray(sel_kps)
    assert sel_kps.shape[0]!=0, "sampling threshold is too small."
    sel_kps = np.transpose(sel_kps, (1, 0, 2))
    sel_kps = np.reshape(sel_kps, (4, -1))

    kp1_best = kp1[:, sel_kps[1], sel_kps[2]]
    kp2_best = kp2[:, sel_kps[1], sel_kps[2]]

    outputs['kp1_depth'] = kp1_best.copy()
    outputs['kp2_depth'] = kp2_best.copy()

    # uniform
    sel_kps = np.asarray(sel_kps_uniform)
    assert sel_kps.shape[0]!=0, "sampling threshold is too small."
    sel_kps = np.transpose(sel_kps, (1, 0, 2))
    sel_kps = np.reshape(sel_kps, (4, -1))

    kp1_best = kp1[:, sel_kps[1], sel_kps[2]]
    kp2_best = kp2[:, sel_kps[1], sel_kps[2]]

    outputs['kp1_depth_uniform'] = kp1_best.copy()
    outputs['kp2_depth_uniform'] = kp2_best.copy()



    # mask generation
    outputs['rigid_flow_mask'] = rigid_flow_diff[0,:,:,0]
    return outputs


def sampled_kp(kp1, kp2, ref_data, kp_list, cfg, outputs):
    """select sampled keypoints with given keypoint index list
    
    Args:
        kp1 (array, [1xHxWx2]): keypoints on view-1
        kp2 (array, [1xHxWx2]): keypoints on view-2
        ref_data (dict): data of reference view, a dictionary containing
        kp_list (list): list of keypoint index
        cfg (edict): configuration dictionary
        outputs (dict): output data dictionary
    
    Returns:
        outputs (dict): output data dictionary. New data added

            - **kp1_list** (array, [Nx2]): keypoints on view-1
            - **kp2_list** (array, [Nx2]): keypoints on view-2
    """
    kp_cfg = cfg.kp_selection
    img_crop = cfg.crop.flow_crop

    # initialization
    h, w = ref_data['depth'].shape
    n = 1

    outputs['kp1_list'] = {}
    outputs['kp2_list'] = {}

    y0, y1 = 0, h
    x0, x1 = 0, w

    # Get uniform sampled keypoints
    if img_crop is not None:
        y0, y1 = int(h*img_crop[0][0]), int(h*img_crop[0][1])
        x0, x1 = int(w*img_crop[1][0]), int(w*img_crop[1][1])

        kp1 = kp1[:, y0:y1, x0:x1]
        kp2 = kp2[:, y0:y1, x0:x1]

    kp1_list = kp1.reshape(n, -1, 2)
    kp2_list = kp2.reshape(n, -1, 2)
    kp1_list = np.transpose(kp1_list, (1,0,2))
    kp2_list = np.transpose(kp2_list, (1,0,2))

    # select kp from sampled kp_list
    kp1_list = kp1_list[kp_list]
    kp2_list = kp2_list[kp_list]
    kp1_list = np.transpose(kp1_list, (1,0,2))
    kp2_list = np.transpose(kp2_list, (1,0,2))

    outputs['kp1_list'] = kp1_list
    outputs['kp2_list'] = kp2_list
    return outputs
