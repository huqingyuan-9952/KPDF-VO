''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-03-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-07-06
@LastEditors: Huangying Zhan
@Description: KeypointSampler is an interface for keypoint sampling
'''


import numpy as np

from .kp_selection import *
from libs.general.utils import image_grid
from libs.geometry.camera_modules import SE3

class KeypointSampler():
    """KeypointSampler is an interface for keypoint sampling 
    """
    def __init__(self, cfg):
        """
        Args:
            cfg (edict): configuration dictionary
        """
        self.cfg = cfg
        self.kps = {}

        # generate uniform kp list
        if self.cfg.kp_selection.sampled_kp.enable:
            self.kps['uniform'] = self.generate_kp_samples(
                                        img_h=self.cfg.image.height,
                                        img_w=self.cfg.image.width,
                                        crop=self.cfg.crop.flow_crop,
                                        N=self.cfg.kp_selection.sampled_kp.num_kp
                                        )

    def get_feat_track_methods(self, method_idx):
        """Get feature tracking method
        
        Args:
            method_idx (int): feature tracking method index 
        
        Returns:
            feat_track_method (str): feature tracking method
        """
        feat_track_methods = {
            1: "deep_flow",
        }
        return feat_track_methods[method_idx]

    def generate_kp_samples(self, img_h, img_w, crop, N):
        """generate uniform keypoint samples according to image height, width
        and cropping scheme

        Args:
            img_h (int): image height
            img_w (int): image width
            crop (list): normalized cropping ratio, [[y0, y1],[x0, x1]]
            N (int): number of keypoint

        Returns:
            kp_list (array, [N]): keypoint list
        """
        # get cropped image shape
        y0, y1 = crop[0]
        y0, y1 = int(y0 * img_h), int(y1 * img_h)
        x0, x1 = crop[1]
        x0, x1 = int(x0 * img_w), int(x1 * img_w)

        # uniform sampling keypoints
        total_num = (x1-x0) * (y1-y0) - 1
        kp_list = np.linspace(0, total_num, N, dtype=np.int)
        return kp_list
    def kp_selection(self, cur_data, ref_data):
        """Choose valid kp from a series of operations

        Args:
            cur_data (dict): data of current frame (view-2)
            ref_data (dict): data of reference frame (view-1)
        
        Returns:
            outputs (dict): a dictionary containing some of the following items

                - **kp1_best** (array, [Nx2]): keypoints on view-1
                - **kp2_best** (array, [Nx2]): keypoints on view-2
                - **kp1_list** (array, [Nx2]): keypoints on view-1
                - **kp2_list** (array, [Nx2]): keypoints on view-2  
                - **kp1_depth** (array, [Nx2]): keypoints in view-1
                - **kp2_depth** (array, [Nx2]): keypoints in view-2
                - **rigid_flow_mask** (array, [HxW]): rigid-optical flow consistency 

        """
        outputs = {}
        outputs['good_kp_found'] = True

        # initialization
        h, w = cur_data['depth'].shape

        kp1 = image_grid(h, w)
        kp1 = np.expand_dims(kp1, 0)
        tmp_flow_data = np.transpose(np.expand_dims(ref_data['flow'], 0), (0, 2, 3, 1))
        kp2 = kp1 + tmp_flow_data

        """ best-N selection """
        if self.cfg.kp_selection.local_bestN.enable:
            kp_sel_method = local_bestN
            outputs.update(
                kp_sel_method(
                    kp1=kp1,
                    kp2=kp2,
                    ref_data=ref_data,
                    cfg=self.cfg,
                    outputs=outputs
                    )
            )
        elif self.cfg.kp_selection.bestN.enable:
            kp_sel_method = bestN_flow_kp
            outputs.update(
                kp_sel_method(
                    kp1=kp1,
                    kp2=kp2,
                    ref_data=ref_data,
                    cfg=self.cfg,
                    outputs=outputs
                    )
            )

        """ sampled kp selection """
        if self.cfg.kp_selection.sampled_kp.enable:
            outputs.update(
                sampled_kp(
                    kp1=kp1,
                    kp2=kp2,
                    ref_data=ref_data,
                    kp_list=self.kps['uniform'],
                    cfg=self.cfg,
                    outputs=outputs
                    )
        )  

        return outputs
    
    def kp_selection1(self, cur_data, ref_data,kp1):
        """Choose valid kp from a series of operations

        Args:
            cur_data (dict): data of current frame (view-2)
            ref_data (dict): data of reference frame (view-1)
        
        Returns:
            outputs (dict): a dictionary containing some of the following items

                - **kp1_best** (array, [Nx2]): keypoints on view-1
                - **kp2_best** (array, [Nx2]): keypoints on view-2
                - **kp1_list** (array, [Nx2]): keypoints on view-1
                - **kp2_list** (array, [Nx2]): keypoints on view-2  
                - **kp1_depth** (array, [Nx2]): keypoints in view-1
                - **kp2_depth** (array, [Nx2]): keypoints in view-2
                - **rigid_flow_mask** (array, [HxW]): rigid-optical flow consistency 

        """
        outputs = {}
        outputs['good_kp_found'] = True

        # initialization
        h, w = cur_data['depth'].shape
        #筛选特征点
        kp1 = kp1.cpu()
        kp1 = kp1.squeeze(0)
        kp1 = kp1.permute(1, 2, 0)
        kp1 = kp1.detach().numpy()
        # print(kp1.shape)
        kp1 = kp1.astype(int)
        # kp1 = image_grid(h, w)
        # print(kp1.shape)
        kp1 = np.expand_dims(kp1, 0)
        tmp_flow_data = np.transpose(np.expand_dims(ref_data['flow'], 0), (0, 2, 3, 1))
        flow_diff = np.expand_dims(ref_data['flow_diff'], 0)
        # print(tmp_flow_data.shape)
        # print(flow_diff.shape)
        kp2 = np.zeros_like(kp1, dtype=np.float32)
        tmp_flow_diff = np.zeros((1,24,80,1), dtype=np.float32)
        for i in range(kp1.shape[1]):  # 24
            for j in range(kp1.shape[2]):  # 80
                x, y = kp1[0, i, j]
                # 确保索引在tmp_flow_data的范围内
                if 0 <= x < tmp_flow_data.shape[2] and 0 <= y < tmp_flow_data.shape[1]:
                    flow_value = tmp_flow_data[0, y, x]############
                    flow_diff_value = flow_diff[0, y, x]#########
                    kp2[0, i, j] = [x, y] + flow_value
                    tmp_flow_diff[0, i, j] = flow_diff_value
        # print(kp2.shape)
        # print(tmp_flow_diff.shape)

        """ best-N selection """
        # if self.cfg.kp_selection.local_bestN.enable:
        #     kp_sel_method = local_bestN
        #     outputs.update(
        #         kp_sel_method(
        #             kp1=kp1,
        #             kp2=kp2,
        #             ref_data=ref_data,
        #             cfg=self.cfg,
        #             outputs=outputs
        #             )
        #     )
        if self.cfg.kp_selection.local_bestN.enable:
            kp_sel_method = local_bestN1
            outputs.update(
                kp_sel_method(
                    kp1=kp1,
                    kp2=kp2,
                    ref_data=ref_data,
                    cur_data=cur_data,
                    cfg=self.cfg,
                    outputs=outputs,
                    tmp_flow_diff = tmp_flow_diff
                    )
            )
        elif self.cfg.kp_selection.bestN.enable:
            kp_sel_method = bestN_flow_kp
            outputs.update(
                kp_sel_method(
                    kp1=kp1,
                    kp2=kp2,
                    ref_data=ref_data,
                    cfg=self.cfg,
                    outputs=outputs
                    )
            )

        """ sampled kp selection """
        if self.cfg.kp_selection.sampled_kp.enable:
            outputs.update(
                sampled_kp(
                    kp1=kp1,
                    kp2=kp2,
                    ref_data=ref_data,
                    kp_list=self.kps['uniform'],
                    cfg=self.cfg,
                    outputs=outputs
                    )
        )  




        return outputs

    def update_kp_data(self, cur_data, ref_data, kp_sel_outputs):
        """update cur_data and ref_data with the kp_selection output

        Args:
            cur_data (dict): data of current frame
            ref_data (dict): data of reference frame
            kp_sel_outputs (dict): data of keypoint selection outputs
        """
        if self.cfg.kp_selection.local_bestN.enable or self.cfg.kp_selection.bestN.enable:
            # save selected kp
            ref_data['kp_best'] = kp_sel_outputs['kp1_best'][0]
            cur_data['kp_best'] = kp_sel_outputs['kp2_best'][0]
            # print(ref_data['kp_best'].shape)
            # print(kp_sel_outputs['kp1_best'].shape)
            
            
            # save mask
            cur_data['fb_flow_mask'] = kp_sel_outputs['fb_flow_mask']
            
        if self.cfg.kp_selection.sampled_kp.enable:
            ref_data['kp_list'] = kp_sel_outputs['kp1_list'][0]
            cur_data['kp_list'] = kp_sel_outputs['kp2_list'][0]

        visualize_keypoints_on_combined_image(ref_data["img"],cur_data["img"], ref_data['kp_best'], cur_data['kp_best'])
        