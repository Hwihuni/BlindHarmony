# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains content licensed by https://github.com/xinntao/BasicSR/blob/master/LICENSE/LICENSE

import os
import subprocess
import torch.utils.data as data
import numpy as np
import time
import torch
import pickle
import math

class BRATS(data.Dataset):
    def __init__(self, path, mode=None, transform=None):

        self.hr_images = self.load_pkls(path)
        self.mode = mode
        self.transform = transform
        
    def load_pkls( self,path):
        assert os.path.isfile(path), path
        images = []
        with open(path, "rb") as f:
            images += pickle.load(f)
        assert len(images) > 0, path
        # images = images[0:500]
        # to change shape to (num, c, h, w)
        # images = np.expand_dims(np.squeeze(np.array(images)), axis = 1)
        return images
    
    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, item):
        gt = np.abs(self.hr_images[item].astype('float32'))

        
        if self.transform is not None:
            gt_min, gt_max = gt.min(), gt.max()
            # gt_min = 0
            gt = ((gt - gt_min)/(gt_max - gt_min)).astype('float32')
            gt = self.transform(torch.from_numpy(gt))
        else:
            gt = (gt-np.mean(gt))/np.std(gt)
            gt = torch.from_numpy(gt)
        

        #return gt.transpose([2, 0, 1]),gt15.transpose([2, 0, 1])
        # return gt.permute(2,0,1), gt15.permute(2,0,1)
        return gt.permute(2,0,1),gt