from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math

class CTDetDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)


    img = cv2.imread(img_path)
    org_img = img

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if self.opt.keep_res:
      input_h = (height | self.opt.pad) + 1
      input_w = (width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:

      s = max(img.shape[0], img.shape[1]) * 1.0
      input_h, input_w = self.opt.input_h, self.opt.input_w #height, width

   
    flipped = False
    if self.split == 'train': #train
      if not self.opt.not_rand_crop:
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf) # cmnt this to run in jupyter lab
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf) # cmnt this to run in jupyter lab
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)

      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1
    trans_input = get_affine_transform(
      c, s, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, 
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)
    #inp = (inp.astype(np.float32) / 255.)
    inp = inp.astype(np.float32)
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

    inp = inp.transpose(2, 0, 1)

    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    num_classes = self.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])
    
    segMapsize = (num_classes,input_h, input_w) #curImg['height'], curImg['width'] #height, width input_h,input_w
    segMapTotal = np.zeros(segMapsize, dtype = np.uint8)
    #segMapaffine = np.zeros(segMapsize, dtype = np.uint8)
    ctMapsize = (num_classes,input_h, input_w)
    ctMapTotal = np.zeros (ctMapsize, dtype = bool)
    widthMapTotal = np.zeros(ctMapsize, dtype = np.uint8)
    heightMapTotal= np.zeros(ctMapsize, dtype = np.uint8)

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])
      if (ann['category_id']==15 or ann['category_id']==20):
          if ann['category_id']==15:
            ann['category_id']=1
          if ann['category_id']==20:
            ann['category_id']=2
          cls_id = int(self.cat_ids[ann['category_id']])
          if flipped:
            bbox[[0, 2]] = width - bbox[[2, 0]] - 1
          #Mehdi
          bbox[:2] = affine_transform(bbox[:2], trans_output)
          bbox[2:] = affine_transform(bbox[2:], trans_output)
          bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
          bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
          h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
          if h > 0 and w > 0:
            ct = np.array(
              [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            labelMask = self.coco.annToMask(ann) == 1
            if flipped:
              labelMask = labelMask[:, ::-1]
            labelMask = labelMask.astype(np.uint8)
            labelMask = cv2.warpAffine(labelMask, trans_input,
                                 (input_w, input_h),
                                 flags=cv2.INTER_LINEAR)


            #segMapTotal[cls_id,:,:] = segMapTotal[cls_id,:,:] | labelMask
            segMapTotal[cls_id,:,:] = np.logical_or (segMapTotal[cls_id,:,:], labelMask)


            

            sq= self.opt.bb_size
            sq_w = int(max(min(sq, w/2),sq/4))
            sq_h = int(max(min(sq, h/2),sq/4))
                
            for i in range(-sq_w, sq_w):
              for j in range(-sq_h, sq_h):
              #  print( ct_int[0] , ct_int[1] )
                  border_width = max(min(ct_int[1]+j, input_w-1),0)
                  border_height = max(min(ct_int[0]+i, input_h-1),0)
                  ctMapTotal[cls_id, border_width, border_height] = True
                  widthMapTotal[cls_id,  border_width, border_height] = w
                  heightMapTotal[cls_id ,  border_width, border_height] = h

    ret = {'input':inp , 'gt_segmap':segMapTotal, 'gt_ctmap': ctMapTotal.astype(np.uint8), 'gt_widmap':widthMapTotal.astype(np.uint8), 'gt_heimap':heightMapTotal.astype(np.uint8)}
    return img_id,ret #org_img