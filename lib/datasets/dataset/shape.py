from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

import pycocotools.coco as coco

from pycocotools.cocoeval import COCOeval

import numpy as np

import json

import os

import torch.utils.data as data



class SHAPES(data.Dataset):
    num_classes = 5
    default_resolution = [128, 128]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    
    def __init__(self, opt, split):
        super(SHAPES, self).__init__()
#         _folder_name = {'train': opt.shape_param + '_' + opt.shape_param_value,
#                         'test': opt.shape_param + '_' + opt.shape_param_value,
#                         'val': opt.shape_param + '_' + opt.shape_param_value,
#                         }

        
        self.class_names = ['__background__', "square", "circle", "triangle", "hexagon", "eclipse"]
        self.default_resolution = [128, 128]

        # ONLY CHANGE 
        self.data_dir =os.path.join(opt.data_dir, 'shape_multicombined/')
        self.img_dir = os.path.join(self.data_dir, 'multicombined_800_train2019')
        self.annot_path = os.path.join(self.data_dir, 'annotations', 'instances_shape_multicombined_800_train2019.json')

        self.max_objs = self.num_classes
        self.class_name = self.class_names[:self.num_classes+1]
        self._valid_ids = np.arange(1, self.num_classes+1, dtype=np.int32)


        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

        self._data_rng = np.random.RandomState(123)

        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],

                                 dtype=np.float32)

        self._eig_vec = np.array([

            [-0.58752847, -0.69563484, 0.41340352],

            [-0.5832747, 0.00994535, -0.81221408],

            [-0.56089297, 0.71832671, 0.41158938]

        ], dtype=np.float32)


        self.split = split

        self.opt = opt

        print('==> initializing shapes {} data.'.format(split))

        self.coco = coco.COCO(self.annot_path)

        self.images = self.coco.getImgIds()

        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):

        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):

        # import pdb; pdb.set_trace()

        detections = []

        for image_id in all_bboxes:

            for cls_ind in all_bboxes[image_id]:

                category_id = self._valid_ids[cls_ind - 1]

                for bbox in all_bboxes[image_id][cls_ind]:

                    bbox[2] -= bbox[0]

                    bbox[3] -= bbox[1]

                    score = bbox[4]

                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {

                        "image_id": int(image_id),

                        "category_id": int(category_id),

                        "bbox": bbox_out,

                        "score": float("{:.2f}".format(score))

                    }

                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))

                        detection["extreme_points"] = extreme_points

                    detections.append(detection)

        return detections

    def __len__(self):

        return self.num_samples

    def save_results(self, results, save_dir):

        json.dump(self.convert_eval_format(results),

                  open('{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):

        # result_json = os.path.join(save_dir, "results.json")

        # detections  = self.convert_eval_format(results)

        # json.dump(detections, open(result_json, "w"))

        # import ipdb; ipdb.set_trace()

        self.save_results(results, save_dir)

        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))

        coco_eval = COCOeval(self.coco, coco_dets, "bbox")

        coco_eval.evaluate()

        coco_eval.accumulate()

        coco_eval.summarize()






