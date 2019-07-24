from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#mehdi
from .ctdet import CtdetTrainer
#from .ctdet import WidthHeightUNet
from .ddd import DddTrainer
from .exdet import ExdetTrainer
from .multi_pose import MultiPoseTrainer

train_factory = {
  'exdet': ExdetTrainer, 
  'ddd': DddTrainer,
  #Mehdi
  #'ctdet': WidthHeightUNet,
  'ctdet':CtdetTrainer,
  'multi_pose': MultiPoseTrainer, 
}
