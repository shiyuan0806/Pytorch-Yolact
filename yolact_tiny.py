import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck
import numpy as np
from itertools import product
from math import sqrt
from typing import List

from data.config import cfg, mask_type
from layers import Detect
from layers.interpolate import InterpolateModule
from backbone import construct_backbone

import torch.backends.cudnn as cudnn
from utils import timer
from utils.functions import MovingAverage
from model import *
# This is required for Pytorch 1.0.1 on Windows to initialize Cuda on some driver versions.
# See the bug report here: https://github.com/pytorch/pytorch/issues/17108
torch.cuda.set_device(0)

# As of March 10, 2019, Pytorch DataParallel still doesn't support JIT Script Modules
use_jit = torch.cuda.device_count() <= 1
if not use_jit:
    print('Multiple GPUs detected! Turning off JIT.')

ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn



#last_conv_size=None
#some_set=[
#
#   [cfg.backbone.pred_aspect_ratios[0],
#    cfg.backbone.pred_scales[0]],
#
#   [cfg.backbone.pred_aspect_ratios[1],
#    cfg.backbone.pred_scales[1]],
#
#   [cfg.backbone.pred_aspect_ratios[2],
#    cfg.backbone.pred_scales[2]],
#
#   [cfg.backbone.pred_aspect_ratios[3],
#    cfg.backbone.pred_scales[3]],
#
#   [cfg.backbone.pred_aspect_ratios[4],
#    cfg.backbone.pred_scales[4]]
#]

def make_priors(conv_h,conv_w,ratios,scales):

    global last_conv_size
    scales=scales
    aspect_ratios=ratios

   # if last_conv_size != (conv_w,conv_h):

    prior_data=[]
    for j,i in product(range(conv_h),range(conv_w)):
        x=(i+0.5)/conv_w
        y=(j+0.5)/conv_h
        for scale,ars in zip(scales,aspect_ratios):
            for ar in ars:
                ar=sqrt(ar)
                w=scale*ar/cfg.max_size
                h=scale/ar/cfg.max_size
                   # h=w
                prior_data +=[x,y,w,h]
    priors=torch.Tensor(prior_data).view(-1,4)
  #  last_conv_size=(conv_w,conv_h)
    return priors

priors_0=make_priors(69,69,cfg.backbone.pred_aspect_ratios[0],cfg.backbone.pred_scales[0])
priors_1=make_priors(35,35,cfg.backbone.pred_aspect_ratios[1],cfg.backbone.pred_scales[1])
priors_2=make_priors(18,18,cfg.backbone.pred_aspect_ratios[2],cfg.backbone.pred_scales[2])
priors_3=make_priors(9,9,cfg.backbone.pred_aspect_ratios[3],cfg.backbone.pred_scales[3])
priors_4=make_priors(5,5,cfg.backbone.pred_aspect_ratios[4],cfg.backbone.pred_scales[4])
print(cfg.backbone.pred_scales[0],cfg.backbone.pred_scales[4])

class Yolact(nn.Module):
    """
    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║   
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║   
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║   
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝ 
    You can set the arguments by chainging them in the backbone config object in config.py.
    Parameters (in cfg.backbone):
        - selected_layers: The indices of the conv layers to use for prediction.
        - pred_scales:     A list with len(selected_layers) containing tuples of scales (see PredictionModule)
        - pred_aspect_ratios: A list of lists of aspect ratios with len(selected_layers) (see PredictionModule)
    """

    def __init__(self):
        super(Yolact,self).__init__()
####################################################
#                   for mainly net                 #
####################################################
        self.backbone = resnet18(pretrained=True)

        self.fpn1=nn.Sequential(
            nn.Conv2d(512,256,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,1,1,0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.fpn2=nn.Sequential(
            nn.Conv2d(1024,256,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,1,1,0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.fpn3=nn.Sequential(
            nn.Conv2d(2048,256,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,1,1,0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv_b=nn.Sequential(
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

        )
        self.conv_c=nn.Sequential(
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

        )
        self.conv_m=nn.Sequential(
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

        )
     
        self.downsample_layers1=nn.Sequential(
            nn.Conv2d(256,256,3,2,1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
        )
        self.downsample_layers2=nn.Sequential(
            nn.Conv2d(256,256,3,2,1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
        )


        self.bbox_layer=nn.Conv2d(256,12,3,1,1)
        self.conf_layer=nn.Conv2d(256,243,3,1,1)
        self.mask_layer=nn.Conv2d(256,96,3,1,1)

        self.semantic_set_conv=nn.Conv2d(256,80,1,1)
##################################################
#                for proto net                   #
##################################################

        self.proto_net1=nn.Sequential(
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.proto_net2=nn.Sequential(
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256,32,1,1)
         )
#########################################################
# forward process                                       #
#########################################################
        self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=200, conf_thresh=0.05, nms_thresh=0.5)

    def forward(self,x):
        out=self.backbone(x)
        x1=out[0]
        x2=out[1]
        x3=out[2]
        x4=out[3]
      
        ######### all of is 256 channels ############## 
        
        x2=self.fpn1(x2)
        x3=self.fpn2(x3)
        x4=self.fpn3(x4)

        x5=self.downsample_layers1(x4)
        x6=self.downsample_layers2(x5)


        x3=x3+torch.nn.functional.interpolate(x4,size=(35,35),mode='bilinear')
        x2=x2+torch.nn.functional.interpolate(x3,size=(69,69),mode='bilinear')

        out=[x2,x3,x4,x5,x6]
  
        ###########for proto_net######################
        proto_x=out[0]
        proto_out=self.proto_net1(proto_x)
        torch.nn.functional.interpolate(proto_out,scale_factor=2,mode='bilinear')
        proto_out=self.proto_net2(proto_out)
        proto_out=torch.nn.functional.relu(proto_out,inplace=True)
        proto_out=proto_out.permute(0,2,3,1).contiguous()
        ###########for prediction#####################

        pred_outs={'loc':[],'conf':[],'mask':[],'priors':[]}

        for i,x in enumerate(out):
            bbox_x=self.conv_b(x)
            conf_x=self.conv_c(x)
            mask_x=self.conv_m(x)

            bbox_x=self.bbox_layer(bbox_x)
            conf_x=self.conf_layer(conf_x)
            mask_x=self.mask_layer(mask_x)

            bbox=bbox_x.permute(0,2,3,1).contiguous().view(x.size(0),-1,4)
            conf=conf_x.permute(0,2,3,1).contiguous().view(x.size(0),-1,81)
            mask=mask_x.permute(0,2,3,1).contiguous().view(x.size(0),-1,32)
            mask=torch.tanh(mask)

            if i==0:
                priors=priors_0
            if i==1:
                priors=priors_1
            if i==2:
                priors=priors_2
            if i==3:
                priors=priors_3
            if i==4:
                priors=priors_4  

            preds={'loc':bbox,'conf':conf,'mask':mask,'priors':priors}

            for k,v in preds.items():
                pred_outs[k].append(v)

        for k,v in pred_outs.items():
            pred_outs[k]=torch.cat(v,-2)

        pred_outs['proto']=proto_out
       
        flag=False
        
        if flag:
            pred_outs['segm']=self.semantic_set_conv(out[0])
            return pred_outs
        else:
            pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)
            return self.detect(pred_outs)
       
            
      
