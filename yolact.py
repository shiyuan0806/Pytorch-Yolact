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


#
#
#class Concat(nn.Module):
#    def __init__(self, nets, extra_params):
#        super().__init__()
#
#        self.nets = nn.ModuleList(nets)
#        self.extra_params = extra_params
#    
#    def forward(self, x):
#        # Concat each along the channel dimension
#        return torch.cat([net(x) for net in self.nets], dim=1, **self.extra_params)
#
#
#
#def make_net(in_channels, conf, include_last_relu=True):
#    """
#    A helper function to take a config setting and turn it into a network.
#    Used by protonet and extrahead. Returns (network, out_channels)
#    """
#    def make_layer(layer_cfg):
#        nonlocal in_channels
#        
#        # Possible patterns:
#        # ( 256, 3, {}) -> conv
#        # ( 256,-2, {}) -> deconv
#        # (None,-2, {}) -> bilinear interpolate
#        # ('cat',[],{}) -> concat the subnetworks in the list
#        #
#        # You know it would have probably been simpler just to adopt a 'c' 'd' 'u' naming scheme.
#        # Whatever, it's too late now.
#        if isinstance(layer_cfg[0], str):
#            layer_name = layer_cfg[0]
#
#            if layer_name == 'cat':
#                nets = [make_net(in_channels, x) for x in layer_cfg[1]]
#                layer = Concat([net[0] for net in nets], layer_cfg[2])
#                num_channels = sum([net[1] for net in nets])
#        else:
#            num_channels = layer_cfg[0]
#            kernel_size = layer_cfg[1]
#
#            if kernel_size > 0:
#                layer = nn.Conv2d(in_channels, num_channels, kernel_size, **layer_cfg[2])
#            else:
#                if num_channels is None:
#                    layer = InterpolateModule(scale_factor=-kernel_size, mode='bilinear', align_corners=False, **layer_cfg[2])
#                else:
#                    layer = nn.ConvTranspose2d(in_channels, num_channels, -kernel_size, **layer_cfg[2])
#        
#        in_channels = num_channels if num_channels is not None else in_channels
#
#        # Don't return a ReLU layer if we're doing an upsample. This probably doesn't affect anything
#        # output-wise, but there's no need to go through a ReLU here.
#        # Commented out for backwards compatibility with previous models
#        # if num_channels is None:
#        #     return [layer]
#        # else:
#        return [layer,nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
#
#    # Use sum to concat together all the component layer lists
#    net = sum([make_layer(x) for x in conf], [])
#    if not include_last_relu:
#        net = net[:-1]
#  
#    return nn.Sequential(*(net[:-1])), in_channels
#
#
#
#class PredictionModule(nn.Module):
#    """
#    The (c) prediction module adapted from DSSD:
#    https://arxiv.org/pdf/1701.06659.pdf
#    Note that this is slightly different to the module in the paper
#    because the Bottleneck block actually has a 3x3 convolution in
#    the middle instead of a 1x1 convolution. Though, I really can't
#    be arsed to implement it myself, and, who knows, this might be
#    better.
#    Args:
#        - in_channels:   The input feature size.
#        - out_channels:  The output feature size (must be a multiple of 4).
#        - aspect_ratios: A list of lists of priorbox aspect ratios (one list per scale).
#        - scales:        A list of priorbox scales relative to this layer's convsize.
#                         For instance: If this layer has convouts of size 30x30 for
#                                       an image of size 600x600, the 'default' (scale
#                                       of 1) for this layer would produce bounding
#                                       boxes with an area of 20x20px. If the scale is
#                                       .5 on the other hand, this layer would consider
#                                       bounding boxes with area 10x10px, etc.
#        - parent:        If parent is a PredictionModule, this module will use all the layers
#                         from parent instead of from this module.
#    """
#    
#    def __init__(self, in_channels, out_channels=1024, aspect_ratios=[[1]], scales=[1], parent=None):
#        super().__init__()
#
#        self.num_classes = cfg.num_classes
#        self.mask_dim    = cfg.mask_dim
#        self.num_priors  = sum(len(x) for x in aspect_ratios)
#        self.parent      = [parent] # Don't include this in the state dict
#
#        if cfg.mask_proto_prototypes_as_features:
#            in_channels += self.mask_dim
#        
#        if parent is None:
#            if cfg.extra_head_net is None:
#                out_channels = in_channels
#            else:
#                self.upfeature, out_channels = make_net(in_channels, cfg.extra_head_net)
#
#            if cfg.use_prediction_module:
#                self.block = Bottleneck(out_channels, out_channels // 4)
#                self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
#                self.bn = nn.BatchNorm2d(out_channels)
#
#            self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4,                **cfg.head_layer_params)
#            self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes, **cfg.head_layer_params)
#            self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim,    **cfg.head_layer_params)
#
#            if cfg.use_instance_coeff:
#                self.inst_layer = nn.Conv2d(out_channels, self.num_priors * cfg.num_instance_coeffs, **cfg.head_layer_params)
#            
#            # What is this ugly lambda doing in the middle of all this clean prediction module code?
#            def make_extra(num_layers):
#                if num_layers == 0:
#                    return lambda x: x
#                else:
#                    # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
#                    return nn.Sequential(*sum([[
#                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#                        nn.ReLU(inplace=True)
#                    ] for _ in range(num_layers)], []))
#
#            self.bbox_extra, self.conf_extra, self.mask_extra = [make_extra(x) for x in cfg.extra_layers]
#            
#            if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_coeff_gate:
#                self.gate_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, kernel_size=3, padding=1)
#
#        self.aspect_ratios = aspect_ratios
#        self.scales = scales
#
#        self.priors = None
#        self.last_conv_size = None
#
#    def forward(self, x):
#        """
#        Args:
#            - x: The convOut from a layer in the backbone network
#                 Size: [batch_size, in_channels, conv_h, conv_w])
#        Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes
#            - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]
#            - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
#            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_dim]
#            - prior_boxes: [conv_h*conv_w*num_priors, 4]
#        """
#        # In case we want to use another module's layers
#        src = self if self.parent[0] is None else self.parent[0]
#        
#        conv_h = x.size(2)
#        conv_w = x.size(3)
#        
#        if cfg.extra_head_net is not None:
#            x = src.upfeature(x)
#        
#        if cfg.use_prediction_module:
#            # The two branches of PM design (c)
#            a = src.block(x)
#            
#            b = src.conv(x)
#            b = src.bn(b)
#            b = F.relu(b)
#            
#            # TODO: Possibly switch this out for a product
#            x = a + b
#
#        bbox_x = src.bbox_extra(x)
#        conf_x = src.conf_extra(x)
#        mask_x = src.mask_extra(x)
#
#        bbox = src.bbox_layer(bbox_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
#        conf = src.conf_layer(conf_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
#        if cfg.eval_mask_branch:
#            mask = src.mask_layer(mask_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
#        else:
#            mask = torch.zeros(x.size(0), bbox.size(1), self.mask_dim, device=bbox.device)
#
#        if cfg.use_instance_coeff:
#            inst = src.inst_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, cfg.num_instance_coeffs)
#
#        # See box_utils.decode for an explanation of this
#        if cfg.use_yolo_regressors:
#            bbox[:, :, :2] = torch.sigmoid(bbox[:, :, :2]) - 0.5
#            bbox[:, :, 0] /= conv_w
#            bbox[:, :, 1] /= conv_h
#
#        if cfg.eval_mask_branch:
#            if cfg.mask_type == mask_type.direct:
#                mask = torch.sigmoid(mask)
#            elif cfg.mask_type == mask_type.lincomb:
#                mask = cfg.mask_proto_coeff_activation(mask)
#
#                if cfg.mask_proto_coeff_gate:
#                    gate = src.gate_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
#                    mask = mask * torch.sigmoid(gate)
#        
#        priors = self.make_priors(conv_h, conv_w)
#
#        preds = { 'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors }
#
#        if cfg.use_instance_coeff:
#            preds['inst'] = inst
#        
#        return preds
#    
#    def make_priors(self, conv_h, conv_w):
#        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
#        
#        with timer.env('makepriors'):
#            if self.last_conv_size != (conv_w, conv_h):
#                prior_data = []
#
#                # Iteration order is important (it has to sync up with the convout)
#                for j, i in product(range(conv_h), range(conv_w)):
#                    # +0.5 because priors are in center-size notation
#                    x = (i + 0.5) / conv_w
#                    y = (j + 0.5) / conv_h
#                    
#                    for scale, ars in zip(self.scales, self.aspect_ratios):
#                        for ar in ars:
#                            if not cfg.backbone.preapply_sqrt:
#                                ar = sqrt(ar)
#
#                            if cfg.backbone.use_pixel_scales:
#                                w = scale * ar / cfg.max_size
#                                h = scale / ar / cfg.max_size
#                            else:
#                                w = scale * ar / conv_w
#                                h = scale / ar / conv_h
#                            
#                            # This is for backward compatability with a bug where I made everything square by accident
#                            if cfg.backbone.use_square_anchors:
#                                h = w
#
#                            prior_data += [x, y, w, h]
#                
#                self.priors = torch.Tensor(prior_data).view(-1, 4)
#                self.last_conv_size = (conv_w, conv_h)
#        
#        return self.priors
#
#class FPN(ScriptModuleWrapper):
#    """
#    Implements a general version of the FPN introduced in
#    https://arxiv.org/pdf/1612.03144.pdf
#    Parameters (in cfg.fpn):
#        - num_features (int): The number of output features in the fpn layers.
#        - interpolation_mode (str): The mode to pass to F.interpolate.
#        - num_downsample (int): The number of downsampled layers to add onto the selected layers.
#                                These extra layers are downsampled from the last selected layer.
#    Args:
#        - in_channels (list): For each conv layer you supply in the forward pass,
#                              how many features will it have?
#    """
#    __constants__ = ['interpolation_mode', 'num_downsample', 'use_conv_downsample',
#                     'lat_layers', 'pred_layers', 'downsample_layers']
#
#    def __init__(self, in_channels):
#        super().__init__()
#
#        self.lat_layers  = nn.ModuleList([
#            nn.Conv2d(x, cfg.fpn.num_features, kernel_size=1)
#            for x in reversed(in_channels)
#        ])
#
#        self.lat_layers_bn  = nn.ModuleList([
#            nn.BatchNorm2d(cfg.fpn.num_features)
#            for x in reversed(in_channels)
#        ])
#
#        # This is here for backwards compatability
#        padding = 1 if cfg.fpn.pad else 0
#        self.pred_layers = nn.ModuleList([
#            nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=padding)
#            for _ in in_channels
#        ])
#
#        if cfg.fpn.use_conv_downsample:
#            self.downsample_layers = nn.ModuleList([
#                nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=1, stride=2)
#                for _ in range(cfg.fpn.num_downsample)
#            ])
#        
#        self.interpolation_mode  = cfg.fpn.interpolation_mode
#        self.num_downsample      = cfg.fpn.num_downsample
#        self.use_conv_downsample = cfg.fpn.use_conv_downsample
#
#    @script_method_wrapper
#    def forward(self, convouts:List[torch.Tensor]):
#        """
#        Args:
#            - convouts (list): A list of convouts for the corresponding layers in in_channels.
#        Returns:
#            - A list of FPN convouts in the same order as x with extra downsample layers if requested.
#        """
#        out = []
#        x = torch.zeros(1, device=convouts[0].device)
#        for i in range(len(convouts)):
#            out.append(x)
#
#        # For backward compatability, the conv layers are stored in reverse but the input and output is
#        # given in the correct order. Thus, use j=-i-1 for the input and output and i for the conv layers.
#        j = len(convouts)
#        for lat_layer,bn in zip(self.lat_layers, self.lat_layers_bn):
#            j -= 1
#
#            if j < len(convouts) - 1:
#                _, _, h, w = convouts[j].size()
#                x = F.interpolate(x, size=(h, w), mode=self.interpolation_mode, align_corners=False)
#          
#            x = x + F.relu(bn(lat_layer(convouts[j])))
#            out[j] = x
#        
#        # This janky second loop is here because TorchScript.
#        j = len(convouts)
#        for pred_layer in self.pred_layers:
#            j -= 1
#            out[j] = F.relu(pred_layer(out[j]))
#
#        # In the original paper, this takes care of P6
#        if self.use_conv_downsample:
#            for downsample_layer in self.downsample_layers:
#                out.append(downsample_layer(out[-1]))
#        else:
#            for idx in range(self.num_downsample):
#                # Note: this is an untested alternative to out.append(out[-1][:, :, ::2, ::2]). Thanks TorchScript.
#                out.append(nn.functional.max_pool2d(out[-1], 1, stride=2))
#
#        return out

last_conv_size=None
some_set=[

   [cfg.backbone.pred_aspect_ratios[0],
    cfg.backbone.pred_scales[0]],

   [cfg.backbone.pred_aspect_ratios[1],
    cfg.backbone.pred_scales[1]],

   [cfg.backbone.pred_aspect_ratios[2],
    cfg.backbone.pred_scales[2]],

   [cfg.backbone.pred_aspect_ratios[3],
    cfg.backbone.pred_scales[3]],

   [cfg.backbone.pred_aspect_ratios[4],
    cfg.backbone.pred_scales[4]]
]

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
#scale=[24],[48],[96],[192],[384]
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

        self.fpn1=nn.Conv2d(512,256,1,1)
        self.fpn2=nn.Conv2d(1024,256,1,1)
        self.fpn3=nn.Conv2d(2048,256,1,1)

        self.lat_layers_bn1=nn.BatchNorm2d(256)
        self.lat_layers_bn2=nn.BatchNorm2d(256)
        self.lat_layers_bn3=nn.BatchNorm2d(256)

        self.pred_layers1=nn.Conv2d(256,256,3,1,1)
        self.pred_layers1_bn=nn.BatchNorm2d(256)

        self.pred_layers2=nn.Conv2d(256,256,3,1,1)
        self.pred_layers2_bn=nn.BatchNorm2d(256)

        self.pred_layers3=nn.Conv2d(256,256,3,1,1)
        self.pred_layers3_bn=nn.BatchNorm2d(256)
     
        self.downsample_layers1=nn.Conv2d(256,256,3,2,1)
        self.downsample_layers2=nn.Conv2d(256,256,3,2,1)

        self.upfeature=nn.Conv2d(256,256,3,1,1)
        self.upfeature_bn=nn.BatchNorm2d(256)

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
###################################################
#for large resolution feature map                 #
###################################################
        self.bbox_extra=nn.Sequential(
            nn.Conv2d(256,256,1,1,0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
         )
       
        self.conf_extra=nn.Sequential(

            nn.Conv2d(256,256,1,1,0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)

         )
        self.mask_extra=nn.Sequential(
         
            nn.Conv2d(256,256,1,1,0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)

         )
######################################################
#for small resolution feature map                    #
######################################################
        self.bbox_extra_tiny=nn.Sequential(
            nn.Conv2d(256,256,1,1,0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
         )

        self.conf_extra_tiny=nn.Sequential(

            nn.Conv2d(256,256,1,1,0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)

         )
        self.mask_extra_tiny=nn.Sequential(

            nn.Conv2d(256,256,1,1,0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)

         )
        self.bbox_layer_tiny=nn.Conv2d(256,12,3,1,1)
        self.conf_layer_tiny=nn.Conv2d(256,243,3,1,1)
        self.mask_layer_tiny=nn.Conv2d(256,96,3,1,1)
#########################################################
# forward process                                       #
#########################################################

    def forward(self,x):
        out=self.backbone(x)
        x1=out[0]
        x2=out[1]
        x3=out[2]
        x4=out[3]
      
        ######### all of is 256 channels ############## 
        x2=self.fpn1(x2)
        x2=self.lat_layers_bn2(x2)
        x2=F.relu(x2)

        x3=self.fpn2(x3)
        x3=self.lat_layers_bn2(x3)
        x3=F.relu(x3)

        x4=self.fpn3(x4)
        x4=self.lat_layers_bn3(x4)
        x4=F.relu(x4)
        ##############################################
        x2=F.relu(self.pred_layers1_bn(self.pred_layers1(x2)))
        x3=F.relu(self.pred_layers2_bn(self.pred_layers2(x3)))
        x4=F.relu(self.pred_layers3_bn(self.pred_layers3(x4)))
        x5=self.downsample_layers1(x4)
        x6=self.downsample_layers2(x5)

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
          #  if i==0:
          #      bbox_x=self.bbox_extra(x)
          #      conf_x=self.conf_extra(x)
          #      mask_x=self.mask_extra(x)

          #      bbox_x=self.bbox_layer(bbox_x)
          #      conf_x=self.conf_layer(conf_x)
          #      mask_x=self.mask_layer(mask_x)

          #      bbox=bbox_x.permute(0,2,3,1).contiguous().view(x.size(0),-1,4)
          #      conf=conf_x.permute(0,2,3,1).contiguous().view(x.size(0),-1,81)
          #      mask=mask_x.permute(0,2,3,1).contiguous().view(x.size(0),-1,32)
          #      mask=torch.tanh(mask)
          #      feat_h=x.size(2) 
          #      feat_w=x.size(3) 
          #     #priors=make_priors(feat_h,feat_w,some_set[i])            
          #      preds={'loc':bbox,'conf':conf,'mask':mask,'priors':priors_0}
          #  else:
            bbox_x=self.bbox_extra_tiny(x)
            conf_x=self.conf_extra_tiny(x)
            mask_x=self.mask_extra_tiny(x)

            bbox_x=self.bbox_layer_tiny(bbox_x)
            conf_x=self.conf_layer_tiny(conf_x)
            mask_x=self.mask_layer_tiny(mask_x)

            bbox=bbox_x.permute(0,2,3,1).contiguous().view(x.size(0),-1,4)
            conf=conf_x.permute(0,2,3,1).contiguous().view(x.size(0),-1,81)
            mask=mask_x.permute(0,2,3,1).contiguous().view(x.size(0),-1,32)
            mask=torch.tanh(mask)

            feat_h=x.size(2)
            feat_w=x.size(3)
                #priors=make_priors(feat_h,feat_w,some_set[i])
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
        pred_outs['segm']=self.semantic_set_conv(out[0])
        return pred_outs


       
            
      
