"""
DESCRIPTION:    this file contains all the models as well as the methods to call the models

AUTHOR:         Lou Chenfei  Li Xiangjie

INSTITUTE:      Shanghai Jiao Tong University, UM-SJTU Joint Institute

PROJECT:        ECE4730J Advanced Embedded System Capstone Project
"""

from nbformat import current_nbformat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import utils_new as utils
from nikolaos.bof_utils import LogisticConvBoF
from models_new_2 import vgg19_exits_eval_jump_stl10, vgg19_exits_train_jump_stl10, vgg19_exits_eval_jump_mnist, vgg19_exits_train_jump_mnist
# import utils
import global_param as gp

'''
a list of models:
    cifar_exits_train:          cifar with early-exits for training
    cifar_exits_eval:           cifar with early-exits for evaluation
    cifar_normal:               cifar without early-exits
    vgg_exits_train:            vgg with early-exits for training
    vgg_exits_eval:             vgg with early-exits for evaluation
    vgg_normal:                 vgg without early-exits
    vgg19_exits_train:          vgg19 with early-exits for training
    vgg19_exits_eval:           vgg19 with early-exits for evaluation
    vgg19_normal:               vgg19 without early-exits
'''


def get_train_model( args ):
    '''
    get the model according to args.model_name and args.train_mode
    '''
    if args.model_name == 'cifar':
        if args.train_mode == 'normal':
            return cifar_normal()
        elif args.train_mode in ['original', 'exits']:
            return cifar_exits_train()
        else:
            print( f'Error: args.train_mode ({args.train_mode}) is not valid. Should be normal, original or exits' )
            raise NotImplementedError
    elif args.model_name == 'vgg':
        if args.train_mode == 'normal':
            return vgg_normal()
        elif args.train_mode in ['original', 'exits']:
            return vgg_exits_train()
    elif args.model_name == 'vgg19':
        if args.train_mode == 'normal':
            return vgg19_normal()
        elif args.train_mode in ['original', 'exits']:
            if args.jump == 0:
                return vgg19_exits_train()
            else:
                if args.dataset_type == 'mnist':
                    return vgg19_exits_train_jump_mnist()
                elif args.dataset_type == 'stl10':
                    return vgg19_exits_train_jump_stl10()
                else:
                    return vgg19_exits_train_jump()
    elif args.model_name == 'resnet':
        if args.train_mode == 'normal':
            return ResNet34_normal(args)
        elif args.train_mode in ['original', 'exits']:
            return ResNet34_exits_train(args)
    else:
        print( f'Error: args.model_name ({args.model_name}) is not valid. Should be either cifar or vgg or vgg19' )
        raise NotImplementedError


def get_eval_model( args ):
    '''
    get the model according to args.model_name and args.train_mode
    '''
    if args.model_name == 'cifar':
        return cifar_exits_eval() if args.evaluate_mode == 'exits' else cifar_normal()
    elif args.model_name == 'vgg':
        return vgg_exits_eval() if args.evaluate_mode == 'exits' else vgg_normal()
    elif args.model_name == 'vgg19':
        if args.evaluate_mode == 'normal':
            return vgg19_normal()
        elif args.evaluate_mode == 'exits':
            if args.jump == 0:
                return vgg19_exits_eval()
            else:
                if args.dataset_type == 'mnist':
                    return vgg19_exits_eval_jump_mnist()
                elif args.dataset_type == 'stl10':
                    return vgg19_exits_eval_jump_stl10()
                else:
                    return vgg19_exits_eval_jump()
    elif args.model_name == 'resnet':
        return ResNet34_exits_eval(args) if args.evaluate_mode == 'exits' else ResNet34_normal(args)
    else:
        print( f'Error: args.model_name ({args.model_name}) is not valid. Should be either cifar or vgg or vgg19' )
        raise NotImplementedError


class cifar_exits_eval( nn.Module ):
    '''
    has two early-exiting options
    '''
    def __init__(self):
        super(cifar_exits_eval, self).__init__()
        init = gp.cifar_exits_eval_init
        self.exit_num = 2
        self.aggregation = init.aggregation
        # Base network
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1)
        self.fc1 = nn.Linear(5 * 5 * 128, 1024)
        self.fc2 = nn.Linear(1024, 100)
        # Exit layer 1:
        if self.aggregation == 'spatial_bof_1':
            self.exit_1 = LogisticConvBoF(32, 32, split_horizon=6)
            self.exit_1_fc = nn.Linear(4 * 32, 100)
            # Exit layer 2:
            self.exit_2 = LogisticConvBoF(128, 32, split_horizon=2)
        elif self.aggregation == 'spatial_bof_2':
            self.exit_1 = LogisticConvBoF(32, 64, split_horizon=6)
            self.exit_1_fc = nn.Linear(4 * 64, 100)
            # Exit layer 2:
            self.exit_2 = LogisticConvBoF(128, 64, split_horizon=2)
        elif self.aggregation == 'spatial_bof_3':
            self.exit_1 = LogisticConvBoF(32, 256, split_horizon=6)
            self.exit_1_fc = nn.Linear(4 * 256, 100)
            # Exit layer 2:
            self.exit_2 = LogisticConvBoF(128, 256, split_horizon=2)
        elif self.aggregation == 'bof':
            self.exit_1 = LogisticConvBoF(32, 64, split_horizon=14)
            self.exit_1_fc = nn.Linear(64, 100)
            # Exit layer 2:
            self.exit_2 = LogisticConvBoF(128, 64, split_horizon=5)
        # threshold for switching between layers
        self.activation_threshold_1 = 0
        self.activation_threshold_combined = 0
        # the number of early exits
        self.num_early_exit_1 = 0
        self.num_early_exit_3 = 0
        self.original = 0
        # the beta coefficient used for accuracy-speed trade-off, the higher the more accurate
        self.beta = 0
    
    def set_activation_thresholds( self, threshold_list:list ):
        if len( threshold_list ) != self.exit_num:
            print( f'the length of the threshold_list ({len(threshold_list)}) is invalid, should be {self.exit_num}' )
            raise NotImplementedError
        self.activation_threshold_1 = threshold_list[0]
        self.activation_threshold_combined = threshold_list[1]
    
    def set_beta( self, beta ):
        self.beta = beta

    def print_exit_percentage( self ):
        total_inference = self.num_early_exit_1 + self.num_early_exit_3 + self.original
        print( f'early exit 1: {100*self.num_early_exit_1/total_inference:.3f}% ({self.num_early_exit_1}/{total_inference})', end=' | ' )
        print( f'early exit 3: {100*self.num_early_exit_3/total_inference:.3f}% ({self.num_early_exit_3}/{total_inference})', end=' | ' )
        print( f'original: {100*self.original/total_inference:.3f}% ({self.original}/{total_inference})' )
    
    def _calculate_max_activation( self, param ):
        '''
        return the maximum activation item in [param]
        '''
        return torch.max( param )

    def forward( self, x ):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x_exit_1 = self.exit_1(x)
        exit1 = self.exit_1_fc(x_exit_1)
        if self._calculate_max_activation( exit1 ) > self.beta * self.activation_threshold_1:
            self.num_early_exit_1 += 1
            return 0, exit1
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x_exit_2 = self.exit_2(x)
        x_exit_3 = (x_exit_1 + x_exit_2) / 2
        exit3 = self.exit_1_fc(x_exit_3)
        if self._calculate_max_activation( exit3 ) > self.beta * self.activation_threshold_combined:
            self.num_early_exit_3 += 1
            return 1, exit3
        x = x.view(-1, 5 * 5 * 128)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        self.original += 1
        return 2, x


class cifar_exits_train( cifar_exits_eval ):
    '''
    adds functions to specify the exit layers
    should have:
    1. the ability to set_exit_layers
    '''
    def __init__( self ):
        super().__init__()
        self.exit_layer = 'original'
    
    def set_exit_layer(self, exit_layer):
        if exit_layer not in ['original', 'exits']:
            print( f'Error: exit_layer ({exit_layer}) is invalid. Should be original or exits' )
            raise NotImplementedError
        self.exit_layer = exit_layer
    
    # the functions starting from here should be updated by json initializations!
    def forward( self, x ):
        if self.exit_layer == 'original':
            return self.forward_original( x )
        elif self.exit_layer == 'exits':
            return self.forward_exits( x )

    def forward_original( self, x ):
        x = F.relu( self.conv1( x ) )
        x = F.relu( self.conv2( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = F.relu( self.conv3( x ) )
        x = F.relu( self.conv4( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = x.view( -1, 5 * 5 * 128 )
        x = F.relu( self.fc1( x ) )
        x = F.dropout( x, p=0.5, training=self.training )
        x = self.fc2( x )
        return x

    def forward_exits( self, x ):
        x = F.relu( self.conv1( x ) )
        x = F.relu( self.conv2( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x_exit_1 = self.exit_1(x)
        # calculate exit1
        exit1 = self.exit_1_fc(x_exit_1)
        # continue inference
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        # calculate exit3
        x_exit_2 = self.exit_2(x)
        x_exit_3 = (x_exit_1 + x_exit_2) / 2
        exit3 = self.exit_1_fc(x_exit_3)
        return ( exit1, exit3 )


class cifar_normal(nn.Module):
    def __init__(self):
        super(cifar_normal, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1)
        self.fc1 = nn.Linear(5 * 5 * 128, 1024)
        self.fc2 = nn.Linear(1024, 100)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 128)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x


class vgg_exits_eval( nn.Module ):
    '''
    has two early-exiting options
    '''
    def __init__(self):
        super(vgg_exits_eval, self).__init__()
        self.exit_num = 3
        # Base network
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1)         # 32
        self.conv2 = nn.Conv2d(64, 128, 3, 1, padding=1)        # 32
        self.conv3 = nn.Conv2d(128, 256, 3, 1, padding=1)        # 16
        self.conv4 = nn.Conv2d(256, 256, 3, 1, padding=1)        # 16
        self.conv5 = nn.Conv2d(256, 512, 3, 1, padding=1)       # 8
        self.conv6 = nn.Conv2d(512, 512, 3, 1, padding=1) 
        self.conv7 = nn.Conv2d(512, 512, 3, 1, padding=1)  # 8
        self.conv8 = nn.Conv2d(512, 512, 3, 1, padding=1) 
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 100)
        # early exits
        self.exit_1 = LogisticConvBoF(128, 64, split_horizon=8)
        self.exit_2 = LogisticConvBoF(256, 64, split_horizon=4)
        self.exit_3 = LogisticConvBoF(512, 64, split_horizon=2)
        self.exit_1_fc = nn.Linear(64, 100)
        # threshold for switching between layers
        self.activation_threshold_1 = 0
        self.activation_threshold_2 = 0
        self.activation_threshold_3 = 0
        self.num_early_exit_1 = 0
        self.num_early_exit_2 = 0
        self.num_early_exit_3 = 0
        self.original = 0
        # the beta coefficient used for accuracy-speed trade-off, the higher the more accurate
        self.beta = 0
    
    def set_activation_thresholds( self, threshold_list:list ):
        if len( threshold_list ) != self.exit_num:
            print( f'the length of the threshold_list ({len(threshold_list)}) is invalid, should be {self.exit_num}' )
            raise NotImplementedError
        self.activation_threshold_1 = threshold_list[0]
        self.activation_threshold_2 = threshold_list[1]
        self.activation_threshold_3 = threshold_list[2]

    def print_exit_percentage( self ):
        total_inference = self.num_early_exit_1 + self.num_early_exit_2 + self.num_early_exit_3 + self.original
        print( f'early exit 1: {100*self.num_early_exit_1/total_inference:.3f}% ({self.num_early_exit_1}/{total_inference})', end=' | ' )
        print( f'early exit 2: {100*self.num_early_exit_2/total_inference:.3f}% ({self.num_early_exit_2}/{total_inference})', end=' | ' )
        print( f'early exit 3: {100*self.num_early_exit_3/total_inference:.3f}% ({self.num_early_exit_3}/{total_inference})', end=' | ' )
        print( f'original: {100*self.original/total_inference:.3f}% ({self.original}/{total_inference})' )

    def set_beta( self, beta ):
        self.beta = beta
    
    def _calculate_max_activation( self, param ):
        '''
        return the maximum activation item in [param]
        '''
        return torch.max( param )

    def forward( self, x ):
       
        x = F.relu( self.conv1( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = F.relu( self.conv2( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x_exit_1 = self.exit_1(x)
        exit1 = self.exit_1_fc(x_exit_1)
        if self._calculate_max_activation( exit1 ) > self.beta * self.activation_threshold_1:
            self.num_early_exit_1 += 1
            return 0, exit1
        
        x = F.relu( self.conv3( x ) )
        x = F.relu( self.conv4( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x_exit_2 = self.exit_2(x)
        exit2 = (x_exit_1 + x_exit_2) / 2
        exit2 = self.exit_1_fc(exit2)
        if self._calculate_max_activation( exit2 ) > self.beta * self.activation_threshold_2:
            self.num_early_exit_2 += 1
            return 1, exit2

        x = F.relu( self.conv5( x ) )
        x = F.relu( self.conv6( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x_exit_3 = self.exit_3(x)
        exit3 = (x_exit_1 + x_exit_2 + x_exit_3) / 3
        exit3 = self.exit_1_fc(exit3)
        if self._calculate_max_activation( exit3 ) > self.beta * self.activation_threshold_3:
            self.num_early_exit_3 += 1
            return 2, exit3

        x = F.relu( self.conv7( x ) )
        x = F.relu( self.conv8( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = x.view( -1, 512 )
        self.original += 1
        # debug begin
        # print( f"x's shape is {x.shape}" )
        # debug end
        x = F.relu( self.fc1( x ) )
        x = F.dropout( x, p=0.5, training=self.training )
        x = self.fc2( x )
        return 3, x


class vgg_exits_train( vgg_exits_eval ):
    def __init__( self ):
        super().__init__()
        self.exit_layer = 'original'
    
    def set_exit_layer(self, exit_layer):
        if exit_layer not in ['original', 'exits']:
            print( f'Error: exit_layer ({exit_layer}) is invalid. Should be original or exits' )
            raise NotImplementedError
        self.exit_layer = exit_layer
    
    # the functions starting from here should be updated by json initializations!
    def forward( self, x ):
        if self.exit_layer == 'original':
            return self.forward_original( x )
        elif self.exit_layer == 'exits':
            return self.forward_exits( x )

    def forward_original( self, x ):
        x = F.relu( self.conv1( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = F.relu( self.conv2( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = F.relu( self.conv3( x ) )
        x = F.relu( self.conv4( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = F.relu( self.conv5( x ) )
        x = F.relu( self.conv6( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = F.relu( self.conv7( x ) )
        x = F.relu( self.conv8( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = x.view( -1, 512 )
        # debug begin
        # print( f"x's shape is {x.shape}" )
        # debug end
        x = F.relu( self.fc1( x ) )
        x = F.dropout( x, p=0.5, training=self.training )
        x = self.fc2( x )
        return x

    def forward_exits( self, x ):
        x = F.relu( self.conv1( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = F.relu( self.conv2( x ) )
        x = F.max_pool2d( x, 2, 2 )
        # calculate exit1
        x_exit_1 = self.exit_1(x)
        exit1 = self.exit_1_fc(x_exit_1)
        # continue inference
        x = F.relu( self.conv3( x ) )
        x = F.relu( self.conv4( x ) )
        x = F.max_pool2d( x, 2, 2 )
        # calculate exit2
        x_exit_2 = self.exit_2(x)
        exit2 = (x_exit_1 + x_exit_2) / 2
        exit2 = self.exit_1_fc(exit2)
        # continue inference
        x = F.relu( self.conv5( x ) )
        x = F.relu( self.conv6( x ) )
        x = F.max_pool2d( x, 2, 2 )
        # calculate exit3
        x_exit_3 = self.exit_3(x)
        exit3 = (x_exit_1 + x_exit_2 + x_exit_3) / 3
        exit3 = self.exit_1_fc(exit3)
        return ( exit1, exit2, exit3 )
    pass


class vgg_normal( nn.Module ):
    def __init__(self):
        super(vgg_normal, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1)         # 32
        self.conv2 = nn.Conv2d(64, 128, 3, 1, padding=1)        # 32
        self.conv3 = nn.Conv2d(128, 256, 3, 1, padding=1)        # 16
        self.conv4 = nn.Conv2d(256, 256, 3, 1, padding=1)        # 16
        self.conv5 = nn.Conv2d(256, 512, 3, 1, padding=1)       # 8
        self.conv6 = nn.Conv2d(512, 512, 3, 1, padding=1) 
        self.conv7 = nn.Conv2d(512, 512, 3, 1, padding=1)  # 8
        self.conv8 = nn.Conv2d(512, 512, 3, 1, padding=1) 
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 100)
    
    def forward(self, x):
        x = F.relu( self.conv1( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = F.relu( self.conv2( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = F.relu( self.conv3( x ) )
        x = F.relu( self.conv4( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = F.relu( self.conv5( x ) )
        x = F.relu( self.conv6( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = F.relu( self.conv7( x ) )
        x = F.relu( self.conv8( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = x.view( -1, 512 )
        # debug begin
        # print( f"x's shape is {x.shape}" )
        # debug end
        x = F.relu( self.fc1( x ) )
        x = F.dropout( x, p=0.5, training=self.training )
        x = self.fc2( x )
        return x

##############################################################################
###                            Tests on Vgg 19                             ###
##############################################################################
Jump_step = [3,4,5]
data_type = 10
class vgg19_exits_eval_jump( nn.Module ):
    def __init__(self):
        super(vgg19_exits_eval_jump, self).__init__()
        self.exit_num = 20
        # init = gp.vgg_exits_eval_init
        # Base network
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, padding=1)     # maxpool2d
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)   # maxpool2d, exit, 8*8
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, 3, 1, padding=1)   # maxpool2d, exit, 4*4
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 512, 3, 1, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, 1, padding=1)  # maxpool2d, exit, 2*2
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        self.conv14 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn14 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(512, 512, 3, 1, padding=1)  # maxpool2d
        self.bn16 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, data_type)
        # early exits
        self.exit_0 = LogisticConvBoF(64, 64, split_horizon=16)
        self.exit_1 = LogisticConvBoF(128, 64, split_horizon=8)
        self.exit_2 = LogisticConvBoF(256, 64, split_horizon=4)
        self.exit_3 = LogisticConvBoF(512, 64, split_horizon=2)
        self.exit_0_fc = nn.Linear(256,64)
        self.exit_1_fc = nn.Linear(64, data_type)
        # threshold for switching between layers
        self.activation_threshold_list = []
        # the number of early exits
        '''
        self.num_early_exit_1 = 0
        self.num_early_exit_2 = 0
        self.num_early_exit_3 = 0
        '''
        self.num_early_exit_list = [0]*self.exit_num
        self.original = 0

        # the parameter for test on determining jump step

        self.statics = [([0] * 16) for i in range(11)]
        self.total = [0] * 11

        # the beta coefficient used for accuracy-speed trade-off, the higher the more accurate
        self.beta = 0
        self.target_layer = 6

        self.ratio_store = []
        self.jumpstep_store = []
        self.entropy_store = []
        self.temp_ratio_list = []
    
    def set_activation_thresholds( self, threshold_list:list ):
        if len( threshold_list ) != self.exit_num:
            print( f'the length of the threshold_list ({len(threshold_list)}) is invalid, should be {self.exit_num}' )
            raise NotImplementedError
        for i in range(len( threshold_list )):
            self.activation_threshold_list.append(abs( threshold_list[i] ))
        '''
        self.activation_threshold_1 = abs( threshold_list[0] )
        self.activation_threshold_2 = abs( threshold_list[1] )
        self.activation_threshold_3 = abs( threshold_list[2] )
        '''

    def print_exit_percentage( self ):
        total_inference = sum(self.num_early_exit_list)+ self.original
        for i in range( self.exit_num ):
            print('Early Exit', i+1, ': ', "{:.2f}".format(100 * self.num_early_exit_list[i] / total_inference))
        '''s
        print( f'early exit 1: {100*self.num_early_exit_1/total_inference:.3f}% ({self.num_early_exit_1}/{total_inference})', end=' | ' )
        print( f'early exit 2: {100*self.num_early_exit_2/total_inference:.3f}% ({self.num_early_exit_2}/{total_inference})', end=' | ' )
        print( f'early exit 3: {100*self.num_early_exit_3/total_inference:.3f}% ({self.num_early_exit_3}/{total_inference})', end=' | ' )
        '''
        print( f'original: {100*self.original/total_inference:.3f}% ({self.original}/{total_inference})' )
    
    def print_statics( self ):
        sum1 = sum(self.total)
        if sum1 != 0:
            for i in range(len(self.statics)-1,0,-1):
                print(i, ": ", self.total[i]/sum1, end = '\n')
                for j in range(len(self.statics[i])):
                    if self.total[i]!= 0:
                        print("{:.2f}".format(100*self.statics[i][j]/self.total[i]), end='\t')
                print()
        else: print("The sum is 0!")
        
    def set_beta( self, beta ):
        self.beta = beta
    
    def _calculate_max_activation( self, param ):
        '''
        return the maximum activation item in [param]
        '''
        return torch.max( torch.abs( torch.max( param ) ), torch.abs( torch.min( param ) ) )

    
    def _calculate_cross_entropy(self, param):

        A = torch.clone(param)
        temp = torch.min(A)
        if temp.item() < 0:
            for i in range(data_type):
                A[0][i] += torch.abs(temp)

        temp_sum = torch.sum(A[0])
        for i in range(data_type):
            A[0][i] /= temp_sum
        
        a = 0
        for i in range(data_type):
            if A[0][i]!= 0:
                a += -A[0][i]*torch.log2(A[0][i])

        return a 
    '''
    def _calculate_cross_entropy(self, param):
        A = F.softmax(param, dim = 1)
        a = -((A * torch.log2(A)).sum())
        return a
    '''

    def output( self ):
        return self.ratio_store, self.jumpstep_store, self.entropy_store, self.temp_ratio_list
      #  return self.ratio_store, self.jumpstep_store
    
    def forward( self, x ):
        flag = 0
        temp_ratio = 0
        temp_entropy = 0
        temp_exit_layer = 0
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
     

        x = F.max_pool2d(x, 2, 2)
        current_layer = 6
        if self.target_layer == 6:
            x_exit_1 = self.exit_1(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            temp_ratio = ratio
            temp_exit_layer = exit1
            flag = int(ratio/0.1)
            if ratio >= 1:
              #  self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[5] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 5, exit1
            self.ratio_store.append(ratio.item())
            self.entropy_store.append(self._calculate_cross_entropy(exit1).item())
            temp_entropy = self._calculate_cross_entropy(exit1).item()
            self.total[flag] = self.total[flag] + 1

        x = F.relu(self.bn5(self.conv5(x)))
        current_layer = 7
        x_exit_1 = self.exit_2(x)
        x_exit_1 = self.exit_0_fc(x_exit_1)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[6])
        if ratio >= 1:
            if temp_entropy > 2.8 and temp_entropy < 3.2:
                self.temp_ratio_list.append(temp_ratio)
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]= self.statics[flag][current_layer-self.target_layer] + 1
            self.num_early_exit_list[6] += 1
            return 6, exit1

        x = F.relu(self.bn6(self.conv6(x)))
        current_layer = 8
        x_exit_1 = self.exit_2(x)
        x_exit_1 = self.exit_0_fc(x_exit_1)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[7])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]= self.statics[flag][current_layer-self.target_layer] + 1
            self.num_early_exit_list[7] += 1
            return 7, exit1

        x = F.relu(self.bn7(self.conv7(x)))
        current_layer += 1
        x_exit_1 = self.exit_2(x)
        x_exit_1 = self.exit_0_fc(x_exit_1)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[8])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]+= 1
            self.num_early_exit_list[8] += 1
            return 8, exit1
   
        
        x = F.relu(self.bn8(self.conv8(x)))
        current_layer += 1
        x_exit_1 = self.exit_2(x)
        x_exit_1 = self.exit_0_fc(x_exit_1)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[9])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]+= 1
            self.num_early_exit_list[9] += 1
            return 9, exit1
       
        x = F.max_pool2d(x, 2, 2)
        current_layer += 1
        x_exit_1 = self.exit_2(x)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]+= 1
            self.num_early_exit_list[10] += 1
            return 10, exit1
        
        
        x = F.relu(self.bn9(self.conv9(x)))
        current_layer += 1
        x_exit_1 = self.exit_3(x)
        x_exit_1 = self.exit_0_fc(x_exit_1)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[11])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]+= 1
            self.num_early_exit_list[11] += 1
            return 11, exit1
        

        x = F.relu(self.bn10(self.conv10(x)))
        current_layer += 1
        x_exit_1 = self.exit_3(x)
        x_exit_1 = self.exit_0_fc(x_exit_1)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[12])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]+= 1
            self.num_early_exit_list[12] += 1
            return 12, exit1
           
             
        x = F.relu(self.bn11(self.conv11(x)))
        current_layer += 1
        x_exit_1 = self.exit_3(x)
        x_exit_1 = self.exit_0_fc(x_exit_1)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[13])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]+= 1
            self.num_early_exit_list[13] += 1
            return 13, exit1
        

        x = F.relu(self.bn12(self.conv12(x)))
        current_layer += 1
        x_exit_1 = self.exit_3(x)
        x_exit_1 = self.exit_0_fc(x_exit_1)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[14])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]+= 1
            self.num_early_exit_list[14] += 1
            return 14, exit1
        

        x = F.max_pool2d(x, 2, 2)
        current_layer += 1
        x_exit_1 = self.exit_3(x)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[15])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]+= 1
            self.num_early_exit_list[15] += 1
            return 15, exit1
    

        x = F.relu(self.bn13(self.conv13(x)))
        current_layer += 1
        x_exit_1 = self.exit_3(x)
         #   x_exit_1 = self.exit_0_fc(x_exit_1)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[16])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]+= 1
            self.num_early_exit_list[16] += 1
            return 16, exit1

        x = F.relu(self.bn14(self.conv14(x)))
        current_layer += 1
        x_exit_1 = self.exit_3(x)
         #   x_exit_1 = self.exit_0_fc(x_exit_1)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[17])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]+= 1
            self.num_early_exit_list[17] += 1
            return 17, exit1
            

        x = F.relu(self.bn15(self.conv15(x)))
        current_layer += 1
        x_exit_1 = self.exit_3(x)
       #     x_exit_1 = self.exit_0_fc(x_exit_1)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[18])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]+= 1
            self.num_early_exit_list[18] += 1
            return 18, exit1
            
        
        x = F.relu(self.bn16(self.conv16(x)))
        current_layer += 1
        x_exit_1 = self.exit_3(x)
       #     x_exit_1 = self.exit_0_fc(x_exit_1)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[19])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]+= 1
            self.num_early_exit_list[19] += 1
            return 19, exit1
    
        x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.view(-1, 512)
        x = self.fc(x)
        self.original += 1
        current_layer += 1
        self.jumpstep_store.append(current_layer-self.target_layer)
        self.statics[flag][current_layer-self.target_layer]+= 1
        return 20, x
    '''
    def forward( self, x ):
        x = F.relu(self.bn1(self.conv1(x)))
        if self.target_layer == 1:
            x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / self.beta * self.activation_threshold_list[0]
            if ratio >= 1:
                self.num_early_exit_list[0] += 1
                return 0, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 

        x = F.relu(self.bn2(self.conv2(x)))
        if self.target_layer == 2:
            x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[1])
            if ratio >= 1:
                self.num_early_exit_list[1] += 1
                return 1, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2]
            
        x = F.max_pool2d(x, 2, 2)
        if self.target_layer == 3:
            x_exit_1 = self.exit_0(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[2])
            if ratio >= 1:
                self.num_early_exit_list[2] += 1
                return 2, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 

        x = F.relu(self.bn3(self.conv3(x)))
        if self.target_layer == 4:
            x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[3])
            if ratio >= 1:
                self.num_early_exit_list[3] += 1
                return 3, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 
            
        x = F.relu(self.bn4(self.conv4(x)))
        if self.target_layer == 5:
            x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[4])
            if ratio >= 1:
                self.num_early_exit_list[4] += 1
                return 4, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 

        x = F.max_pool2d(x, 2, 2)
        if self.target_layer == 6:
            x_exit_1 = self.exit_1(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            if ratio >= 1:
                self.num_early_exit_list[5] += 1
                return 5, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 

        x = F.relu(self.bn5(self.conv5(x)))
        if self.target_layer == 7:
            x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[6])
            if ratio >= 1:
                self.num_early_exit_list[6] += 1
                return 6, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 

        x = F.relu(self.bn6(self.conv6(x)))
        if self.target_layer == 8:
            x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[7])
            if ratio >= 1:
                self.num_early_exit_list[7] += 1
                return 7, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 

        x = F.relu(self.bn7(self.conv7(x)))
        if self.target_layer == 9:
            x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[8])
            if ratio >= 1:
                self.num_early_exit_list[8] += 1
                return 8, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 
        
        x = F.relu(self.bn8(self.conv8(x)))
        if self.target_layer == 10:
            x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[9])
            if ratio >= 1:
                self.num_early_exit_list[9] += 1
                return 9, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 

        x = F.max_pool2d(x, 2, 2)
        if self.target_layer == 11:
            x_exit_1 = self.exit_2(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
            if ratio >= 1:
                self.num_early_exit_list[10] += 1
                return 10, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 
        
        x = F.relu(self.bn9(self.conv9(x)))
        if self.target_layer == 12:
            x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[11])
            if ratio >= 1:
                self.num_early_exit_list[11] += 1
                return 11, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num 

        x = F.relu(self.bn10(self.conv10(x)))
        if self.target_layer == 13:
            x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[12])
            if ratio >= 1:
                self.num_early_exit_list[12] += 1
                return 12, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
             
        x = F.relu(self.bn11(self.conv11(x)))
        if self.target_layer == 14:
            x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[13])
            if ratio >= 1:
                self.num_early_exit_list[13] += 1
                return 13, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn12(self.conv12(x)))
        if self.target_layer == 15:
            x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[14])
            if ratio >= 1:
                self.num_early_exit_list[14] += 1
                return 14, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.max_pool2d(x, 2, 2)
        if self.target_layer == 16:
            x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[15])
            if ratio >= 1:
                self.num_early_exit_list[15] += 1
                return 15, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn13(self.conv13(x)))
        if self.target_layer == 17:
            x_exit_1 = self.exit_3(x)
         #   x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[16])
            if ratio >= 1:
                self.num_early_exit_list[16] += 1
                return 16, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn14(self.conv14(x)))
        if self.target_layer == 18:
            x_exit_1 = self.exit_3(x)
         #   x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[17])
            if ratio >= 1:
                self.num_early_exit_list[17] += 1
                return 17, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn15(self.conv15(x)))
        if self.target_layer == 19:
            x_exit_1 = self.exit_3(x)
       #     x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[18])
            if ratio >= 1:
                self.num_early_exit_list[18] += 1
                return 18, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
        
        x = F.relu(self.bn16(self.conv16(x)))
        if self.target_layer == 20:
            x_exit_1 = self.exit_3(x)
       #     x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[19])
            if ratio >= 1:
                self.num_early_exit_list[19] += 1
                return 19, exit1
    
        x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.view(-1, 512)
        x = self.fc(x)
        self.original += 1
        return 3, x
    '''

class vgg19_exits_train_jump( vgg19_exits_eval_jump ):
    def __init__( self ):
        super().__init__()
        self.exit_layer = 'original'
    
    def set_exit_layer(self, exit_layer):
        if exit_layer not in ['original', 'exits']:
            print( f'Error: exit_layer ({exit_layer}) is invalid. Should be original or exits' )
            raise NotImplementedError
        self.exit_layer = exit_layer
    
    # the functions starting from here should be updated by json initializations!
    def forward( self, x ):
        if self.exit_layer == 'original':
            return self.forward_original( x )
        elif self.exit_layer == 'exits':
            return self.forward_exits( x )

    def forward_original( self, x ):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn13(self.conv13(x)))
        x = F.relu(self.bn14(self.conv14(x)))
        x = F.relu(self.bn15(self.conv15(x)))
        x = F.relu(self.bn16(self.conv16(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x

    def forward_exits( self, x ):
        x = F.relu(self.bn1(self.conv1(x)))
        x_exit_1 = self.exit_0(x)
        x_exit_1 = self.exit_0_fc(x_exit_1)
        exit1 = self.exit_1_fc(x_exit_1)

        x = F.relu(self.bn2(self.conv2(x)))
        x_exit_2 = self.exit_0(x)
        x_exit_2 = self.exit_0_fc(x_exit_2)
        exit2 = self.exit_1_fc(x_exit_2)

        x = F.max_pool2d(x, 2, 2)
        x_exit_3 = self.exit_0(x)
        exit3 = self.exit_1_fc(x_exit_3)

        x = F.relu(self.bn3(self.conv3(x)))
        x_exit_4 = self.exit_1(x)
        x_exit_4 = self.exit_0_fc(x_exit_4)
        exit4 = self.exit_1_fc(x_exit_4)

        x = F.relu(self.bn4(self.conv4(x)))
        x_exit_5 = self.exit_1(x)
        x_exit_5 = self.exit_0_fc(x_exit_5)
        exit5 = self.exit_1_fc(x_exit_5)

        x = F.max_pool2d(x, 2, 2)
        x_exit_6 = self.exit_1(x)
        exit6 = self.exit_1_fc(x_exit_6)

        x = F.relu(self.bn5(self.conv5(x)))
        x_exit_7 = self.exit_2(x)
        x_exit_7 = self.exit_0_fc(x_exit_7)
        exit7 = self.exit_1_fc(x_exit_7)

        x = F.relu(self.bn6(self.conv6(x)))
        x_exit_8 = self.exit_2(x)
        x_exit_8 = self.exit_0_fc(x_exit_8)
        exit8 = self.exit_1_fc(x_exit_8)

        x = F.relu(self.bn7(self.conv7(x)))
        x_exit_9 = self.exit_2(x)
        x_exit_9 = self.exit_0_fc(x_exit_9)
        exit9 = self.exit_1_fc(x_exit_9)

        x = F.relu(self.bn8(self.conv8(x)))
        x_exit_10 = self.exit_2(x)
        x_exit_10 = self.exit_0_fc(x_exit_10)
        exit10 = self.exit_1_fc(x_exit_10)

        x = F.max_pool2d(x, 2, 2)
        x_exit_11 = self.exit_2(x)
     #   exit2 = (x_exit_1 + x_exit_2) / 2
        exit11 = self.exit_1_fc(x_exit_11)

        x = F.relu(self.bn9(self.conv9(x)))
        x_exit_12 = self.exit_3(x)
        x_exit_12 = self.exit_0_fc(x_exit_12)
        exit12 = self.exit_1_fc(x_exit_12)

        x = F.relu(self.bn10(self.conv10(x)))
        x_exit_13 = self.exit_3(x)
        x_exit_13 = self.exit_0_fc(x_exit_13)
        exit13 = self.exit_1_fc(x_exit_13)

        x = F.relu(self.bn11(self.conv11(x)))
        x_exit_14 = self.exit_3(x)
        x_exit_14 = self.exit_0_fc(x_exit_14)
        exit14 = self.exit_1_fc(x_exit_14)

        x = F.relu(self.bn12(self.conv12(x)))
        x_exit_15 = self.exit_3(x)
        x_exit_15 = self.exit_0_fc(x_exit_15)
        exit15 = self.exit_1_fc(x_exit_15)

        x = F.max_pool2d(x, 2, 2)
        x_exit_16 = self.exit_3(x)
    #    exit3 = (x_exit_1 + x_exit_2 + x_exit_3) / 3
        exit16 = self.exit_1_fc(x_exit_16)

        x = F.relu(self.bn13(self.conv13(x)))
        x_exit_17 = self.exit_3(x)
        exit17 = self.exit_1_fc(x_exit_17)

        x = F.relu(self.bn14(self.conv14(x)))
        x_exit_18 = self.exit_3(x)
        exit18 = self.exit_1_fc(x_exit_18)

        x = F.relu(self.bn15(self.conv15(x)))
        x_exit_19 = self.exit_3(x)
        exit19 = self.exit_1_fc(x_exit_19)

        x = F.relu(self.bn16(self.conv16(x)))
        x_exit_20 = self.exit_3(x)
        exit20 = self.exit_1_fc(x_exit_20)

        return (exit1, exit2, exit3, exit4, exit5, exit6, exit7, exit8, exit9, exit10, exit11, exit12, exit13, exit14, exit15, exit16, exit17, exit18, exit19,exit20)



##############################################################################
###                           Vgg 19 Normal Design                         ###
##############################################################################

class vgg19_exits_eval( nn.Module ):
    def __init__(self):
        super(vgg19_exits_eval, self).__init__()
        self.exit_num = 3
        # init = gp.vgg_exits_eval_init
        # Base network
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, padding=1)     # maxpool2d
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)   # maxpool2d, exit, 8*8
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, 3, 1, padding=1)   # maxpool2d, exit, 4*4
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 512, 3, 1, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, 1, padding=1)  # maxpool2d, exit, 2*2
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        self.conv14 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn14 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(512, 512, 3, 1, padding=1)  # maxpool2d
        self.bn16 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, data_type)
        # early exits
        self.exit_1 = LogisticConvBoF(128, 64, split_horizon=8)
        self.exit_2 = LogisticConvBoF(256, 64, split_horizon=4)
        self.exit_3 = LogisticConvBoF(512, 64, split_horizon=2)
        self.exit_1_fc = nn.Linear(64, data_type)
        # threshold for switching between layers
        self.activation_threshold_1 = 0
        self.activation_threshold_2 = 0
        self.activation_threshold_3 = 0
        # the number of early exits
        self.num_early_exit_1 = 0
        self.num_early_exit_2 = 0
        self.num_early_exit_3 = 0
        self.original = 0
        # the beta coefficient used for accuracy-speed trade-off, the higher the more accurate
        self.beta = 0
    
    def set_activation_thresholds( self, threshold_list:list ):
        if len( threshold_list ) != self.exit_num:
            print( f'the length of the threshold_list ({len(threshold_list)}) is invalid, should be {self.exit_num}' )
            raise NotImplementedError
        self.activation_threshold_1 = abs( threshold_list[0] )
        self.activation_threshold_2 = abs( threshold_list[1] )
        self.activation_threshold_3 = abs( threshold_list[2] )

    def print_exit_percentage( self ):
        total_inference = self.num_early_exit_1 + self.num_early_exit_2 + self.num_early_exit_3 + self.original
        print( f'early exit 1: {100*self.num_early_exit_1/total_inference:.3f}% ({self.num_early_exit_1}/{total_inference})', end=' | ' )
        print( f'early exit 2: {100*self.num_early_exit_2/total_inference:.3f}% ({self.num_early_exit_2}/{total_inference})', end=' | ' )
        print( f'early exit 3: {100*self.num_early_exit_3/total_inference:.3f}% ({self.num_early_exit_3}/{total_inference})', end=' | ' )
        print( f'original: {100*self.original/total_inference:.3f}% ({self.original}/{total_inference})' )

    def set_beta( self, beta ):
        self.beta = beta
    
    def _calculate_max_activation( self, param ):
        
        return torch.max( torch.abs( torch.max( param ) ), torch.abs( torch.min( param ) ) )

    def forward( self, x ):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2, 2)
        x_exit_1 = self.exit_1(x)
        exit1 = self.exit_1_fc(x_exit_1)
        if self._calculate_max_activation( exit1 ) > self.beta * self.activation_threshold_1:
            self.num_early_exit_1 += 1
            return 0, exit1
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.max_pool2d(x, 2, 2)
        x_exit_2 = self.exit_2(x)
    #    exit2 = (x_exit_1 + x_exit_2) / 2
        exit2 = self.exit_1_fc(x_exit_2)
        if self._calculate_max_activation( exit2 ) > self.beta * self.activation_threshold_2:
            self.num_early_exit_2 += 1
            return 1, exit2
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.max_pool2d(x, 2, 2)
        x_exit_3 = self.exit_3(x)
    #    exit3 = (x_exit_1 + x_exit_2 + x_exit_3) / 3
        exit3 = self.exit_1_fc(x_exit_3)
        if self._calculate_max_activation( exit3 ) > self.beta * self.activation_threshold_3:
            self.num_early_exit_3 += 1
            return 2, exit3
        x = F.relu(self.bn13(self.conv13(x)))
        x = F.relu(self.bn14(self.conv14(x)))
        x = F.relu(self.bn15(self.conv15(x)))
        x = F.relu(self.bn16(self.conv16(x)))

        x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.view(-1, 512)
        x = self.fc(x)
        self.original += 1
        return 3, x


class vgg19_exits_train( vgg19_exits_eval ):
    def __init__( self ):
        super().__init__()
        self.exit_layer = 'original'
    
    def set_exit_layer(self, exit_layer):
        if exit_layer not in ['original', 'exits']:
            print( f'Error: exit_layer ({exit_layer}) is invalid. Should be original or exits' )
            raise NotImplementedError
        self.exit_layer = exit_layer
    
    # the functions starting from here should be updated by json initializations!
    def forward( self, x ):
        if self.exit_layer == 'original':
            return self.forward_original( x )
        elif self.exit_layer == 'exits':
            return self.forward_exits( x )

    def forward_original( self, x ):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn13(self.conv13(x)))
        x = F.relu(self.bn14(self.conv14(x)))
        x = F.relu(self.bn15(self.conv15(x)))
        x = F.relu(self.bn16(self.conv16(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x

    def forward_exits( self, x ):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2, 2)
        x_exit_1 = self.exit_1(x)
        exit1 = self.exit_1_fc(x_exit_1)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.max_pool2d(x, 2, 2)
        x_exit_2 = self.exit_2(x)
     #   exit2 = (x_exit_1 + x_exit_2) / 2
        exit2 = self.exit_1_fc(x_exit_2)
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.max_pool2d(x, 2, 2)
        x_exit_3 = self.exit_3(x)
    #    exit3 = (x_exit_1 + x_exit_2 + x_exit_3) / 3
        exit3 = self.exit_1_fc(x_exit_3)
        return (exit1, exit2, exit3)


class vgg19_normal( nn.Module ):
    def __init__(self):
        super(vgg19_normal, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, padding=1)     # maxpool2d
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)   # maxpool2d
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, 3, 1, padding=1)   # maxpool2d
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 512, 3, 1, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, 1, padding=1)  # maxpool2d
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        self.conv14 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn14 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(512, 512, 3, 1, padding=1)  # maxpool2d
        self.bn16 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, data_type)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn13(self.conv13(x)))
        x = F.relu(self.bn14(self.conv14(x)))
        x = F.relu(self.bn15(self.conv15(x)))
        x = F.relu(self.bn16(self.conv16(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet_exits_eval(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_exits_eval, self).__init__()
        self.in_planes = 64
        self.exit_num = 3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        # early exits
        self.exit_1 = LogisticConvBoF(input_features=256, n_codewords=128, split_horizon=32)
        self.exit_2 = LogisticConvBoF(input_features=512, n_codewords=128, split_horizon=16)
        self.exit_3 = LogisticConvBoF(input_features=1024, n_codewords=128, split_horizon=8)
        self.exit_1_fc = nn.Linear(128, num_classes)
        # threshold for switching between layers
        self.activation_threshold_1 = 0
        self.activation_threshold_2 = 0
        self.activation_threshold_3 = 0
        self.num_early_exit_1 = 0
        self.num_early_exit_2 = 0
        self.num_early_exit_3 = 0
        self.original = 0
        # the beta coefficient used for accuracy-speed trade-off, the higher the more accurate
        self.beta = 0

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def set_activation_thresholds( self, threshold_list:list ):
        if len( threshold_list ) != self.exit_num:
            print( f'the length of the threshold_list ({len(threshold_list)}) is invalid, should be {self.exit_num}' )
            raise NotImplementedError
        self.activation_threshold_1 = threshold_list[0]
        self.activation_threshold_2 = threshold_list[1]
        self.activation_threshold_3 = threshold_list[2]

    def print_exit_percentage( self ):
        total_inference = self.num_early_exit_1 + self.num_early_exit_2 + self.num_early_exit_3 + self.original
        print( f'early exit 1: {100*self.num_early_exit_1/total_inference:.3f}% ({self.num_early_exit_1}/{total_inference})', end=' | ' )
        print( f'early exit 2: {100*self.num_early_exit_2/total_inference:.3f}% ({self.num_early_exit_2}/{total_inference})', end=' | ' )
        print( f'early exit 3: {100*self.num_early_exit_3/total_inference:.3f}% ({self.num_early_exit_3}/{total_inference})', end=' | ' )
        print( f'original: {100*self.original/total_inference:.3f}% ({self.original}/{total_inference})' )

    def set_beta( self, beta ):
        self.beta = beta
    
    def _calculate_max_activation( self, param ):
        '''
        return the maximum activation item in [param]
        '''
        return torch.max( torch.abs( torch.max( param ) ), torch.abs( torch.min( param ) ) )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        # Early Exit Part
        
        out_exit_1 = self.exit_1(out)
        exit1 = self.exit_1_fc(out_exit_1)
        if self._calculate_max_activation( exit1 ) > self.beta * self.activation_threshold_1:
            self.num_early_exit_1 += 1
            return 0, exit1
        # Normal NetWork
        out = self.layer2(out)
        # Early Exit Part
        out_exit_2 = self.exit_2(out)
        exit2 = (out_exit_1 + out_exit_2) / 2
        exit2 = self.exit_1_fc(exit2)
        if self._calculate_max_activation( exit2 ) > self.beta * self.activation_threshold_2:
            self.num_early_exit_2 += 1
            return 1, exit2
        # Normal NetWork
        out = self.layer3(out)
        # Early Exit Part
        out_exit_3 = self.exit_3(out)
        exit3 = (out_exit_1 + out_exit_2 + out_exit_3) / 3
        exit3 = self.exit_1_fc(exit3)
        if self._calculate_max_activation( exit3 ) > self.beta * self.activation_threshold_3:
            self.num_early_exit_3 += 1
            return 2, exit3
        # Normal NetWork
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        self.original += 1
        return 3, out



class ResNet_exits_train(ResNet_exits_eval):
    def __init__(self, block, num_blocks, num_classes):
        super().__init__(block, num_blocks, num_classes)
        self.exit_layer = 'original'
    
    def set_exit_layer(self, exit_layer):
        if exit_layer not in ['original', 'exits']:
            print( f'Error: exit_layer ({exit_layer}) is invalid. Should be original or exits' )
            raise NotImplementedError
        self.exit_layer = exit_layer
    
    # the functions starting from here should be updated by json initializations!
    def forward( self, x ):
        if self.exit_layer == 'original':
            return self.forward_original( x )
        elif self.exit_layer == 'exits':
            return self.forward_exits( x )
    
    def forward_original( self, x ):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def forward_exits(self,x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out_exit_1 = self.exit_1(out)
        exit1 = self.exit_1_fc(out_exit_1)

        out = self.layer2(out)
        out_exit_2 = self.exit_2(out)
        exit2 = (out_exit_1 + out_exit_2) / 2
        exit2 = self.exit_1_fc(exit2)

        out = self.layer3(out)
        out_exit_3 = self.exit_3(out)
        exit3 = (out_exit_1 + out_exit_2 + out_exit_3) / 3
        exit3 = self.exit_1_fc(exit3)

        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return (exit1, exit2, exit3)


def ResNet18_normal(args):
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34_normal(args):
    if args.dataset_type == 'cifar10':
        return ResNet(BasicBlock, [3,4,6,3], num_classes=10)
    else:
        return ResNet(BasicBlock, [3,4,6,3], num_classes=100)

def ResNet50_normal(args):
    if args.dataset_type == 'cifar10':
        return ResNet(Bottleneck, [3,4,6,3], num_classes=10)
    else:
        return ResNet(Bottleneck, [3,4,6,3], num_classes=100)

def ResNet101_normal(args):
    if args.dataset_type == 'cifar10':
        return ResNet(Bottleneck, [3,4,23,3], num_classes=10)
    else:
        return ResNet(Bottleneck, [3,4,23,3], num_classes=100)

def ResNet152_normal(args):
    if args.dataset_type == 'cifar10':
        return ResNet(Bottleneck, [3,8,36,3], num_classes=10)
    else:
        return ResNet(Bottleneck, [3,8,36,3], num_classes=100)


def ResNet34_exits_train(args):
    if args.dataset_type == 'cifar10':
        return ResNet_exits_train(Bottleneck, [3,8,36,3], num_classes=10)
    else:
        return ResNet_exits_train(Bottleneck, [3,8,36,3], num_classes=100)


def ResNet34_exits_eval(args):
    if args.dataset_type == 'cifar10':
        return ResNet_exits_eval(Bottleneck, [3,8,36,3], num_classes=10)
    else :
        return ResNet_exits_eval(Bottleneck, [3,8,36,3], num_classes=100)





if __name__ == '__main__':
    args = utils.Namespace( model_name='vgg19',
                            pretrained_file='autodl-tmp/vgg19_train_exits_cifar10.pt',
                            optimizer='adam',
                            train_mode='exits',
                            evaluate_mode='exits',
                            task='evaluate',
                            device='cuda',
                            trained_file_suffix='update_1',
                            beta=9,
                            save=0,
                            dataset_type = 'cifar10',
                            jump = 1
                            )
    model = ResNet34_exits_train(args)

    hyper = gp.get_hyper( args )
    model = torch.load( args.pretrained_file, map_location='cpu' ).to( args.device )
    
    model.set_exit_layer( 'exits' )
    optimizer = gp.get_optimizer( params=model.parameters(), lr=hyper.learning_rate, op_type=args.optimizer )
    train_loader = gp.get_dataloader( args, task='train' )
    model.train()
    for epoch_idx in range( hyper.epoch_num ):
        print(f'\nEpoch: {(epoch_idx + 1)}')
        train_loss = 0
        correct_exit = None
        total = 0
        for batch_idx, ( images, labels ) in enumerate( train_loader ):
            images, labels = images.to( args.device ), labels.to( args.device )
            optimizer.zero_grad()
            exit_tuple = model( images )
            loss = 0
            for exit_idx in range( len( exit_tuple ) ):
                loss += gp.criterion( exit_tuple[exit_idx], labels )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            exit_predicted_list = [None for i in range( len( exit_tuple ) )]
            for exit_idx in range( len( exit_tuple ) ):
                _, exit_predicted_list[exit_idx] = exit_tuple[exit_idx].max( 1 )
            total += labels.size( 0 )
            if correct_exit is None:
                correct_exit = [0 for i in range( len( exit_tuple ) )]
            for exit_idx in range( len( exit_tuple ) ):
                correct_exit[exit_idx] += exit_predicted_list[exit_idx].eq( labels ).sum().item()
            if batch_idx % 100 == 99:    # print every 100 mini-batches
                print( '[%d, %5d] loss: %.5f |' % (epoch_idx + 1, batch_idx + 1, train_loss / 2000), end='' )
                for exit_idx in range( len( exit_tuple ) ):
                    print( '| exit'+str(exit_idx)+': %.3f%% (%d/%d)' % (100.*correct_exit[exit_idx]/total, correct_exit[exit_idx], total), end='' )
                print( '' )
                # model.print_average_activations()
                train_loss, total = 0, 0
                correct_exit = None
        if epoch_idx % 5 == 4:
            print( f'begin middle test:' )