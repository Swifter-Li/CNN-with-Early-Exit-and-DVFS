"""
DESCRIPTION:    this file contains all the models as well as the methods to call the models

"""

from nbformat import current_nbformat
from pytz import country_timezones
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import utils_new as utils
from nikolaos.bof_utils import LogisticConvBoF
import time
from models_2 import vgg19_exits_eval_jump_stl10, vgg19_exits_train_jump_stl10, vgg19_exits_eval_jump_mnist, vgg19_exits_train_jump_mnist, \
    vgg19_exits_eval_jump_svhn, vgg19_exits_train_jump_svhn
from models_3 import ResNet_exits_eval, ResNet_exits_train

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
                elif args.dataset_type == 'svhn':
                    return vgg19_exits_train_jump_svhn()
                else:
                    return vgg19_exits_train_jump()
    elif args.model_name == 'resnet':
        if args.dataset_type == 'cifar100':
            return ResNet_exits_train(100)
        else: 
            return ResNet_exits_train(10)
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
                elif args.dataset_type == 'svhn':
                    return vgg19_exits_eval_jump_svhn()
                else:
                    return vgg19_exits_eval_jump()
    elif args.model_name == 'resnet':
        if args.dataset_type == 'cifar100':
            return ResNet_exits_eval(100)
        else: 
            return ResNet_exits_eval(10)
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

        self.statics = [([0] * 20) for i in range(11)]
        self.total = [0] * 11

        # the beta coefficient used for accuracy-speed trade-off, the higher the more accurate
        self.beta = 0
        self.target_layer = 6
        self.start_layer = 6
        
        # The index of 3 stands for the computational cost
        self.count_store = [0]*4

        self.layer_store = [0]*20
        self.jumpstep_store = []
        self.prediction_store = []

        # The test forward normal to choose 'normal_forward', 'accuracy_forward', 'quant_forward'
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.forward_mode = 'normal_forward'

        self.quant_switch = 0
        # Test on Train parameter
        self.possible_layer = []
    
    def set_activation_thresholds( self, threshold_list:list ):
        if len( threshold_list ) != self.exit_num:
            print( f'the length of the threshold_list ({len(threshold_list)}) is invalid, should be {self.exit_num}' )
            raise NotImplementedError
        for i in range(len( threshold_list )):
            self.activation_threshold_list.append(abs( threshold_list[i] ))
        

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
    
    def set_possible_layer(self, temp):
        self.possible_layer = temp

    def get_specific_exit_number(self, iterate):
        return self.num_early_exit_list[iterate]
    
    def _calculate_max_activation( self, param ):
        '''
        return the maximum activation item in [param]
        '''
        return torch.max( torch.abs( torch.max( param ) ), torch.abs( torch.min( param ) ) )    

    def simple_conv1d(self, param):
        copy = param.copy()
        temp = []
        copy.insert(0,0)
        copy.append(0)
        for i in range(len(copy)-2):
            temp.append(copy[i]+copy[i+1]+copy[i+2])
        return temp
            
    def prediction(self, param, layer):
        round = 8 
        temp = []
        for i in range(data_type):
            temp.append(param[0][i].item())
        for i in range(round):
            temp = self.simple_conv1d(temp)
            length = len(self.activation_threshold_list)
            temp_thresold = self.activation_threshold_list[layer+i] if (layer+i) < length else self.activation_threshold_list[length-1]
            if max(abs(max(temp)), abs(min(temp))) > self.beta * temp_thresold:
                return i+1
        return round
    
    def output( self ):
        return (self.count_store, self.layer_store) , self.jumpstep_store, self.prediction_store
        
    def settings( self, layer, forward_mode, p = 0 ):
        self.target_layer = layer
        self.start_layer = layer
        self.forward_mode = forward_mode
        self.quant_switch = p

    def forward( self, x):
        if self.forward_mode == 'accuracy_forward':
            return self.accuracy_forward(x)
        elif self.forward_mode == 'quant_forward':
            return self.quant_forward(x)
        elif self.forward_mode == 'test_on_train_forward':
            return self.test_on_train_forward(x)
        elif self.forward_mode == 'test_on_train_normal_forward':
            return self.test_on_train_normal_forward(x)       
        else:
            return self.normal_forward(x)    
    
    def accuracy_forward( self, x ):
        
        flag = 0
        if self.quant_switch == 1:
            x = self.quant(x)
        x = F.relu(self.bn1(self.conv1(x)))
        current_layer = 1
        if self.target_layer == 1:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_0(self.dequant(x))
            else:
                x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / self.beta * self.activation_threshold_list[0]
            if ratio >= 1:
                self.num_early_exit_list[0] += 1
                return 0, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        

        x = F.relu(self.bn2(self.conv2(x)))
        current_layer = 2
        if self.target_layer == 2:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_0(self.dequant(x))
            else:
                x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[1])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[1] += 1
                return 1, exit1
         #   self.count_store.append(ratio.item())
         #  self.prediction_store.append(1)
            self.prediction_store.append(self.prediction(exit1, current_layer))
            self.total[flag] = self.total[flag] + 1
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_0(self.dequant(x))
            else:
                x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)       
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[1])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[1] += 1
                self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 1, exit1       


        x = F.max_pool2d(x, 2, 2)
        current_layer = 3
        if self.target_layer == 3:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_0(self.dequant(x))
            else:
                x_exit_1 = self.exit_0(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[2])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[2] += 1
                return 2, exit1
           # self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
            self.total[flag] = self.total[flag] + 1
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_0(self.dequant(x))
            else:
                x_exit_1 = self.exit_0(x)
            exit1 = self.exit_1_fc(x_exit_1)        
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[2])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[2] += 1
                self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 2, exit1          
        

        x = F.relu(self.bn3(self.conv3(x)))
        current_layer = 4
        if self.target_layer == 4:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[3])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[3] += 1
                return 3, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
            self.total[flag] = self.total[flag] + 1
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)          
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[3])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[3] += 1
                self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 3, exit1           
        

        x = F.relu(self.bn4(self.conv4(x)))
        current_layer = 5
        if self.target_layer == 5:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[4])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[4] += 1
                return 4, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
            self.total[flag] = self.total[flag] + 1
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)          
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[4])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[4] += 1
                self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 4, exit1        
        

        x = F.max_pool2d(x, 2, 2)
        current_layer = 6
        if self.target_layer == 6:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[5] += 1
                return 5, exit1
            #self.count_store.append(self._calculate_cross_entropy(exit1).item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
            self.total[flag] = self.total[flag] + 1
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])            
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[5] += 1
                self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 5, exit1

        x = F.relu(self.bn5(self.conv5(x)))
        current_layer = 7
        if self.target_layer == 7:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[6])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[6] += 1
                return 6, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
            self.total[flag] = self.total[flag] + 1
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[6])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.statics[flag][current_layer-self.target_layer]= self.statics[flag][current_layer-self.target_layer] + 1
                self.num_early_exit_list[6] += 1
                return 6, exit1


        x = F.relu(self.bn6(self.conv6(x)))
        current_layer = 8
        if self.target_layer == 8:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[7])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[7] += 1
                return 7, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
            self.total[flag] = self.total[flag] + 1
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
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
        current_layer = 9
        if self.target_layer == 9:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[8])
            if ratio >= 1:
                self.num_early_exit_list[8] += 1
                return 8, exit1    
            self.prediction_store.append(self.prediction(exit1, current_layer))
            self.total[flag] = self.total[flag] + 1            
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
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
        current_layer = 10
        if self.target_layer == 10:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[9])
            if ratio >= 1:
                self.num_early_exit_list[9] += 1
                return 9, exit1
            self.prediction_store.append(self.prediction(exit1, current_layer))
            self.total[flag] = self.total[flag] + 1                                 
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
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
        if self.quant_switch == 1:
            x_exit_1 =  self.exit_2(self.dequant(x))
        else:
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
        if self.quant_switch == 1:
            x_exit_1 =  self.exit_3(self.dequant(x))
        else:
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
        if self.quant_switch == 1:
            x_exit_1 =  self.exit_3(self.dequant(x))
        else:
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
        if self.quant_switch == 1:
            x_exit_1 =  self.exit_3(self.dequant(x))
        else:
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
        if self.quant_switch == 1:
            x_exit_1 =  self.exit_3(self.dequant(x))
        else:
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
        if self.quant_switch == 1:
            x_exit_1 =  self.exit_3(self.dequant(x))
        else:
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
        if self.quant_switch == 1:
            x_exit_1 =  self.exit_3(self.dequant(x))
        else:
            x_exit_1 = self.exit_3(x)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[16])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]+= 1
            self.num_early_exit_list[16] += 1
            return 16, exit1

        x = F.relu(self.bn14(self.conv14(x)))
        current_layer += 1
        if self.quant_switch == 1:
            x_exit_1 =  self.exit_3(self.dequant(x))
        else:
            x_exit_1 = self.exit_3(x)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[17])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]+= 1
            self.num_early_exit_list[17] += 1
            return 17, exit1
            

        x = F.relu(self.bn15(self.conv15(x)))
        current_layer += 1
        if self.quant_switch == 1:
            x_exit_1 =  self.exit_3(self.dequant(x))
        else:
            x_exit_1 = self.exit_3(x)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[18])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]+= 1
            self.num_early_exit_list[18] += 1
            return 18, exit1
            
        
        x = F.relu(self.bn16(self.conv16(x)))
        current_layer += 1
        if self.quant_switch == 1:
            x_exit_1 =  self.exit_3(self.dequant(x))
        else:
            x_exit_1 = self.exit_3(x)
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
        if self.quant_switch == 1:
            x = self.dequant(x)
        self.original += 1
        current_layer += 1
        self.jumpstep_store.append(current_layer-self.target_layer)
        self.statics[flag][current_layer-self.target_layer]+= 1
        return 20, x
    

    def normal_forward( self, x ):
        self.target_layer = self.start_layer
        count = 0
        
        if self.quant_switch == 1:
            x = self.quant(x)
        x = F.relu(self.bn1(self.conv1(x))) #3*32*64*32
        current_layer = 1
        if self.target_layer == 1:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_0(self.dequant(x))
            else:
                x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 256*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / self.beta * self.activation_threshold_list[0]
            if ratio >= 1:
                self.num_early_exit_list[0] += 1
                return 0, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                count += 1

        x = F.relu(self.bn2(self.conv2(x))) #32*32*64*64
        current_layer = 2
        if self.target_layer == 2:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_0(self.dequant(x))
            else:
                x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[1])
            if ratio >= 1:
                self.num_early_exit_list[1] += 1
                return 1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                count+=1
            
        x = F.max_pool2d(x, 2, 2) #32*32*64*64
        current_layer = 3
        if self.target_layer == 3:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_0(self.dequant(x))
            else:
                x_exit_1 = self.exit_0(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[2])
            if ratio >= 1:
                self.num_early_exit_list[2] += 1
                return 2, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                count+=1                

        x = F.relu(self.bn3(self.conv3(x))) #16*16*128*128
        current_layer = 4
        if self.target_layer == 4:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[3])
            if ratio >= 1:
                self.num_early_exit_list[3] += 1
                return 3, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                count+=1
            
        x = F.relu(self.bn4(self.conv4(x))) #16*16*128*128
        current_layer = 5
        if self.target_layer == 5:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[4])    
            if ratio >= 1:
                self.num_early_exit_list[4] += 1
                return 4, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                count+=1

        x = F.max_pool2d(x, 2, 2) #16*16*128*128
        current_layer = 6
        if self.target_layer == 6:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * 19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            if ratio >= 1:
                self.num_early_exit_list[5] += 1
                return 5, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                count+=1

        x = F.relu(self.bn5(self.conv5(x))) # 8*8*128*256
        current_layer = 7
        if self.target_layer == 7:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 256 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[6])
            if ratio >= 1:
                self.num_early_exit_list[6] += 1
                return 6, exit1
            else:
               self.target_layer +=  self.prediction(exit1, current_layer)
               count+=1

        x = F.relu(self.bn6(self.conv6(x))) # 8*8*256*256
        current_layer = 8
        if self.target_layer == 8:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 256 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[7])
            if ratio >= 1:
                self.num_early_exit_list[7] += 1
                return 7, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                count+=1

        x = F.relu(self.bn7(self.conv7(x))) # 8*8*256*256
        current_layer = 9
        if self.target_layer == 9:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 256 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[8])
            if ratio >= 1:
                self.num_early_exit_list[8] += 1
                return 8, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                count+=1
        
        x = F.relu(self.bn8(self.conv8(x))) # 8*8*256*256
        current_layer = 10
        if self.target_layer == 10:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)# 256 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[9])
            if ratio >= 1:
                self.num_early_exit_list[9] += 1
                return 9, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                count+=1

        x = F.max_pool2d(x, 2, 2) # 4*4*256*256
        current_layer = 11
        if self.target_layer == 11:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            exit1 = self.exit_1_fc(x_exit_1)  # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
            if ratio >= 1:
                self.num_early_exit_list[10] += 1
                return 10, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                count+=1 
        
        x = F.relu(self.bn9(self.conv9(x))) # 256*512*4*4
        current_layer = 12
        if self.target_layer == 12:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 512 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[11])
            if ratio >= 1:
                self.num_early_exit_list[11] += 1
                return 11, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
                count+=1 

        x = F.relu(self.bn10(self.conv10(x))) #512*512*4*4
        current_layer = 13
        if self.target_layer == 13:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 512 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[12])
            if ratio >= 1:
                self.num_early_exit_list[12] += 1
                return 12, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
                count+=1
             
        x = F.relu(self.bn11(self.conv11(x))) #512*512*4*4
        current_layer = 14
        if self.target_layer == 14:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 512 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1)  # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[13])
            if ratio >= 1:
                self.num_early_exit_list[13] += 1
                return 13, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
                count+=1

        x = F.relu(self.bn12(self.conv12(x))) #512*512*4*4
        current_layer = 15
        if self.target_layer == 15:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 512 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[14])
            if ratio >= 1:
                self.num_early_exit_list[14] += 1
                return 14, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
                count+=1

        x = F.max_pool2d(x, 2, 2) #512*512*2*2
        self.count_store[3] += 512*512*2*2
        current_layer = 16
        if self.target_layer == 16:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[15])
            if ratio >= 1:
                self.num_early_exit_list[15] += 1
                return 15, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
                count+=1

        x = F.relu(self.bn13(self.conv13(x))) #512*512*2*2
        current_layer = 17
        if self.target_layer == 17:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1)  # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[16])
            if ratio >= 1:
                self.num_early_exit_list[16] += 1
                return 16, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
                count+=1

        x = F.relu(self.bn14(self.conv14(x))) #512*512*2*2
        current_layer = 18
        if self.target_layer == 18:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1)  # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[17])
            if ratio >= 1:
                self.num_early_exit_list[17] += 1
                return 17, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
                count+=1

        x = F.relu(self.bn15(self.conv15(x))) #512*512*2*2
        current_layer = 19
        if self.target_layer == 19:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[18])
            if ratio >= 1:
                self.num_early_exit_list[18] += 1
                return 18, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
                count+=1
        
        x = F.relu(self.bn16(self.conv16(x))) #512*512*2*2
        current_layer = 20
        if self.target_layer == 20:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[19])
            if ratio >= 1:
                self.num_early_exit_list[19] += 1
                return 19, exit1
    
        current_layer+=1
        x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.reshape(-1, 512)
        x = self.fc(x)
        if self.quant_switch == 1:
            x = self.dequant(x)
        #count += 512*512*2*2*4 + 512*512*2*2 + 2*2*512*64 + 64 * (2*10-1) + 512*512*4*4*4 + 4*4*256*256 + 4*4*256*64 + 64 * (2*10-1) + 8*8*256*256*4 + 8*8*128*64 + 64 * 19
        self.original += 1
        return 20, x


    def test_on_train_forward( self, x ):
        self.target_layer = self.start_layer
        if self.quant_switch == 1:
            x = self.quant(x)
        x = F.relu(self.bn1(self.conv1(x)))
        current_layer = 1
        if self.target_layer == 1:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_0(self.dequant(x))
            else:
                x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / self.beta * self.activation_threshold_list[0]     
            if ratio >= 1:
                self.num_early_exit_list[0] += 1
                return 0, exit1

        x = F.relu(self.bn2(self.conv2(x)))
        current_layer = 2
        if self.target_layer == 2:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_0(self.dequant(x))
            else:
                x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[1])
            if ratio >= 1:
                self.num_early_exit_list[1] += 1
                return 1, exit1

            
        x = F.max_pool2d(x, 2, 2)
        current_layer = 3
        if self.target_layer == 3:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_0(self.dequant(x))
            else:
                x_exit_1 = self.exit_0(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[2])  
            if ratio >= 1:
                self.num_early_exit_list[2] += 1
                return 2, exit1

        x = F.relu(self.bn3(self.conv3(x)))
        current_layer = 4
        if self.target_layer == 4:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[3])
            if ratio >= 1:
                self.num_early_exit_list[3] += 1
                return 3, exit1
            
        x = F.relu(self.bn4(self.conv4(x)))
        current_layer = 5
        if self.target_layer == 5:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[4])
            if ratio >= 1:
                self.num_early_exit_list[4] += 1
                return 4, exit1

        x = F.max_pool2d(x, 2, 2)
        current_layer = 6
        if self.target_layer == 6:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * 19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            #count += 8*8*128*64 + 64 * 19
            if ratio >= 1:
                self.num_early_exit_list[5] += 1
                return 5, exit1

        x = F.relu(self.bn5(self.conv5(x))) # 8*8*128*256
        current_layer = 7
        if self.target_layer == 7:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 256 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[6])
            #count += 8*8*128*256 + 8*8*256*64 + 256 * (2*64-1) + 64*19
            if ratio >= 1:
                self.num_early_exit_list[6] += 1
                return 6, exit1
               
        x = F.relu(self.bn6(self.conv6(x))) # 8*8*256*256
        current_layer = 8
        if self.target_layer == 8:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 256 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[7])
            #count += 8*8*256*256 + 8*8*256*64 + 256 * (2*64-1) + 64*19
            if ratio >= 1:
                self.num_early_exit_list[7] += 1
                return 7, exit1

        x = F.relu(self.bn7(self.conv7(x))) # 8*8*256*256
        current_layer = 9
        if self.target_layer == 9:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 256 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[8])
            #count +=  8*8*256*256 + 8*8*256*64 + 256 * (2*64-1) + 64*19
            if ratio >= 1:
                self.num_early_exit_list[8] += 1
                return 8, exit1
        
        x = F.relu(self.bn8(self.conv8(x))) # 8*8*256*256
        current_layer = 10
        if self.target_layer == 10:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)# 256 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[9])
            #count += 8*8*256*256 + 8*8*256*64 + 256 * (2*64-1) + 64*19  
            if ratio >= 1:
                self.num_early_exit_list[9] += 1
                return 9, exit1

        x = F.max_pool2d(x, 2, 2) # 4*4*256*256
        current_layer = 11
        if self.target_layer == 11:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x) 
            exit1 = self.exit_1_fc(x_exit_1)  # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
            #count += 4*4*256*256 + 4*4*256*64 + 64 * (2*10-1)
            if ratio >= 1:
                self.num_early_exit_list[10] += 1
                return 10, exit1
        
        x = F.relu(self.bn9(self.conv9(x))) # 256*512*4*4
        current_layer = 12
        if self.target_layer == 12:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 512 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[11])
            #count += 256*512*4*4 + 4*4*512*64 + 512 * (2*64-1) + 64 * (2*10-1)
            if ratio >= 1:
                self.num_early_exit_list[11] += 1
                return 11, exit1

        x = F.relu(self.bn10(self.conv10(x))) #512*512*4*4
        current_layer = 13
        if self.target_layer == 13:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 512 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[12])
            #count += 512*512*4*4 +  4*4*512*64 + 512 * (2*64-1) + 64 * (2*10-1)
            if ratio >= 1:
                self.num_early_exit_list[12] += 1
                return 12, exit1
          
        x = F.relu(self.bn11(self.conv11(x))) #512*512*4*4
        current_layer = 14
        if self.target_layer == 14:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 512 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1)  # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[13])
            #count += 512*512*4*4 + 4*4*512*64 + 512 * (2*64-1) + 64 * (2*10-1)
            if ratio >= 1:
                self.num_early_exit_list[13] += 1
                return 13, exit1

        x = F.relu(self.bn12(self.conv12(x))) #512*512*4*4
        current_layer = 15
        if self.target_layer == 15:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 512 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[14])
            #count += 512*512*4*4 + 4*4*512*64 + 512 * (2*64-1) + 64 * (2*10-1)
            if ratio >= 1:
                self.num_early_exit_list[14] += 1
                return 14, exit1

        x = F.max_pool2d(x, 2, 2) #512*512*2*2     
        current_layer = 16
        if self.target_layer == 16:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[15])
            #count += 512*512*2*2 + 2*2*512*64 + 64 * (2*10-1)
            time1 = time.perf_counter()
            layer_st = current_layer    
            if ratio >= 1:
                self.num_early_exit_list[15] += 1
                return 15, exit1

        x = F.relu(self.bn13(self.conv13(x))) #512*512*2*2
        current_layer = 17
        if self.target_layer == 17:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1)  # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[16])
            #count += 512*512*2*2 + 2*2*512*64 + 64 * (2*10-1)
            if ratio >= 1:
                self.num_early_exit_list[16] += 1
                return 16, exit1

        x = F.relu(self.bn14(self.conv14(x))) #512*512*2*2
        current_layer = 18
        if self.target_layer == 18:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1)  # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[17])
            #count += 512*512*2*2 + 2*2*512*64 + 64 * (2*10-1)  
            if ratio >= 1:
                self.num_early_exit_list[17] += 1
                return 17, exit1

        x = F.relu(self.bn15(self.conv15(x))) #512*512*2*2
        current_layer = 19
        if self.target_layer == 19:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[18])
            #count += 512*512*2*2 +  2*2*512*64 + 64 * (2*10-1)
            if ratio >= 1:
                self.num_early_exit_list[18] += 1
                return 18, exit1
        
        x = F.relu(self.bn16(self.conv16(x))) #512*512*2*2
        current_layer = 20
        if self.target_layer == 20:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[19])
            #count += 512*512*2*2 + 2*2*512*64 + 64 * (2*10-1)
            layer_st = current_layer    
            if ratio >= 1:
                self.num_early_exit_list[19] += 1
                return 19, exit1
    
        x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.reshape(-1, 512)
        x = self.fc(x)
        #count += 512*512*2*2*4 + 512*512*2*2 + 2*2*512*64 + 64 * (2*10-1) + 512*512*4*4*4 + 4*4*256*256 + 4*4*256*64 + 64 * (2*10-1) + 8*8*256*256*4 + 8*8*128*64 + 64 * 19
        self.original += 1
        return 20, x


    def test_on_train_normal_forward( self, x ):
        self.target_layer = self.start_layer
        count = 0
        time2 = 0
        time1 = time.perf_counter()
        layer_st = 0
        layer_end = 0
        if self.quant_switch == 1:
            x = self.quant(x)

        x = F.relu(self.bn1(self.conv1(x))) #3*32*64*32
        current_layer = 1
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_0(self.dequant(x))
            else:
                x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 256*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / self.beta * self.activation_threshold_list[0]  
            if ratio >= 1:
                self.num_early_exit_list[0] += 1
                return 0, exit1


        x = F.relu(self.bn2(self.conv2(x))) #32*32*64*64
        current_layer = 2
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_0(self.dequant(x))
            else:
                x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[1])
            if ratio >= 1:
                self.num_early_exit_list[1] += 1
                return 1, exit1
            
        x = F.max_pool2d(x, 2, 2) #32*32*64*64
        current_layer = 3
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_0(self.dequant(x))
            else:
                x_exit_1 = self.exit_0(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[2])
            if ratio >= 1:
                self.num_early_exit_list[2] += 1
                return 2, exit1    

        x = F.relu(self.bn3(self.conv3(x))) #16*16*128*128
        current_layer = 4
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[3])
            if ratio >= 1:
                self.num_early_exit_list[3] += 1
                return 3, exit1

            
        x = F.relu(self.bn4(self.conv4(x))) #16*16*128*128
        current_layer = 5
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[4])           
            if ratio >= 1:
                self.num_early_exit_list[4] += 1
                return 4, exit1


        x = F.max_pool2d(x, 2, 2) #16*16*128*128
        current_layer = 6
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * 19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5]) 
            if ratio >= 1:
                self.num_early_exit_list[5] += 1
                return 5, exit1

        x = F.relu(self.bn5(self.conv5(x))) # 8*8*128*256
        current_layer = 7
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 256 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[6])
            if ratio >= 1:
                self.num_early_exit_list[6] += 1
                return 6, exit1

        x = F.relu(self.bn6(self.conv6(x))) # 8*8*256*256
        current_layer = 8
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 256 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[7]) 
            if ratio >= 1:
                self.num_early_exit_list[7] += 1
                return 7, exit1


        x = F.relu(self.bn7(self.conv7(x))) # 8*8*256*256
        current_layer = 9
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 256 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[8])
            if ratio >= 1:
                self.num_early_exit_list[8] += 1
                return 8, exit1

        
        x = F.relu(self.bn8(self.conv8(x))) # 8*8*256*256
        current_layer = 10
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)# 256 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[9])
            if ratio >= 1:
                self.num_early_exit_list[9] += 1
                return 9, exit1


        x = F.max_pool2d(x, 2, 2) # 4*4*256*256
        current_layer = 11
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x) 
            exit1 = self.exit_1_fc(x_exit_1)  # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
            if ratio >= 1:
                self.num_early_exit_list[10] += 1
                return 10, exit1

        
        x = F.relu(self.bn9(self.conv9(x))) # 256*512*4*4
        current_layer = 12
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 512 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[11])
            if ratio >= 1:
                self.num_early_exit_list[11] += 1
                return 11, exit1

        x = F.relu(self.bn10(self.conv10(x))) #512*512*4*4
        current_layer = 13
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 512 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[12])
            if ratio >= 1:
                self.num_early_exit_list[12] += 1
                return 12, exit1
             
        x = F.relu(self.bn11(self.conv11(x))) #512*512*4*4
        current_layer = 14
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 512 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1)  # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[13])
            if ratio >= 1:
                self.num_early_exit_list[13] += 1
                return 13, exit1

        x = F.relu(self.bn12(self.conv12(x))) #512*512*4*4
        current_layer = 15
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 512 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[14])  
            if ratio >= 1:
                self.num_early_exit_list[14] += 1
                return 14, exit1

        x = F.max_pool2d(x, 2, 2) #512*512*2*2
        current_layer = 16
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[15])  
            if ratio >= 1:
                self.num_early_exit_list[15] += 1
                return 15, exit1

        x = F.relu(self.bn13(self.conv13(x))) #512*512*2*2
        current_layer = 17
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1)  # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[16]) 
            if ratio >= 1:
                self.num_early_exit_list[16] += 1
                return 16, exit1

        x = F.relu(self.bn14(self.conv14(x))) #512*512*2*2
        current_layer = 18
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1)  # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[17]) 
            if ratio >= 1:
                self.num_early_exit_list[17] += 1
                return 17, exit1

        x = F.relu(self.bn15(self.conv15(x))) #512*512*2*2
        current_layer = 19
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[18])
            if ratio >= 1:
                self.num_early_exit_list[18] += 1
                return 18, exit1
        
        x = F.relu(self.bn16(self.conv16(x))) #512*512*2*2
        current_layer = 20
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[19])  
            if ratio >= 1:
                self.num_early_exit_list[19] += 1
                return 19, exit1
    
        x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.reshape(-1, 512)
        x = self.fc(x)
        if self.quant_switch == 1:
            x = self.dequant(x)
        #count += 512*512*2*2*4 + 512*512*2*2 + 2*2*512*64 + 64 * (2*10-1) + 512*512*4*4*4 + 4*4*256*256 + 4*4*256*64 + 64 * (2*10-1) + 8*8*256*256*4 + 8*8*128*64 + 64 * 19
        time2 = time.perf_counter()  #Time Sign
        layer_end = current_layer          
        self.count_store[0] += time2 - time1
        self.layer_store[0] += layer_end - layer_st
        time1 = time.perf_counter()
        layer_st = current_layer    
        self.original += 1
        return 20, x


    def quant_forward( self, x ):
        count = 0
        x = self.quant(x)
        x = F.relu(self.bn1(self.conv1(x)))
        current_layer = 1
        if self.target_layer == 1:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_0(x_exit_1)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / self.beta * self.activation_threshold_list[0]
            if ratio >= 1:
                self.num_early_exit_list[0] += 1
                return 0, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)

        x = F.relu(self.bn2(self.conv2(x)))
        current_layer = 2
        if self.target_layer == 2:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_0(x_exit_1)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[1])
            if ratio >= 1:
                self.num_early_exit_list[1] += 1
                return 1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
            
        x = F.max_pool2d(x, 2, 2)
        current_layer = 3
        if self.target_layer == 3:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_0(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[2])
            if ratio >= 1:
                self.num_early_exit_list[2] += 1
                return 2, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)                

        x = F.relu(self.bn3(self.conv3(x)))
        current_layer = 4
        if self.target_layer == 4:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_1(x_exit_1)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[3])
            if ratio >= 1:
                self.num_early_exit_list[3] += 1
                return 3, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
            
        x = F.relu(self.bn4(self.conv4(x)))
        current_layer = 5
        if self.target_layer == 5:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_1(x_exit_1)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[4])
            if ratio >= 1:
                self.num_early_exit_list[4] += 1
                return 4, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)

        x = F.max_pool2d(x, 2, 2)
        #####################################################################################
        '''
        x_exit_1 = self.exit_1(x)
        exit1 = self.exit_1_fc(x_exit_1)
        count += 8*8*128*64 + 64 * 19
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
        if ratio >= 1:
            self.count_store.append(count)
            self.num_early_exit_list[5] += 1
            return 5, exit1
        '''
        #####################################################################################
        current_layer = 6
        if self.target_layer == 6:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_1(x_exit_1) #8*8*128*64
            exit1 = self.exit_1_fc(x_exit_1) # 64 * 19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            count += 8*8*128*64 + 64 * 19
            if ratio >= 1:
                self.num_early_exit_list[5] += 1
                self.count_store.append(count)
                return 5, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)

        x = F.relu(self.bn5(self.conv5(x))) # 8*8*128*256
        current_layer = 7
        if self.target_layer == 7:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_2(x_exit_1) # 8*8*256*64
            x_exit_1 = self.exit_0_fc(x_exit_1) # 256 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[6])
            count += 8*8*128*256 + 8*8*256*64 + 256 * (2*64-1) + 64*19
            if ratio >= 1:
                self.num_early_exit_list[6] += 1
                self.count_store.append(count)
                return 6, exit1
            else:
               self.target_layer +=  self.prediction(exit1, current_layer)

        x = F.relu(self.bn6(self.conv6(x))) # 8*8*256*256
        current_layer = 8
        if self.target_layer == 8:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_2(x_exit_1)   # 8*8*256*64
            x_exit_1 = self.exit_0_fc(x_exit_1) # 256 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[7])
            count += 8*8*256*256 + 8*8*256*64 + 256 * (2*64-1) + 64*19
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[7] += 1
                return 7, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)

        x = F.relu(self.bn7(self.conv7(x))) # 8*8*256*256
        current_layer = 9
        if self.target_layer == 9:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_2(x_exit_1)  # 8*8*256*64
            x_exit_1 = self.exit_0_fc(x_exit_1) # 256 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[8])
            count +=  8*8*256*256 + 8*8*256*64 + 256 * (2*64-1) + 64*19
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[8] += 1
                return 8, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
        
        x = F.relu(self.bn8(self.conv8(x))) # 8*8*256*256
        current_layer = 10
        if self.target_layer == 10:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_2(x_exit_1) # 8*8*256*64 
            x_exit_1 = self.exit_0_fc(x_exit_1)# 256 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[9])
            count += 8*8*256*256 + 8*8*256*64 + 256 * (2*64-1) + 64*19
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[9] += 1
                return 9, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)

        x = F.max_pool2d(x, 2, 2) # 4*4*256*256
        ##############################################################################################
        '''
        x_exit_1 = self.exit_2(x) # 4*4*256*64 
        exit1 = self.exit_1_fc(x_exit_1)  # 64 * (2*10-1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
        count += 4*4*256*256 + 4*4*256*64 + 64 * (2*10-1) + 8*8*256*256*4 + 8*8*128*64 + 64 * 19
        if ratio >= 1:
            self.count_store.append(count)
            self.num_early_exit_list[10] += 1
            return 10, exit1
        '''
        ###############################################################################################
        current_layer = 11
        if self.target_layer == 11:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_2(x_exit_1) # 4*4*256*64 
            exit1 = self.exit_1_fc(x_exit_1)  # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
            count += 4*4*256*256 + 4*4*256*64 + 64 * (2*10-1)
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[10] += 1
                return 10, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer) 
        
        x = F.relu(self.bn9(self.conv9(x))) # 256*512*4*4
        current_layer = 12
        if self.target_layer == 12:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_3(x_exit_1) # 4*4*512*64
            x_exit_1 = self.exit_0_fc(x_exit_1) # 512 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[11])
            count += 256*512*4*4 + 4*4*512*64 + 512 * (2*64-1) + 64 * (2*10-1)
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[11] += 1
                return 11, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num 

        x = F.relu(self.bn10(self.conv10(x))) #512*512*4*4
        current_layer = 13
        if self.target_layer == 13:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_3(x_exit_1) # 4*4*512*64
            x_exit_1 = self.exit_0_fc(x_exit_1) # 512 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[12])
            count += 512*512*4*4 +  4*4*512*64 + 512 * (2*64-1) + 64 * (2*10-1)
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[12] += 1
                return 12, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
             
        x = F.relu(self.bn11(self.conv11(x))) #512*512*4*4
        current_layer = 14
        if self.target_layer == 14:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_3(x_exit_1) # 4*4*512*64
            x_exit_1 = self.exit_0_fc(x_exit_1) # 512 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1)  # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[13])
            count += 512*512*4*4 + 4*4*512*64 + 512 * (2*64-1) + 64 * (2*10-1)
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[13] += 1
                return 13, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn12(self.conv12(x))) #512*512*4*4
        current_layer = 15
        if self.target_layer == 15:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_3(x_exit_1) # 4*4*512*64
            x_exit_1 = self.exit_0_fc(x_exit_1) # 512 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[14])
            count += 512*512*4*4 + 4*4*512*64 + 512 * (2*64-1) + 64 * (2*10-1)
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[14] += 1
                return 14, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.max_pool2d(x, 2, 2) #512*512*2*2
        #################################################################################
        '''
        x_exit_1 = self.exit_3(x) # 2*2*512*64
        exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[15])
        count += 512*512*2*2 + 2*2*512*64 + 64 * (2*10-1) + 512*512*4*4*4 + 4*4*256*256 + 4*4*256*64 + 64 * (2*10-1) + 8*8*256*256*4 + 8*8*128*64 + 64 * 19
        if ratio >= 1:
            self.count_store.append(count)
            self.num_early_exit_list[15] += 1
            return 15, exit1   
        '''
        ###############################################################################     
        current_layer = 16
        if self.target_layer == 16:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_3(x_exit_1) # 2*2*512*64
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[15])
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[15] += 1
                return 15, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn13(self.conv13(x))) #512*512*2*2
        current_layer = 17
        if self.target_layer == 17:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_3(x_exit_1)  # 2*2*512*64
            exit1 = self.exit_1_fc(x_exit_1)  # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[16])
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[16] += 1
                return 16, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn14(self.conv14(x))) #512*512*2*2
        current_layer = 18
        if self.target_layer == 18:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_3(x_exit_1)  # 2*2*512*64
            exit1 = self.exit_1_fc(x_exit_1)  # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[17])
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[17] += 1
                return 17, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn15(self.conv15(x))) #512*512*2*2
        current_layer = 19
        if self.target_layer == 19:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_3(x_exit_1) # 2*2*512*64
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[18])
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[18] += 1
                return 18, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
        
        x = F.relu(self.bn16(self.conv16(x))) #512*512*2*2
        current_layer = 20
        if self.target_layer == 20:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_3(x_exit_1) # 2*2*512*64
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[19])
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[19] += 1
                return 19, exit1
    
        x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.reshape(-1, 512)
        x = self.fc(x)
        self.count_store.append(count)
        self.original += 1
        x = self.dequant(x)
        return 20, x


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

        x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.view(-1, 512)
        x = self.fc(x)
        return (exit1, exit2, exit3, exit4, exit5, exit6, exit7, exit8, exit9, exit10, exit11, exit12, exit13, exit14, exit15, exit16, exit17, exit18, exit19,exit20,x)



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
        self.fc = nn.Linear(512*2*2, data_type)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, data_type)
        )
    
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
  #      x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.view(-1, 512*2*2)
        x = self.classifier(x)
        return x






if __name__ == '__main__':
    args = utils.Namespace( model_name='resnet',
                            pretrained_file='autodl-tmp/vgg19_train_exits_cifar10.pt',
                            optimizer='adam',
                            train_mode='original',
                            evaluate_mode='exits',
                            task='train',
                            device='cuda',
                            trained_file_suffix='cifar10',
                            beta=9,
                            save=0,
                            dataset_type = 'cifar10',
                            jump = 1
                            )