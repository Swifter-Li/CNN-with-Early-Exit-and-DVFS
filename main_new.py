"""
DESCRIPTION:    this file contains the main program

AUTHOR:         Lou Chenfei  Li Xiangjie

INSTITUTE:      Shanghai Jiao Tong University, UM-SJTU Joint Institute

PROJECT:        ECE4730J Advanced Embedded System Capstone Project
"""

import argparse
import torch

from inference_new import inference
from train_new import train
from create_custom_dataloader import custom_cifar

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--model_name', default='vgg19', type=str,
                        help='the model structure', metavar='[cifar, vgg, vgg19]' )
    parser.add_argument( '--pretrained_file', default='models_new/vgg19_exits_update_1.pt', type=str,
                        help='the file name that stores the pre-trained model' )
    parser.add_argument( '--optimizer', default='adam', type=str,
                        help='the optimizer for training', metavar='[sgd, adam]' )
    parser.add_argument( '--train_mode', default='exits', type=str,
                        help='the training mode', metavar='[normal (without exits), original (with exits), exits]' )
    parser.add_argument( '--stat_each_layer', default=0, type=int,
                        help='whether to collect the statistics of each layer, 1 for True and 0 for False' )
    parser.add_argument( '--evaluate_mode', default='exits', type=str,
                        help='the evaluating mode', metavar='[normal (without exits), exits (with exits)]' )
    parser.add_argument( '--task', default='evaluate', type=str,
                        help='to train or to evaluate', metavar='[train, evaluate]' )
    parser.add_argument( '--device', default='cuda', type=str,
                        help='the device on which the model is trained', metavar='[cpu, cuda]' )
    parser.add_argument( '--trained_file_suffix', default='update_2', type=str,
                        help='the suffix added to the name of the file that stores the pre-trained model' )
    parser.add_argument( '--beta', default=6, type=float,
                        help='the coefficient used for accuracy-speed trade-off, the higher the more accurate, range from 0 to 1' )
    parser.add_argument( '--save', default=0, type=int,
                        help='whether or not to save the model. 0 or nonzero' )
    # the following are used to cooperate with hardware control on jetson-tx2
    parser.add_argument( '--baseline', default=1, type=int,
                        help='to specify whether or not the current test is baseline (1 for true and 0 for false)' )
    parser.add_argument( '--core_num', default=2, type=int,
                        help='the number of gpu cores used in inference, among [2, 4]' )
    parser.add_argument( '--cpu_freq_level', default=1, type=int,
                        help='the level of frequency of cpu cores used in inference time. The value for \
                              each level is specified in the file global_param.py. among [4, 8, 12]' )
    parser.add_argument( '--gpu_freq_level', default=1, type=int,
                        help='the level of frequency of gpu cores used in inference time. The value for \
                              each level is specified in the file global_param.py. among [2, 5, 8]' )
    parser.add_argument( '--scene', default='continuous', type=str,
                        help='the application scene. from [continuous, periodical]' )
    parser.add_argument( '--baseline_time', default=30, type=float,
                        help='the time it takes for baseline to finish in periodical scene' )
    parser.add_argument( '--sleep_time', default=30, type=float,
                        help='the time we want the board to sleep in periodical scene, baseline setting' )
    parser.add_argument( '--dataset_type', default='cifar10', type=str,
                        help='the dataset we choose' )
    parser.add_argument('--jump', default=0, type=int, 
                        help='the exit type we choose for Vgg19' )
    parser.add_argument('--quantization', default=0, type=int, 
                        help='the mode we choose during inference, whether normal or quantization' )
    args = parser.parse_args()
    args.torch_device = torch.device( 'cpu' ) if args.device == 'cpu' else torch.device( 'cuda' )
    return args

if __name__ == '__main__':
    args = get_args()
    # print tag
    if args.baseline:
        print( f'{args.scene}_baseline' )
    else:
        print( f"{args.scene}_{str(args.core_num)}_{args.cpu_freq_level}_{args.gpu_freq_level}_{'Y' if args.evaluate_mode=='exits' else 'N'}" )
    if args.task == 'evaluate':
        inference( args,  0)
    elif args.task == 'train':
        train( args )
    else:
        print( f'Error: args.task ({args.task}) is not valid. Should be either train or evaluate' )
        raise NotImplementedError
    



'''
some urgent test that you need to do:
1.  make sure whether the average activation is taken over the classification outputs or
    the codebook outputs···

probable directions for future improvements:
1.  test whether average activations for correct early-exits are higher than noncorrect early-exits?
2.  test whether average activations for different labels are different?
3.  test whether average activations have a limit when the model converges

1.  train_new 讲函数改成通用化 ☑️
2.  global param 把 exit layers 改成 non-exit layers ☑️
3.  models_new 写一下 vgg，并开始尝试训练
4.  写一下 inference

to add a new model:
1.  add new model in model_new.py
2.  add hyper, init, as well as normal_layer_names in global_param.py
3.  (possibly) modify the train_exit function to balance the exit layers from different stages

to adjust the early-exit layer structures:
1.  modify the model structures in model_new.py


todo: 
1. test whether the 'easy' images are consistently easy among testbenches with different initialization
'''