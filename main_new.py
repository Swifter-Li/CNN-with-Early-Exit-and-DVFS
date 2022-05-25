"""
DESCRIPTION:    this file contains the main program

"""

import argparse
import torch

from inference import inference
from train import train
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
    parser.add_argument('--forward_mode', default='normal_forward', type=str, 
                        help='the mode we choose during inference, whether normal or accuracy' )
    parser.add_argument('--start_layer', default=6, type=int, 
                        help='the start layer that we set' )
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
        inference( args,  args.start_layer)
    elif args.task == 'train':
        train( args )
    else:
        print( f'Error: args.task ({args.task}) is not valid. Should be either train or evaluate' )
        raise NotImplementedError
    

