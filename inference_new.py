"""
DESCRIPTION:    this file contains all the functions to apply adaptive early-exit mechanisms
                during the inference process of the network

AUTHOR:         Lou Chenfei Li Xiangjie

INSTITUTE:      Shanghai Jiao Tong University, UM-SJTU Joint Institute

PROJECT:        ECE4730J Advanced Embedded System Capstone Project
"""

import time
import torch

from torch.utils.data import DataLoader
import models_new
from models_new import get_eval_model
import global_param as gp
import utils_new as utils
from train_new import test_exits
import create_custom_dataloader as ccd
from create_custom_dataloader import custom_cifar
from power_management_api import api
import numpy as np
from sklearn.svm import NuSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
def get_inference_model( args ):
    '''
    get the model for inference from the pretrained model
    1. initialize a model according to args.model_name
    2. copy the parameters to model for inference 
    3. return the model
    '''
    # load model parameters
    trained_model = torch.load( args.pretrained_file, map_location='cpu' ).to( args.device )
    eval_model = get_eval_model( args ).to( args.device )
    eval_state_dict = eval_model.state_dict()
    for name, parameter in trained_model.state_dict().items():
        if name in eval_state_dict.keys(): eval_state_dict[name].copy_( parameter )
    eval_model.load_state_dict( eval_state_dict )
    # calculate and load average activation thresholds
    if args.evaluate_mode == 'exits':
        average_activation_list = calculate_average_activations( args, trained_model )
        eval_model.set_activation_thresholds( average_activation_list )
        eval_model.set_beta( args.beta )
    return eval_model


def hardware_sanity_check( args ):
    if args.core_num not in [2, 4]:
        print( f'Error: core_num ({args.core_num}) is invalid, should be among [2, 4]' )
        raise NotImplementedError
    if args.cpu_freq_level not in [4, 8, 12]:
        print( f'Error: cpu_freq_level ({args.cpu_freq_level}) is invalid, should be among [1, 2, 3]' )
        raise NotImplementedError
    if args.gpu_freq_level not in [2, 5, 8]:
        print( f'Error: gpu_freq_level ({args.gpu_freq_level}) is invalid, should be among [1, 2, 3]' )
        raise NotImplementedError
    if args.scene not in ['continuous', 'periodical']:
        print( f'Error: scene ({args.scene}) is invalid, should be among [continuous, periodical]' )
        raise NotImplementedError


def hardware_setup( args ):
    '''
    configure the hardwares (cpus, gpus, frequency, number of cores, sleep time and so on)
    '''
    hardware_sanity_check()
    # to configure cpu core nums
    cpu_list = []
    cpu_list.append( gp.get_cpu_target( 0 ) )
    cpu_list.append( gp.get_cpu_target( 1 ) )
    if args.core_num == 4 or args.baseline:
        cpu_list.append( gp.get_cpu_target( 2 ) )
        cpu_list.append( gp.get_cpu_target( 3 ) )
    else:
        cpu_list.append( gp.get_cpu_target( 2, cpu_online=False ) )
        cpu_list.append( gp.get_cpu_target( 3, cpu_online=False ) )
    # to configure cpu frequency
    if args.baseline:
        for cpu in cpu_list: 
            cpu['min_freq'] = gp.cpu_max_freq
            cpu['max_freq'] = gp.cpu_max_freq
    else:
        for cpu in cpu_list:
            cpu['min_freq'] = gp.cpu_freq_levels[args.cpu_freq_level]
            cpu['max_freq'] = gp.cpu_freq_levels[args.cpu_freq_level]
    # to configure gpu frequency
    if args.baseline:
        gpu = gp.get_gpu_target( min_freq=gp.gpu_max_freq, max_freq=gp.gpu_max_freq )
    else:
        gpu = gp.get_cpu_target( min_freq=gp.gpu_freq_levels[args.gpu_freq_level], 
                                 max_freq=gp.gpu_freq_levels[args.gpu_freq_level] )
    # realize the hardware settings
    api.set_cpu_state( cpu_list )
    api.set_gpu_state( gpu )


def inference( args, iterate ):
    '''
    conduct the inference
    1. get and load model according to args.pretrained_file
    2. do the test using the functions in train_new.py
    3. save the model
    '''
    # configure the hardware according to arguments
   # hardware_setup()
    # get the model
    model = get_inference_model( args )
    model.eval()
    correct = 0
    total = 0
    data_size = 10000
    #generate the data loader
    #test_loader = ccd.get_dataloader( 'hard' )
    test_loader = gp.get_dataloader( args, task='test' )
    if args.evaluate_mode == 'exits':
        correct_list = [0 for _ in range( model.exit_num + 1 )]
        total_list = [0 for _ in range( model.exit_num + 1 )]
    st_time = time.perf_counter()
    with torch.no_grad():
        for index, data in enumerate( test_loader ): 
            if index > data_size: break
            images, labels = data
            images, labels = images.to( args.device ), labels.to( args.device )
            outputs = model( images )
            if args.evaluate_mode == 'exits':
                exit_layer, outputs = outputs
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size( 0 )
            correct += ( predicted == labels ).sum().item()
            if args.evaluate_mode == 'exits':
                total_list[exit_layer] += labels.size( 0 )
                correct_list[exit_layer] += ( predicted == labels ).sum().item()
    general_acc = 100 * correct / total
    if args.evaluate_mode == 'exits':
        acc_list = [correct_list[i]/total_list[i] if total_list[i] != 0 else None for i in range( len( correct_list ) )]
    end_time = time.perf_counter()
    print( f'time consumed: {end_time - st_time}' )
    print('Accuracy of the network on the 10000 test images: %d %%' % general_acc)
    if args.evaluate_mode == 'exits':
        for exit_idx in range( len( correct_list ) ):
            if acc_list[exit_idx] != None:
                print( f'exit{str(exit_idx)}: {100*acc_list[exit_idx]: .3f}%', end=' | ' )
            else:
                print( f'exit{str(exit_idx)}: {None}', end=' | ' )
        print( '' )

    if args.evaluate_mode == 'exits': 
        model.print_exit_percentage()
        count = 0
        ratio, jumpstep, entropy, ratio_list =  model.output()
                
        plot_show(ratio, jumpstep, entropy, iterate)
        model.print_statics()
    if args.save: torch.save( model, utils.create_model_file_name( args ) )
    a = 1
    '''
    Hard-ware used Inference
    num_testcase = gp.num_testcase_continuous if args.scene == 'continuous' else gp.num_testcase_periodical
    dataset = custom_cifar()
    dataloader_list = []
    for idx in range( num_testcase ):
        dataloader_list.append( DataLoader( torch.load( args.pretrained_file,
                                            batch_size=1,
                                            shuffle=True,
                                            num_workers=4 ) ) )
    # do the inference
    if args.evaluate_mode == 'exits' and args.stat_each_layer:
        correct_list = [0 for _ in range( model.exit_num + 1 )]
        total_list = [0 for _ in range( model.exit_num + 1 )]
    period = args.baseline_time + args.sleep_time
    st_time = time.perf_counter()
    print( f'timestamp: {st_time}' )
    with torch.no_grad():
        # the loop for test cases
        for case_idx in range( num_testcase ):
            # the loop for images
            for index, data in enumerate( dataloader_list[case_idx] ): 
                pre_inference_time = time.perf_counter()
                images, labels = data
                images, labels = images.to( args.device ), labels.to( args.device )
                outputs = model( images )
                if args.evaluate_mode == 'exits':
                    exit_layer, outputs = outputs
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size( 0 )
                correct += ( predicted == labels ).sum().item()
                if args.evaluate_mode == 'exits' and args.stat_each_layer:
                    total_list[exit_layer] += labels.size( 0 )
                    correct_list[exit_layer] += ( predicted == labels ).sum().item()
                # sleep control
                post_inference_time = time.perf_counter()
                inference_time = post_inference_time - pre_inference_time
                sleep_time = period - inference_time   # at least 0.5 second for wake up
                if args.scene == 'periodical' and args.baseline == 0:
                    # sleep
                    assert api.sleep_with_time( int(sleep_time-0.5) ) == 0
                elif args.scene == 'periodical' and args.baseline == 1:
                    # polling without sleep
                    while time.perf_counter() - post_inference_time < sleep_time: pass
    end_time = time.perf_counter()
    general_acc = 100 * correct / total
    if args.evaluate_mode == 'exits':
        acc_list = [correct_list[i]/total_list[i] if total_list[i] != 0 else None for i in range( len( correct_list ) )]
    print( f'time consumed: {end_time - st_time}' )
    print('Accuracy of the network on the 10000 test images: %d %%' % general_acc)
    if args.evaluate_mode == 'exits' and args.stat_each_layer:
        for exit_idx in range( len( correct_list ) ):
            if acc_list[exit_idx] != None:
                print( f'exit{str(exit_idx)}: {100*acc_list[exit_idx]: .3f}%', end=' | ' )
            else:
                print( f'exit{str(exit_idx)}: {None}', end=' | ' )
        print( '' )
    if args.evaluate_mode == 'exits': model.print_exit_percentage()
    if args.save: torch.save( model, utils.create_model_file_name( args ) )
    '''


def calculate_average_activations( args, model, verbose=True ):
    exit_layer = model.exit_layer
    exit_num = model.exit_num
    average_activation_list = [0 for _ in range( exit_num )]
    average_times_list = [0 for _ in range( exit_num )]
    model.set_exit_layer( 'exits' )
    train_loader = gp.get_dataloader( args, 'train' )
    loop_limit = gp.average_activation_train_size / train_loader.batch_size
    for act_idx, (images, labels) in enumerate( train_loader ):
        images, labels = images.to( args.device ), labels.to( args.device )
        outputs = model( images )
        for exit_idx in range( exit_num ):
            average_times_list[exit_idx] += 1
            average_activation_list[exit_idx] += \
                utils.calculate_average_activations( outputs[exit_idx] )
        if act_idx >= loop_limit:
            break
    average_activation_list = [average_activation_list[i] / average_times_list[i] for i in range( exit_num )]
    if verbose:
        for print_idx in range( exit_num ):
            print( f'average activation {print_idx}: {average_activation_list[print_idx]}' )
    model.set_exit_layer( exit_layer )
    return average_activation_list



##############################################################################
###                          Start on Testing                              ###
##############################################################################

def plot_show(ratio_store, jumpstep_store,  entropy,  iterate):


    '''
    plt.plot(ratio_store,jumpstep_store,'ro',markersize=0.3)
    name = 'cifar10_3/graph' + str(iterate) + '.jpg'
    
    plt.savefig(name)
    '''
    ratio_store_filtered = [ratio_store[i] for i in range(len(ratio_store)) if jumpstep_store[i] <= 3]
    jumpstep_store_filtered = [jumpstep_store[i] for i in range(len(jumpstep_store)) if jumpstep_store[i] <= 3]
    entropy_filtered = [entropy[i] for i in range(len(entropy)) if jumpstep_store[i] <= 3]

    color =['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
    jumpstep_store_filtered = [color[i%len(color)] for i in jumpstep_store_filtered]
    plt.scatter(ratio_store_filtered, entropy_filtered, c = jumpstep_store_filtered, marker = 'o', s = 1)
 
    name = 'cifar10_3/graph' + str(iterate) + '.jpg'

    plt.savefig(name)




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

    for i in range(50, 91):
        args.beta = 1+i*0.1
        inference(args, i)

