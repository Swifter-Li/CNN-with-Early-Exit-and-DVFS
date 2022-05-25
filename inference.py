"""
DESCRIPTION:    this file contains all the functions to apply adaptive early-exit mechanisms
                during the inference process of the network

"""

from socket import SO_OOBINLINE
import time
from requests import models
import torch

from torch.utils.data import DataLoader
import models_1
from models_1 import get_eval_model
import global_param as gp
import utils_new as utils
from train import test_exits
import create_custom_dataloader as ccd
from create_custom_dataloader import custom_cifar
import numpy as np
from global_param import cifar10_computataional_cost, resnet_computataional_cost
from models_3 import total_round
from tutorial import evaluate
import os
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


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def inference( args, iterate ):
    '''
    conduct the inference
    1. get and load model according to args.pretrained_file
    2. do the test using the functions in train_new.py
    3. save the model
    '''
    forward_set = 'accuracy_forward' if args.forward_mode == 'accuracy_forward' else 'normal_forward'
    if args.quantization == 1:
        args.device = 'cpu'
    model = get_inference_model( args )
    some_qconfig =  torch.quantization.get_default_qconfig('fbgemm')
    model.eval()
    if args.model_name == 'vgg19':
        if args.quantization == 1:
            model_fused = torch.quantization.fuse_modules(model, [['conv1', 'bn1'], ['conv2', 'bn2'], ['conv3', 'bn3'],\
            ['conv4', 'bn4'],['conv5', 'bn5'],['conv6', 'bn6'],['conv7', 'bn7'],['conv8', 'bn8'],['conv9', 'bn9'],['conv10', 'bn10'],\
                ['conv11', 'bn11'],['conv12', 'bn12'],['conv13', 'bn13'],['conv14', 'bn14'],['conv15', 'bn15'],['conv16', 'bn16']])
            model_fused.qconfig = some_qconfig
            if args.dataset_type == 'stl10':
                model_fused.exit_0.qconfig = None
                model_fused.exit_0_fc.qconfig = None
                model_fused.exit_1.qconfig = None
                model_fused.exit_1_fc.qconfig = None
                model_fused.exit_2.qconfig = None
                model_fused.exit_2_fc.qconfig = None
                model_fused.exit_3.qconfig = None
            else:
                model_fused.exit_0.qconfig = None
                model_fused.exit_0_fc.qconfig = None
                model_fused.exit_1.qconfig = None
                model_fused.exit_1_fc.qconfig = None
                model_fused.exit_2.qconfig = None
                model_fused.exit_3.qconfig = None
            model_prepared = torch.quantization.prepare(model_fused)
            model_prepared.settings(0, 'normal_forward', 1)
            evaluate(model_prepared,gp.criterion, gp.get_dataloader( args, task='train' ), 8)
            model = torch.quantization.convert(model_prepared)
            print(model)
            model.settings(iterate, forward_set, 1)
        else:
            model.settings(iterate, forward_set, 0)
    else:
        if args.quantization == 1:
            model_fused = torch.quantization.fuse_modules(model,[
                ['layer1_block1_conv1', 'layer1_block1_bn1'],['layer1_block1_conv2', 'layer1_block1_bn2'], \
                ['layer1_block2_conv1', 'layer1_block2_bn1'],['layer1_block2_conv2', 'layer1_block2_bn2'], \
                ['layer1_block3_conv1', 'layer1_block3_bn1'],['layer1_block3_conv2', 'layer1_block3_bn2'], \
                ['layer2_block1_conv1', 'layer2_block1_bn1'],['layer2_block1_conv2', 'layer2_block1_bn2'], \
                ['layer2_block1_shortcut', 'layer2_block1_bn3'], \
                ['layer2_block2_conv1', 'layer2_block2_bn1'],['layer2_block2_conv2', 'layer2_block2_bn2'], \
                ['layer2_block3_conv1', 'layer2_block3_bn1'],['layer2_block3_conv2', 'layer2_block3_bn2'], \
                ['layer2_block4_conv1', 'layer2_block4_bn1'],['layer2_block4_conv2', 'layer2_block4_bn2'], \
                ['layer3_block1_conv1', 'layer3_block1_bn1'],['layer3_block1_conv2', 'layer3_block1_bn2'], \
                ['layer3_block1_shortcut', 'layer3_block1_bn3'], \
                ['layer3_block2_conv1', 'layer3_block2_bn1'],['layer3_block2_conv2', 'layer3_block2_bn2'], \
                ['layer3_block3_conv1', 'layer3_block3_bn1'],['layer3_block3_conv2', 'layer3_block3_bn2'], \
                ['layer3_block4_conv1', 'layer3_block4_bn1'],['layer3_block4_conv2', 'layer3_block4_bn2'], \
                ['layer3_block5_conv1', 'layer3_block5_bn1'],['layer3_block5_conv2', 'layer3_block5_bn2'], \
                ['layer3_block6_conv1', 'layer3_block6_bn1'],['layer3_block6_conv2', 'layer3_block6_bn2'], \
                ['layer4_block1_conv1', 'layer4_block1_bn1'],['layer4_block1_conv2', 'layer4_block1_bn2'], \
                ['layer4_block1_shortcut', 'layer4_block1_bn3'], \
                ['layer4_block2_conv1', 'layer4_block2_bn1'],['layer4_block2_conv2', 'layer4_block2_bn2'], \
                ['layer4_block3_conv1', 'layer4_block3_bn1'],['layer4_block3_conv2', 'layer4_block3_bn2']])
            model_fused.qconfig = some_qconfig
            model_fused.block1_exit_0.qconfig = None
            model_fused.block2_exit_0.qconfig = None
            model_fused.block3_exit_0.qconfig = None

            model_fused.block1_exit_1.qconfig = None
            model_fused.block2_exit_1.qconfig = None
            model_fused.block3_exit_1.qconfig = None
            model_fused.block4_exit_1.qconfig = None

            model_fused.block1_exit_2.qconfig = None
            model_fused.block2_exit_2.qconfig = None
            model_fused.block3_exit_2.qconfig = None
            model_fused.block4_exit_2.qconfig = None
            model_fused.block5_exit_2.qconfig = None
            model_fused.block6_exit_2.qconfig = None

            model_fused.block1_exit_3.qconfig = None
            model_fused.block2_exit_3.qconfig = None
            model_fused.block3_exit_3.qconfig = None

            model_fused.layer1_exit_0_fc.qconfig = None
            model_fused.layer1_exit_1_fc.qconfig = None
            model_fused.layer2_exit_0_fc.qconfig = None
            model_fused.layer2_exit_1_fc.qconfig = None
            model_fused.layer3_exit_0_fc.qconfig = None
            model_fused.layer3_exit_1_fc.qconfig = None
            model_fused.layer4_exit_0_fc.qconfig = None
            model_fused.layer4_exit_1_fc.qconfig = None

            model_prepared = torch.quantization.prepare(model_fused)
            model_prepared.settings(0, 'normal_forward', 1)
            evaluate(model_prepared,gp.criterion, gp.get_dataloader( args, task='train' ), 16)
            model = torch.quantization.convert(model_prepared)
            print(model)
            model.settings(iterate, forward_set,1)
        else:
            model.settings(iterate, forward_set)    
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
        count_store, jumpstep, entropy =  model.output()
     #   print(sum(count_store))
                
        plot_show(count_store, jumpstep, entropy, iterate)
     #   model.print_statics()
    


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
accuracy_store = []

def plot_show(ratio_store, jumpstep_store,  entropy,  iterate):
    
    accuracy = 0
    for i in range(len(jumpstep_store)):
        if entropy[i] == jumpstep_store[i]:
            accuracy+=1
        if jumpstep_store[i] > total_round and entropy[i] == total_round:
            accuracy+=1
    if len(jumpstep_store)!=0:
        print('The prediction accuracy is:', accuracy/len(jumpstep_store))
        accuracy_store.append(accuracy/len(jumpstep_store))

   # print(ratio_store[0])
   # print(ratio_store[1])
    


def Early_Exits_Determined( args ):
    ###################################################################################################################################
    ######################               Fuction for Hierarchical and Placement                                                  #######
    ###################################################################################################################################
    Place = []
    '''
    for position in range(0,31):
        model = get_inference_model(args)
        model.eval()
        correct_exit = [0 for i in range( model.exit_num )]
        datasize = 5000 if args.dataset_type == 'stl10' else 20000
        TH = 0.5
        correct = 0
        total = 0
        test_loader = gp.get_dataloader( args, task='test_on_train' )
        if args.evaluate_mode == 'exits':
            correct_list = [0 for _ in range( model.exit_num + 1 )]
            total_list = [0 for _ in range( model.exit_num + 1 )]
            model.settings(position+1, 'test_on_train_forward')
        ##############################################################################################################################
        with torch.no_grad():
            for index, data in enumerate( test_loader ): 
                if index > datasize: break
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
            count_store, jumpstep, entropy =  model.output()          
            plot_show(count_store, jumpstep, entropy,0)
        ###########################################################################################################################
        cm = (datasize - model.get_specific_exit_number(position))/datasize
        y1 = cifar10_computataional_cost[position+2][0] if args.model_name == 'vgg19' else resnet_computataional_cost[position+2][0]
        y0 = cifar10_computataional_cost[position+1][0] if args.model_name == 'vgg19' else resnet_computataional_cost[position+1][0]
        yc = cifar10_computataional_cost[position+1][1] if args.model_name == 'vgg19' else resnet_computataional_cost[position+1][1]
        if (TH-cm)*y1 - (1-cm)*yc - (1-TH)*y0 >= 0:
            Place.append(position + 1)
    '''
    ###################################################################################################################################
    ######################                      Start on Test                                                                   #######
    ###################################################################################################################################
    if args.quantization == 1:
        args.device = 'cpu'
    model = get_inference_model( args )
    some_qconfig =  torch.quantization.get_default_qconfig('fbgemm')
    model.eval()
    if args.model_name == 'vgg19':
        if args.quantization == 1:
            model_fused = torch.quantization.fuse_modules(model, [['conv1', 'bn1'], ['conv2', 'bn2'], ['conv3', 'bn3'],\
            ['conv4', 'bn4'],['conv5', 'bn5'],['conv6', 'bn6'],['conv7', 'bn7'],['conv8', 'bn8'],['conv9', 'bn9'],['conv10', 'bn10'],\
                ['conv11', 'bn11'],['conv12', 'bn12'],['conv13', 'bn13'],['conv14', 'bn14'],['conv15', 'bn15'],['conv16', 'bn16']])
            model_fused.qconfig = some_qconfig
            if args.dataset_type == 'stl10':
                model_fused.exit_0.qconfig = None
                model_fused.exit_0_fc.qconfig = None
                model_fused.exit_1.qconfig = None
                model_fused.exit_1_fc.qconfig = None
                model_fused.exit_2.qconfig = None
                model_fused.exit_2_fc.qconfig = None
                model_fused.exit_3.qconfig = None
            else:
                model_fused.exit_0.qconfig = None
                model_fused.exit_0_fc.qconfig = None
                model_fused.exit_1.qconfig = None
                model_fused.exit_1_fc.qconfig = None
                model_fused.exit_2.qconfig = None
                model_fused.exit_3.qconfig = None
            model_prepared = torch.quantization.prepare(model_fused)
            model_prepared.settings(0, 'normal_forward', 1)
            evaluate(model_prepared,gp.criterion, gp.get_dataloader( args, task='train' ), 16)
            model = torch.quantization.convert(model_prepared)
            print(model)
            model.settings(0,  'test_on_train_normal_forward', 1)
            model.set_possible_layer(Place)
        else:
            model.settings(0, 'test_on_train_normal_forward')
            model.set_possible_layer(Place)
    else:
        if args.quantization == 1:
            model_fused = torch.quantization.fuse_modules(model,[
                ['layer1_block1_conv1', 'layer1_block1_bn1'],['layer1_block1_conv2', 'layer1_block1_bn2'], \
                ['layer1_block2_conv1', 'layer1_block2_bn1'],['layer1_block2_conv2', 'layer1_block2_bn2'], \
                ['layer1_block3_conv1', 'layer1_block3_bn1'],['layer1_block3_conv2', 'layer1_block3_bn2'], \
                ['layer2_block1_conv1', 'layer2_block1_bn1'],['layer2_block1_conv2', 'layer2_block1_bn2'], \
                ['layer2_block1_shortcut', 'layer2_block1_bn3'], \
                ['layer2_block2_conv1', 'layer2_block2_bn1'],['layer2_block2_conv2', 'layer2_block2_bn2'], \
                ['layer2_block3_conv1', 'layer2_block3_bn1'],['layer2_block3_conv2', 'layer2_block3_bn2'], \
                ['layer2_block4_conv1', 'layer2_block4_bn1'],['layer2_block4_conv2', 'layer2_block4_bn2'], \
                ['layer3_block1_conv1', 'layer3_block1_bn1'],['layer3_block1_conv2', 'layer3_block1_bn2'], \
                ['layer3_block1_shortcut', 'layer3_block1_bn3'], \
                ['layer3_block2_conv1', 'layer3_block2_bn1'],['layer3_block2_conv2', 'layer3_block2_bn2'], \
                ['layer3_block3_conv1', 'layer3_block3_bn1'],['layer3_block3_conv2', 'layer3_block3_bn2'], \
                ['layer3_block4_conv1', 'layer3_block4_bn1'],['layer3_block4_conv2', 'layer3_block4_bn2'], \
                ['layer3_block5_conv1', 'layer3_block5_bn1'],['layer3_block5_conv2', 'layer3_block5_bn2'], \
                ['layer3_block6_conv1', 'layer3_block6_bn1'],['layer3_block6_conv2', 'layer3_block6_bn2'], \
                ['layer4_block1_conv1', 'layer4_block1_bn1'],['layer4_block1_conv2', 'layer4_block1_bn2'], \
                ['layer4_block1_shortcut', 'layer4_block1_bn3'], \
                ['layer4_block2_conv1', 'layer4_block2_bn1'],['layer4_block2_conv2', 'layer4_block2_bn2'], \
                ['layer4_block3_conv1', 'layer4_block3_bn1'],['layer4_block3_conv2', 'layer4_block3_bn2']])
            model_fused.qconfig = some_qconfig
            model_fused.block1_exit_0.qconfig = None
            model_fused.block2_exit_0.qconfig = None
            model_fused.block3_exit_0.qconfig = None

            model_fused.block1_exit_1.qconfig = None
            model_fused.block2_exit_1.qconfig = None
            model_fused.block3_exit_1.qconfig = None
            model_fused.block4_exit_1.qconfig = None

            model_fused.block1_exit_2.qconfig = None
            model_fused.block2_exit_2.qconfig = None
            model_fused.block3_exit_2.qconfig = None
            model_fused.block4_exit_2.qconfig = None
            model_fused.block5_exit_2.qconfig = None
            model_fused.block6_exit_2.qconfig = None

            model_fused.block1_exit_3.qconfig = None
            model_fused.block2_exit_3.qconfig = None
            model_fused.block3_exit_3.qconfig = None

            model_fused.layer1_exit_0_fc.qconfig = None
            model_fused.layer1_exit_1_fc.qconfig = None
            model_fused.layer2_exit_0_fc.qconfig = None
            model_fused.layer2_exit_1_fc.qconfig = None
            model_fused.layer3_exit_0_fc.qconfig = None
            model_fused.layer3_exit_1_fc.qconfig = None
            model_fused.layer4_exit_0_fc.qconfig = None
            model_fused.layer4_exit_1_fc.qconfig = None

            model_prepared = torch.quantization.prepare(model_fused)
            model_prepared.settings(0, 'normal_forward', 1)
            evaluate(model_prepared,gp.criterion, gp.get_dataloader( args, task='train' ), 16)
            model = torch.quantization.convert(model_prepared)
            print(model)
            model.settings(0,  'test_on_train_normal_forward', 1)
            model.set_possible_layer(Place)
        else:
            model.settings(0, 'test_on_train_normal_forward')
            model.set_possible_layer(Place) 
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
        count_store, jumpstep, entropy =  model.output()
     #   print(sum(count_store))
                
        plot_show(count_store, jumpstep, entropy, 0)
     #   model.print_statics()
    print("The determined place is ", Place)
    return Place
    

if __name__ == '__main__':
    args = utils.Namespace( model_name='vgg19',
                            pretrained_file='autodl-tmp/vgg19_train_exits_svhn.pt',
                            optimizer='adam',
                            train_mode='exits',
                            evaluate_mode='exits',
                            task='evaluate',
                            device='cuda',
                            trained_file_suffix='update_1',
                            beta=9,
                            save=0,
                            dataset_type = 'svhn',
                            jump = 1,
                            quantization = 0,
                            forward_mode = 'normal_forward'
                            )
    accuracy_layer_store = []
    
    args.beta = 20
   # Early_Exits_Determined(args) 
    inference(args, 6)
    #   Normal Accuracy Testing
    
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