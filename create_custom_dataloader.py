"""
DESCRIPTION:    this file contains the codes needed to create new datasets that store easy
                and difficult images, respectively.

"""

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

import global_param as gp
import utils_new as utils
import models_1
from models_1 import get_eval_model
import global_param as gp
from train import test_exits

class custom_cifar( Dataset ):
    '''
    the dataset of cifar10 images that is either easy or difficult
    '''
    def __init__( self, data=None ) -> None:
        '''
        create the set of images and their corresponding labels
        '''
        super().__init__()
        self.images = data if data is not None else []
    def __len__( self ):
        return len( self.images )
    def __getitem__( self, index ):
        return self.images[index]
    def add_item( self, image ):
        self.images.append( image )

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

def get_inference_model( args ):
    '''
    get the model for inference from the pretrained model
    1. initialize a model according to args.model_name
    2. copy the parameters to model for inference 
    3. return the model
    '''
    # load model parameters
    trained_model = torch.load( args.pretrained_file ).to( args.device )
    eval_model = get_eval_model( args ).to( args.device )
    eval_state_dict = eval_model.state_dict()
    for name, parameter in trained_model.state_dict().items():
        if name in eval_state_dict.keys(): eval_state_dict[name].copy_( parameter )
    eval_model.load_state_dict( eval_state_dict )
    average_activation_list = calculate_average_activations( args, trained_model )
    eval_model.set_activation_thresholds( average_activation_list )
    # set beta
    eval_model.set_beta( args.beta )
    return eval_model

def build_easy_hard_dataset( difficulty ):
    args = utils.Namespace( model_name='vgg19',
                            pretrained_file='models_new/vgg19_train_exits_round_1.pt',
                            optimizer='adam',
                            train_mode='exits',
                            evaluate_mode='exits',
                            task='evaluate',
                            device='cuda',
                            trained_file_suffix='update_1',
                            beta=9,
                            save=0,
                            dataset_type = 100)
    my_dataset = custom_cifar()
    if difficulty == 'easy':
        args.beta = 9
    elif difficulty == 'hard':
        args.beta = 5
    model = get_inference_model( args )
    model.eval()
    test_loader = gp.get_dataloader( args, task='test' )
    for image, label in test_loader:
        image_, label_ = image.to( args.device ), label.to( args.device )
        exit_layer, _ = model( image_ )
        if exit_layer == 0 and difficulty == 'easy':
            my_dataset.add_item( (image[0], label) )
        if exit_layer == 2 and difficulty == 'hard':
            my_dataset.add_item( (image[0], label) )
    torch.save( my_dataset, 'dataset_new/'+difficulty+'.pt' )
    return my_dataset

def get_dataloader( difficulty ):
    dataset = custom_cifar()
    dataset = torch.load( 'dataset_new/'+difficulty+'.pt' )
    # dataset = build_easy_hard_dataset( difficulty )
    return DataLoader( dataset=dataset, batch_size=1, shuffle=1, num_workers=4 )

def try_out():
    args = utils.Namespace( model_name='vgg19',
                            pretrained_file='models_new/vgg19_train_exits_update_1.pt',
                            optimizer='adam',
                            train_mode='exits',
                            evaluate_mode='exits',
                            task='evaluate',
                            device='cuda',
                            trained_file_suffix='round_1',
                            beta=9,
                            save=0 )
    my_dataset = custom_cifar()
    dataloader = gp.get_dataloader( args, task='test' )
    for idx, (image, label) in enumerate( dataloader ):
        if idx > 100: break
        # image, label = image.to( args.device ), label.to( args.device )
        my_dataset.add_item( (image, label) )
        # torch.save(  'dataset_new/test.pt' )
    my_dataloader = DataLoader( dataset=my_dataset, batch_size=1, shuffle=True, num_workers=4 )
    for idx, (image, label) in enumerate( my_dataloader ):
        print( f'index: {idx}, image size: {image.shape}, label: {label}' )

def dataset_check():
    args = utils.Namespace( model_name='vgg19',
                            pretrained_file='models_new/vgg19_train_exits_round_1.pt',
                            optimizer='adam',
                            train_mode='exits',
                            evaluate_mode='exits',
                            task='evaluate',
                            device='cuda',
                            trained_file_suffix='round_1',
                            beta=9,
                            save=0,
                            dataset_type = 100
                            )
    my_dataloader = get_dataloader( 'hard' )
    data_loader = gp.get_dataloader( args, task='test' )
    print( f'the dataloader from torchvision' )
    for idx, (image, label) in enumerate( data_loader ):
        if idx > 10: break
        print( f'index: {idx}, image size: {image.shape}, label: {label}' )
    print( f'the dataloader build by us' )
    for idx, (image, label) in enumerate( my_dataloader ):
        if idx > 10: break
        print( f'index: {idx}, image size: {image.shape}, label: {label}' )

def store_dataset():
    build_easy_hard_dataset( 'easy' )
    build_easy_hard_dataset( 'hard' )



if __name__ == '__main__':
   # build_easy_hard_dataset( 'hard' )
    store_dataset()
    dataset_check()
  #  build_easy_hard_dataset('easy')


# dataset_list = [custom_cifar() for i in range(10)]
# for idx, (image, label) in enumerate( original_dataloader ):
#     if idx < 1000:
#         dataset_list[0].add_item( (image[0], label) )
#     elif idx < 2000:
#         dataset_list[1].add_item( (image[0], label) )
#     elif idx < 3000:
#         dataset_list[2].add_item( (image[0], label) )
#     elif idx < 4000:
#         dataset_list[3].add_item( (image[0], label) )
#     elif idx < 5000:
#         dataset_list[4].add_item( (image[0], label) )
#     elif idx < 6000:
#         dataset_list[5].add_item( (image[0], label) )
#     elif idx < 7000:
#         dataset_list[6].add_item( (image[0], label) )
#     elif idx < 8000:
#         dataset_list[7].add_item( (image[0], label) )
#     elif idx < 9000:
#         dataset_list[8].add_item( (image[0], label) )
#     else:
#         dataset_list[9].add_item( (image[0], label) )
# print( [len(i) for i in dataset_list] )