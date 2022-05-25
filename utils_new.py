"""
DESCRIPTION:    this file contains general utilities that are used in this project

"""

import torch

class Namespace():
    '''
    a data structure to efficiently pass numerous parameters between function calls
    '''
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def calculate_average_activations( param:torch.Tensor ):
    assert len( param.shape ) == 2
    #A = torch.abs(param)
    return param.sum().item() / param.numel()

def create_model_file_name( args ):
    '''
    the file name should include information for:
        1. model name
        2. train mode
        3. early_exit_configurations
        4. task
    '''
    name =  'autodl-tmp/' + \
            args.model_name + '_' + \
            args.task + '_' + \
            args.train_mode + '_' + \
            args.trained_file_suffix
    return name + '.pt'