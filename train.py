"""
DESCRIPTION:    this file contains all the functions used for training network with early-exit layers

"""

import torch

from models_1 import get_train_model
import global_param as gp
import utils_new as utils

def train( args ):
    '''
    train the model
    1. get model according to args.model_name
    2. load model according to args.pretrained_file
    3. do the training according to args.train_mode
    4. save model
    '''
    if args.train_mode == 'normal':
        train_normal( args )
    elif args.train_mode == 'original':
        train_original( args )
    elif args.train_mode == 'exits':
        train_exits( args )
    pass


def train_normal( args ):
    '''
    train the model without any early-exit mechanisms, i.e. the most traditional scheme
    '''
    # training setup
    hyper = gp.get_hyper( args )
    model = get_train_model( args ).to( args.device )
    optimizer = gp.get_optimizer( params=model.parameters(), lr=hyper.learning_rate, op_type=args.optimizer )
    train_loader = gp.get_dataloader( args, task='train' )
    # begin training
    best_test_acc = 0
    num_no_increase = 0
    model.train()
    for epoch_idx in range( hyper.epoch_num ):
        if num_no_increase >= 5:
            print( 'early exit is triggered' )
            if args.save: torch.save( model, utils.create_model_file_name( args ) )
            return
        print(f'\nEpoch: {(epoch_idx + 1)}')
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, ( images, labels ) in enumerate( train_loader ):
            images, labels = images.to( args.device ), labels.to( args.device )
            optimizer.zero_grad()
            outputs = model( images )
            loss = gp.criterion( outputs, labels )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max( 1 )
            total += labels.size( 0 )
            correct += predicted.eq( labels ).sum().item()
            if batch_idx % 100 == 99:    # print every 100 mini-batches
                # for debug
                # print( f'outputs = {outputs.sum()}' )
                # end debug
                print('[%d, %5d] loss: %.5f |  Acc: %.3f%% (%d/%d)' %
                    (epoch_idx + 1, batch_idx + 1, train_loss / 2000, 100.*correct/total, correct, total))
                train_loss, total, correct = 0.0, 0, 0
        if epoch_idx % 5 == 4:
            print( f'begin middle test (best_test_acc = {best_test_acc}; num_no_increase = {num_no_increase}):' )
            current_test_acc = test_normal( model, args )
            if current_test_acc > best_test_acc:
                best_test_acc = current_test_acc
                num_no_increase = 0
            else:
                num_no_increase += 1
    test_normal( model, args, verbose=True )
    if args.save: 
        torch.save( model, utils.create_model_file_name( args ) )


def test_normal( model, args, verbose=False ):
    class_num = 0
    if args.dataset_type == 'cifar100':
        class_num = 100
    else:
        class_num = 10
    if model.training:
        model.eval()
        train_flag = True
    else:
        train_flag = False
    correct = 0
    total = 0
    test_loader = gp.get_dataloader( args, task='test' )
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to( args.device ), labels.to( args.device )
            outputs = model( images )
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size( 0 )
            correct += ( predicted == labels ).sum().item()
    general_acc = 100 * correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % general_acc)
    if verbose:
        class_correct = list( 0. for i in range( class_num ) )
        class_total = list( 0. for i in range( class_num ) )
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to( args.device ), labels.to( args.device )
                outputs = model( images )
                _, predicted = torch.max( outputs, 1 )
                c = ( predicted == labels ).squeeze()
                if labels.shape.numel() > 1:
                    for i in range(labels.shape.numel()):
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1
                else:
                    label = labels[0]
                    class_correct[label] += c.item()
                    class_total[label] += 1
        for i in range( class_num ):
            print('Accuracy of Type', i, ' ', "{:.2f}".format(100 * class_correct[i] / class_total[i]))
    if train_flag:
        model.train()
    return general_acc


def train_original( args ):
    '''
    train the model with early-exit structures but only train the original part
    '''
    # training setup
    hyper = gp.get_hyper( args )
    model = get_train_model( args )
    model = model.to( args.device )
    model.set_exit_layer( 'original' )
    optimizer = gp.get_optimizer( params=model.parameters(), lr=hyper.learning_rate, op_type=args.optimizer )
    train_loader = gp.get_dataloader( args, task='train' )
    # begin training
    best_test_acc = 0
    num_no_increase = 0
    model.train()
    for epoch_idx in range( hyper.epoch_num ):
        if num_no_increase >= 5:
            print( 'early exit is triggered' )
            if args.save: torch.save( model, utils.create_model_file_name( args ) )
            return
        print(f'\nEpoch: {(epoch_idx + 1)}')
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, ( images, labels ) in enumerate( train_loader ):
            images, labels = images.to( args.device ), labels.to( args.device )
            optimizer.zero_grad()
            outputs = model( images )
            loss = gp.criterion( outputs, labels )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max( 1 )
            total += labels.size( 0 )
            correct += predicted.eq( labels ).sum().item()
            if batch_idx % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.5f |  Acc: %.3f%% (%d/%d)' %
                    (epoch_idx + 1, batch_idx + 1, train_loss / 2000, 100.*correct/total, correct, total))
                train_loss, total, correct = 0.0, 0, 0
        if epoch_idx % 5 == 4:
            print( f'begin middle test (best_test_acc = {best_test_acc}; num_no_increase = {num_no_increase}):' )
            current_test_acc = test_normal( model, args )
            if current_test_acc > best_test_acc:
                best_test_acc = current_test_acc
                num_no_increase = 0
            else:
                num_no_increase += 1
    test_normal( model, args, verbose=True )
    print( utils.create_model_file_name( args ) )
    if args.save: torch.save( model, utils.create_model_file_name( args ) )


def train_exits( args ):
    '''
    train the early-exits part with the original parts fixed
    '''
    # training setup
    hyper = gp.get_hyper( args )
    model = torch.load( args.pretrained_file )
    model = model.to( args.device )
    model.set_exit_layer( 'exits' )
    optimizer = gp.get_optimizer( params=model.parameters(), lr=hyper.learning_rate, op_type=args.optimizer )
    train_loader = gp.get_dataloader( args, task='train' )
    # fix the original parameters
    '''
    for name, parameter in model.named_parameters():
        if name in gp.get_layer_names( args ):
            parameter.requires_grad = False
    '''
    # begin training
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
            test_exits( model, args )
    test_exits( model, args )
    if args.save: torch.save( model, utils.create_model_file_name( args ) )


def test_exits( model, args ):
    '''
    test the accuracies of exit layers on test dataset
    '''
    if model.training:
        model.eval()
        train_flag = True
    else:
        train_flag = False
    correct_exit = [0 for i in range( 33 )]
    total = 0
    test_loader = gp.get_dataloader( args, task='test' )
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to( args.device ), labels.to( args.device )
            
            exit_tuple = model( images )
            exit_predicted_list = [None for i in range( len( exit_tuple ) )]
            for exit_idx in range( len( exit_tuple ) ):
                if len( exit_tuple[exit_idx].shape ) > 1:
                    _, exit_predicted_list[exit_idx] = exit_tuple[exit_idx].max( dim=1 )
                else:
                    _, exit_predicted_list[exit_idx] = exit_tuple[exit_idx].max( dim=0 )
            total += labels.size( 0 )
            for exit_idx in range( len( exit_tuple ) ):
                correct_exit[exit_idx] += exit_predicted_list[exit_idx].eq( labels ).sum().item()
        print( 'Accuracy of the network on the 10000 test images: \n', end='' )
        for exit_idx in range( len( exit_tuple ) ):
            print( 'exit'+str(exit_idx)+': %.3f%% (%d/%d)' % (100.*correct_exit[exit_idx]/total, correct_exit[exit_idx], total), end=' | ' )
        print( '' )
    if train_flag:
        model.train()

def simple_try():
    args = utils.Namespace( model_name='resnet',
                            pretrained_file='autodl-tmp/vgg19_train_exits_cifar100.pt',
                            optimizer='adam',
                            train_mode='original',
                            evaluate_mode='exits',
                            task='train',
                            device='cuda',
                            trained_file_suffix='cifar100',
                            beta=9,
                            save=0,
                            dataset_type = 'cifar100',
                            jump = 1
                            )
    
    train(args)

if __name__ == '__main__':
    simple_try()