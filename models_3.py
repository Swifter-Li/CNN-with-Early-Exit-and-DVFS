from torch import nn
from torch.nn import functional as F
import global_param as gp
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import utils_new as utils
from nikolaos.bof_utils import LogisticConvBoF
import time
total_round = 8
class ResNet_exits_eval(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet_exits_eval, self).__init__()
        self.exit_num = 32
        self.num_classes = num_classes
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
           # nn.MaxPool2d(kernel_size = (3,3)
           # The above line used particularly for stl10 model.
           )
        ########################################################################
        ########################################################################
        self.layer1_block1_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_block1_bn1 = nn.BatchNorm2d(64)
        self.layer1_block1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.layer1_block1_bn2 = nn.BatchNorm2d(64)
        ########################################################################
        self.layer1_block2_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_block2_bn1 = nn.BatchNorm2d(64)
        self.layer1_block2_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.layer1_block2_bn2 = nn.BatchNorm2d(64)
        ########################################################################
        self.layer1_block3_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_block3_bn1 = nn.BatchNorm2d(64)
        self.layer1_block3_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.layer1_block3_bn2 = nn.BatchNorm2d(64)
        ########################################################################
        ########################################################################
        self.layer2_block1_conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer2_block1_bn1 = nn.BatchNorm2d(128)
        self.layer2_block1_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.layer2_block1_bn2 = nn.BatchNorm2d(128)
        self.layer2_block1_shortcut = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)
        self.layer2_block1_bn3 = nn.BatchNorm2d(128)
        ########################################################################
        self.layer2_block2_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_block2_bn1 = nn.BatchNorm2d(128)
        self.layer2_block2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.layer2_block2_bn2 = nn.BatchNorm2d(128)
        ########################################################################
        self.layer2_block3_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_block3_bn1 = nn.BatchNorm2d(128)
        self.layer2_block3_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.layer2_block3_bn2 = nn.BatchNorm2d(128)
        ########################################################################
        self.layer2_block4_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_block4_bn1 = nn.BatchNorm2d(128)
        self.layer2_block4_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.layer2_block4_bn2 = nn.BatchNorm2d(128)
        ########################################################################
        ########################################################################
        self.layer3_block1_conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer3_block1_bn1 = nn.BatchNorm2d(256)
        self.layer3_block1_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.layer3_block1_bn2 = nn.BatchNorm2d(256)
        self.layer3_block1_shortcut = nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False)
        self.layer3_block1_bn3 = nn.BatchNorm2d(256)
        ########################################################################
        self.layer3_block2_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_block2_bn1 = nn.BatchNorm2d(256)
        self.layer3_block2_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.layer3_block2_bn2 = nn.BatchNorm2d(256)
        ########################################################################
        self.layer3_block3_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_block3_bn1 = nn.BatchNorm2d(256)
        self.layer3_block3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.layer3_block3_bn2 = nn.BatchNorm2d(256)
        ########################################################################
        self.layer3_block4_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_block4_bn1 = nn.BatchNorm2d(256)
        self.layer3_block4_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.layer3_block4_bn2 = nn.BatchNorm2d(256)
        ########################################################################
        self.layer3_block5_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_block5_bn1 = nn.BatchNorm2d(256)
        self.layer3_block5_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.layer3_block5_bn2 = nn.BatchNorm2d(256)
        ########################################################################
        self.layer3_block6_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_block6_bn1 = nn.BatchNorm2d(256)
        self.layer3_block6_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.layer3_block6_bn2 = nn.BatchNorm2d(256)
        ########################################################################
        ########################################################################
        self.layer4_block1_conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer4_block1_bn1 = nn.BatchNorm2d(512)
        self.layer4_block1_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.layer4_block1_bn2 = nn.BatchNorm2d(512)
        self.layer4_block1_shortcut = nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False)
        self.layer4_block1_bn3 = nn.BatchNorm2d(512)
        ########################################################################
        self.layer4_block2_conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_block2_bn1 = nn.BatchNorm2d(512)
        self.layer4_block2_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.layer4_block2_bn2 = nn.BatchNorm2d(512)
        ########################################################################
        self.layer4_block3_conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_block3_bn1 = nn.BatchNorm2d(512)
        self.layer4_block3_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.layer4_block3_bn2 = nn.BatchNorm2d(512)
        ########################################################################
        ########################################################################
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 , num_classes)
        ########################################################################
        self.block1_exit_0 = LogisticConvBoF(64, 64, split_horizon=16)
        self.block2_exit_0 = LogisticConvBoF(64, 64, split_horizon=16)
        self.block3_exit_0 = LogisticConvBoF(64, 64, split_horizon=16)

        self.block1_exit_1 = LogisticConvBoF(128, 64, split_horizon=8)
        self.block2_exit_1 = LogisticConvBoF(128, 64, split_horizon=8)
        self.block3_exit_1 = LogisticConvBoF(128, 64, split_horizon=8)
        self.block4_exit_1 = LogisticConvBoF(128, 64, split_horizon=8)

        self.block1_exit_2 = LogisticConvBoF(256, 64, split_horizon=4)
        self.block2_exit_2 = LogisticConvBoF(256, 64, split_horizon=4)
        self.block3_exit_2 = LogisticConvBoF(256, 64, split_horizon=4)
        self.block4_exit_2 = LogisticConvBoF(256, 64, split_horizon=4)
        self.block5_exit_2 = LogisticConvBoF(256, 64, split_horizon=4)
        self.block6_exit_2 = LogisticConvBoF(256, 64, split_horizon=4)
        
        self.block1_exit_3 = LogisticConvBoF(512, 64, split_horizon=2)
        self.block2_exit_3 = LogisticConvBoF(512, 64, split_horizon=2)
        self.block3_exit_3 = LogisticConvBoF(512, 64, split_horizon=2)

        self.layer1_exit_0_fc = nn.Linear(256,64)
        self.layer1_exit_1_fc = nn.Linear(64, num_classes)

        self.layer2_exit_0_fc = nn.Linear(256,64)
        self.layer2_exit_1_fc = nn.Linear(64, num_classes)

        self.layer3_exit_0_fc = nn.Linear(256,64)
        self.layer3_exit_1_fc = nn.Linear(64, num_classes)

        self.layer4_exit_0_fc = nn.Linear(256,64)
        self.layer4_exit_1_fc = nn.Linear(64, num_classes)

        self.activation_threshold_list = []
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
        self.accuracy_record = 0
        self.prediction_total = 0

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
        print( f'original: {100*self.original/total_inference:.3f}% ({self.original}/{total_inference})' )
    
    def set_beta( self, beta ):
        self.beta = beta
    
    def _calculate_max_activation( self, param ):
        '''
        return the maximum activation item in [param]
        '''
        return torch.max( torch.abs( torch.max( param ) ), torch.abs( torch.min( param ) ) )

    def set_possible_layer(self, temp):
        self.possible_layer = temp

    def get_specific_exit_number(self, iterate):
        return self.num_early_exit_list[iterate]

    def simple_conv1d(self, param):
        copy = param.copy()
        temp = []
        copy.insert(0,0)
        copy.append(0)
        for i in range(len(copy)-2):
            temp.append(copy[i]+copy[i+1]+copy[i+2])
        return temp

    def prediction(self, param, layer):
        round = total_round 
        temp = []
        for i in range(self.num_classes):
            temp.append(param[0][i].item())
        for i in range(round):
            temp = self.simple_conv1d(temp)
            length = len(self.activation_threshold_list)
            temp_thresold = self.activation_threshold_list[layer+i] if (layer+i) < length else self.activation_threshold_list[length-1]
            if max(abs(max(temp)), abs(min(temp))) > self.beta * temp_thresold:
                return i+1
        return round
    
    def settings( self, layer, forward_mode, p = 0 ):
        self.target_layer = layer
        self.start_layer = layer
        self.forward_mode = forward_mode
        self.quant_switch = p
        
        
    def output( self ):
        return (self.count_store, self.layer_store, self.accuracy_record, self.prediction_total) , self.jumpstep_store, self.prediction_store

    def forward( self, x):
        if self.forward_mode == 'accuracy_forward':
            return self.accuracy_forward(x)
        elif self.forward_mode == 'test_on_train_forward':
            return self.test_on_train_forward(x)
        elif self.forward_mode == 'test_on_train_normal_forward':
            return self.test_on_train_normal_forward(x)       
        else:
            return self.normal_forward(x)   

    def accuracy_forward(self, x):
        
        
        if self.quant_switch == 1:
            x = self.quant(x)
        x = self.pre(x) 
        temp2 = x
        x = F.relu(self.layer1_block1_bn1(self.layer1_block1_conv1(x)))
        current_layer = 1
        if self.target_layer == 1:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / self.beta * self.activation_threshold_list[0]
            if ratio >= 1:
                self.num_early_exit_list[0] += 1
                return 0, exit1
            self.prediction_store.append(self.prediction(exit1, current_layer))

        x = self.layer1_block1_bn2(self.layer1_block1_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 2
        if self.target_layer == 2:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[1])
            if ratio >= 1:
                self.num_early_exit_list[1] += 1
                return 1, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[1])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[1] += 1
                return 1, exit1       

        temp2 = x
        x = F.relu(self.layer1_block2_bn1(self.layer1_block2_conv1(x)))
        current_layer = 3
        if self.target_layer == 3:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[2])
            if ratio >= 1:
                self.num_early_exit_list[2] += 1
                return 2, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[2])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[2] += 1
                return 2, exit1   

        x = self.layer1_block2_bn2(self.layer1_block2_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 4
        if self.target_layer == 4:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[3])
            if ratio >= 1:
                self.num_early_exit_list[3] += 1
                return 3, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[3])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[3] += 1
                return 3, exit1 
        
        temp2 = x
        x = F.relu(self.layer1_block3_bn1(self.layer1_block3_conv1(x)))
        current_layer = 5
        if self.target_layer == 5:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[4])
            if ratio >= 1:
                self.num_early_exit_list[4] += 1
                return 4, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[4])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[4] += 1
                return 4, exit1 

        x = self.layer1_block3_bn2(self.layer1_block3_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 6
        if self.target_layer == 6:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            if ratio >= 1:
                self.num_early_exit_list[5] += 1
                return 5, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[5] += 1
                return 5, exit1 

        temp2 = self.layer2_block1_bn3(self.layer2_block1_shortcut(x))
        x = F.relu(self.layer2_block1_bn1(self.layer2_block1_conv1(x)))
        current_layer = 7
        if self.target_layer == 7:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[6])
            if ratio >= 1:
                self.num_early_exit_list[6] += 1
                return 6, exit1
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[6])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[6] += 1
                return 6, exit1 

        x = self.layer2_block1_bn2(self.layer2_block1_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 8
        if self.target_layer == 8:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[7])
            if ratio >= 1:
                self.num_early_exit_list[7] += 1
                return 7, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[7])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[7] += 1
                return 7, exit1 

        temp2 = x
        x = F.relu(self.layer2_block2_bn1(self.layer2_block2_conv1(x)))
        current_layer = 9
        if self.target_layer == 9:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[8])
            if ratio >= 1:
                self.num_early_exit_list[8] += 1
                return 8, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[8])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[8] += 1
                return 8, exit1 

        x = self.layer2_block2_bn2(self.layer2_block2_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 10
        if self.target_layer == 10:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[9])
            if ratio >= 1:
                self.num_early_exit_list[9] += 1
                return 9, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[9])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[9] += 1
                return 9, exit1 

        temp2 = x
        x = F.relu(self.layer2_block3_bn1(self.layer2_block3_conv1(x)))
        current_layer = 11
        if self.target_layer == 11:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
            if ratio >= 1:
                self.num_early_exit_list[10] += 1
                return 10, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[10] += 1
                return 10, exit1 

        x = self.layer2_block3_bn2(self.layer2_block3_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 12
        if self.target_layer == 12:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[11])
            if ratio >= 1:
                self.num_early_exit_list[11] += 1
                return 11, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[11])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[11] += 1
                return 11, exit1 

        temp2 = x
        x = F.relu(self.layer2_block4_bn1(self.layer2_block4_conv1(x)))
        current_layer = 13
        if self.target_layer == 13:
            if self.quant_switch == 1:
                x_exit_1 =  self.block4_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block4_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[12])
            if ratio >= 1:
                self.num_early_exit_list[12] += 1
                return 12, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block4_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block4_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[12])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[12] += 1
                return 12, exit1 

        x = self.layer2_block4_bn2(self.layer2_block4_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 14
        if self.target_layer == 14:
            if self.quant_switch == 1:
                x_exit_1 =  self.block4_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block4_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[13])
            if ratio >= 1:
                self.num_early_exit_list[13] += 1
                return 13, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block4_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block4_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[13])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[13] += 1
                return 13, exit1 


        temp2 = self.layer3_block1_bn3(self.layer3_block1_shortcut(x))
        x = F.relu(self.layer3_block1_bn1(self.layer3_block1_conv1(x)))
        current_layer = 15
        if self.target_layer == 15:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[14])
            if ratio >= 1:
                self.num_early_exit_list[14] += 1
                return 14, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[14])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[14] += 1
                return 14, exit1 

        x = self.layer3_block1_bn2(self.layer3_block1_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 16
        if self.target_layer == 16:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[15])
            if ratio >= 1:
                self.num_early_exit_list[15] += 1
                return 15, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[15])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[15] += 1
                return 15, exit1 

        temp2 = x
        x = F.relu(self.layer3_block2_bn1(self.layer3_block2_conv1(x)))
        current_layer = 17
        if self.target_layer == 17:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1 

        x = self.layer3_block2_bn2(self.layer3_block2_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 18
        if self.target_layer == 18:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1 

        temp2 = x
        x = F.relu(self.layer3_block3_bn1(self.layer3_block3_conv1(x)))
        current_layer = 19
        if self.target_layer == 19:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1 

        x = self.layer3_block3_bn2(self.layer3_block3_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 20
        if self.target_layer == 20:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1 
        
        temp2 = x
        x = F.relu(self.layer3_block4_bn1(self.layer3_block4_conv1(x)))
        current_layer = 21
        if self.target_layer == 21:
            if self.quant_switch == 1:
                x_exit_1 =  self.block4_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block4_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block4_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block4_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1 

        x = self.layer3_block4_bn2(self.layer3_block4_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 22
        if self.target_layer == 22:
            if self.quant_switch == 1:
                x_exit_1 =  self.block4_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block4_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block4_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block4_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1 

        temp2 = x
        x = F.relu(self.layer3_block5_bn1(self.layer3_block5_conv1(x)))
        current_layer = 23
        if self.target_layer == 23:
            if self.quant_switch == 1:
                x_exit_1 =  self.block5_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block5_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block5_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block5_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1 

        x = self.layer3_block5_bn2(self.layer3_block5_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 24
        if self.target_layer == 24:
            if self.quant_switch == 1:
                x_exit_1 =  self.block5_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block5_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block5_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block5_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1 

        temp2 = x
        x = F.relu(self.layer3_block6_bn1(self.layer3_block6_conv1(x)))
        current_layer = 25
        if self.target_layer == 25:
            if self.quant_switch == 1:
                x_exit_1 =  self.block6_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block6_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block6_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block6_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1 

        x = self.layer3_block6_bn2(self.layer3_block6_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 26
        if self.target_layer == 26:
            if self.quant_switch == 1:
                x_exit_1 =  self.block6_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block6_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block6_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block6_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1 


        temp2 = self.layer4_block1_bn3(self.layer4_block1_shortcut(x))
        x = F.relu(self.layer4_block1_bn1(self.layer4_block1_conv1(x)))
        current_layer = 27
        if self.target_layer == 27:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1 

        x = self.layer4_block1_bn2(self.layer4_block1_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 28
        if self.target_layer == 28:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1 

        temp2 = x
        x = F.relu(self.layer4_block2_bn1(self.layer4_block2_conv1(x)))
        current_layer = 29
        if self.target_layer == 29:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1 

        x = self.layer4_block2_bn2(self.layer4_block2_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 30
        if self.target_layer == 30:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1 

        temp2 = x
        x = F.relu(self.layer4_block3_bn1(self.layer4_block3_conv1(x)))
        current_layer = 31
        if self.target_layer == 31:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        x = self.layer4_block3_bn2(self.layer4_block3_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 32
        if self.target_layer == 32:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            #self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        x = self.avg_pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        if self.quant_switch == 1:
            x = self.dequant(x)
        self.original += 1
        self.jumpstep_store.append(current_layer-self.target_layer)
        return 32, x

    def normal_forward(self, x):
        count = 0
        self.target_layer = self.start_layer
    
        if self.quant_switch == 1:
            x = self.quant(x)
        x = self.pre(x)
        temp2 = x
        x = F.relu(self.layer1_block1_bn1(self.layer1_block1_conv1(x)))
        current_layer = 1
        if self.target_layer == 1:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / self.beta * self.activation_threshold_list[0]
            if ratio >= 1:
                self.num_early_exit_list[0] += 1
                if count == 1: 
                    self.accuracy_record += 1
                return 0, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        x = self.layer1_block1_bn2(self.layer1_block1_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 2
        if self.target_layer == 2:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[1])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[1] += 1
                return 1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1
            

        temp2 = x
        x = F.relu(self.layer1_block2_bn1(self.layer1_block2_conv1(x)))
        current_layer = 3
        if self.target_layer == 3:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[2])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[2] += 1
                return 2, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        x = self.layer1_block2_bn2(self.layer1_block2_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 4
        if self.target_layer == 4:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[3])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[3] += 1
                return 3, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1
            
        temp2 = x
        x = F.relu(self.layer1_block3_bn1(self.layer1_block3_conv1(x)))
        current_layer = 5
        if self.target_layer == 5:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[4])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[4] += 1
                return 4, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        x = self.layer1_block3_bn2(self.layer1_block3_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 6
        if self.target_layer == 6:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[5] += 1
                return 5, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        temp2 = self.layer2_block1_bn3(self.layer2_block1_shortcut(x))
        x = F.relu(self.layer2_block1_bn1(self.layer2_block1_conv1(x)))
        current_layer = 7
        if self.target_layer == 7:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[6])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[6] += 1
                return 6, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        x = self.layer2_block1_bn2(self.layer2_block1_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 8
        if self.target_layer == 8:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[7])
            time2 = time.perf_counter()
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[7] += 1
                return 7, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        temp2 = x
        x = F.relu(self.layer2_block2_bn1(self.layer2_block2_conv1(x)))
        current_layer = 9
        if self.target_layer == 9:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[8])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[8] += 1
                return 8, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        x = self.layer2_block2_bn2(self.layer2_block2_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 10
        if self.target_layer == 10:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[9])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        temp2 = x
        x = F.relu(self.layer2_block3_bn1(self.layer2_block3_conv1(x)))
        current_layer = 11
        if self.target_layer == 11:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        x = self.layer2_block3_bn2(self.layer2_block3_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 12
        if self.target_layer == 12:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[11])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        temp2 = x
        x = F.relu(self.layer2_block4_bn1(self.layer2_block4_conv1(x)))
        current_layer = 13
        if self.target_layer == 13:
            if self.quant_switch == 1:
                x_exit_1 =  self.block4_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block4_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[12])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        x = self.layer2_block4_bn2(self.layer2_block4_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 14
        if self.target_layer == 14:
            if self.quant_switch == 1:
                x_exit_1 =  self.block4_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block4_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[13])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1


        temp2 = self.layer3_block1_bn3(self.layer3_block1_shortcut(x))
        x = F.relu(self.layer3_block1_bn1(self.layer3_block1_conv1(x)))
        current_layer = 15
        if self.target_layer == 15:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[14])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        x = self.layer3_block1_bn2(self.layer3_block1_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 16
        if self.target_layer == 16:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[15])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        temp2 = x
        x = F.relu(self.layer3_block2_bn1(self.layer3_block2_conv1(x)))
        current_layer = 17
        if self.target_layer == 17:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1


        x = self.layer3_block2_bn2(self.layer3_block2_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 18
        if self.target_layer == 18:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        temp2 = x
        x = F.relu(self.layer3_block3_bn1(self.layer3_block3_conv1(x)))
        current_layer = 19
        if self.target_layer == 19:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1 

        x = self.layer3_block3_bn2(self.layer3_block3_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 20
        if self.target_layer == 20:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1
        
        temp2 = x
        x = F.relu(self.layer3_block4_bn1(self.layer3_block4_conv1(x)))
        current_layer = 21
        if self.target_layer == 21:
            if self.quant_switch == 1:
                x_exit_1 =  self.block4_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block4_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        x = self.layer3_block4_bn2(self.layer3_block4_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 22
        if self.target_layer == 22:
            if self.quant_switch == 1:
                x_exit_1 =  self.block4_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block4_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        temp2 = x
        x = F.relu(self.layer3_block5_bn1(self.layer3_block5_conv1(x)))
        current_layer = 23
        if self.target_layer == 23:
            if self.quant_switch == 1:
                x_exit_1 =  self.block5_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block5_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        x = self.layer3_block5_bn2(self.layer3_block5_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 24
        if self.target_layer == 24:
            if self.quant_switch == 1:
                x_exit_1 =  self.block5_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block5_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        temp2 = x
        x = F.relu(self.layer3_block6_bn1(self.layer3_block6_conv1(x)))
        current_layer = 25
        if self.target_layer == 25:
            if self.quant_switch == 1:
                x_exit_1 =  self.block6_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block6_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        x = self.layer3_block6_bn2(self.layer3_block6_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 26
        if self.target_layer == 26:
            if self.quant_switch == 1:
                x_exit_1 =  self.block6_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block6_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1


        temp2 = self.layer4_block1_bn3(self.layer4_block1_shortcut(x))
        x = F.relu(self.layer4_block1_bn1(self.layer4_block1_conv1(x)))
        current_layer = 27
        if self.target_layer == 27:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        x = self.layer4_block1_bn2(self.layer4_block1_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 28
        if self.target_layer == 28:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        temp2 = x
        x = F.relu(self.layer4_block2_bn1(self.layer4_block2_conv1(x)))
        current_layer = 29
        if self.target_layer == 29:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        x = self.layer4_block2_bn2(self.layer4_block2_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 30
        if self.target_layer == 30:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        temp2 = x
        x = F.relu(self.layer4_block3_bn1(self.layer4_block3_conv1(x)))
        current_layer = 31
        if self.target_layer == 31:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        x = self.layer4_block3_bn2(self.layer4_block3_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 32
        if self.target_layer == 32:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                if count == 1: 
                    self.accuracy_record += 1
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if count == 0:
                    self.prediction_total += 1
                count += 1

        x = self.avg_pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        self.original += 1
        if self.quant_switch == 1:
            x = self.dequant(x)
        
        return 32, x

    def test_on_train_forward(self, x):
        
        
        if self.quant_switch == 1:
            x = self.quant(x)
        x = self.pre(x)
        temp2 = x
        x = F.relu(self.layer1_block1_bn1(self.layer1_block1_conv1(x)))
        current_layer = 1
        if self.target_layer == 1:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / self.beta * self.activation_threshold_list[0]
            if ratio >= 1:
                self.num_early_exit_list[0] += 1
                return 0, exit1

        x = self.layer1_block1_bn2(self.layer1_block1_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 2
        if self.target_layer == 2:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[1])
            if ratio >= 1:
                self.num_early_exit_list[1] += 1
                return 1, exit1   

        temp2 = x
        x = F.relu(self.layer1_block2_bn1(self.layer1_block2_conv1(x)))
        current_layer = 3
        if self.target_layer == 3:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[2])
            if ratio >= 1:
                self.num_early_exit_list[2] += 1
                return 2, exit1 

        x = self.layer1_block2_bn2(self.layer1_block2_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 4
        if self.target_layer == 4:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[3])
            if ratio >= 1:
                self.num_early_exit_list[3] += 1
                return 3, exit1

        temp2 = x
        x = F.relu(self.layer1_block3_bn1(self.layer1_block3_conv1(x)))
        current_layer = 5
        if self.target_layer == 5:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[4])
            if ratio >= 1:
                self.num_early_exit_list[4] += 1
                return 4, exit1

        x = self.layer1_block3_bn2(self.layer1_block3_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 6
        if self.target_layer == 6:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            if ratio >= 1:
                self.num_early_exit_list[5] += 1
                return 5, exit1

        temp2 = self.layer2_block1_bn3(self.layer2_block1_shortcut(x))
        x = F.relu(self.layer2_block1_bn1(self.layer2_block1_conv1(x)))
        current_layer = 7
        if self.target_layer == 7:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[6])
            if ratio >= 1:
                self.num_early_exit_list[6] += 1
                return 6, exit1

        x = self.layer2_block1_bn2(self.layer2_block1_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 8
        if self.target_layer == 8:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[7])
            if ratio >= 1:
                self.num_early_exit_list[7] += 1
                return 7, exit1

        temp2 = x
        x = F.relu(self.layer2_block2_bn1(self.layer2_block2_conv1(x)))
        current_layer = 9
        if self.target_layer == 9:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[8])
            if ratio >= 1:
                self.num_early_exit_list[8] += 1
                return 8, exit1

        x = self.layer2_block2_bn2(self.layer2_block2_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 10
        if self.target_layer == 10:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[9])
            if ratio >= 1:
                self.num_early_exit_list[9] += 1
                return 9, exit1

        temp2 = x
        x = F.relu(self.layer2_block3_bn1(self.layer2_block3_conv1(x)))
        current_layer = 11
        if self.target_layer == 11:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
            if ratio >= 1:
                self.num_early_exit_list[10] += 1
                return 10, exit1

        x = self.layer2_block3_bn2(self.layer2_block3_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 12
        if self.target_layer == 12:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[11])
            if ratio >= 1:
                self.num_early_exit_list[11] += 1
                return 11, exit1

        temp2 = x
        x = F.relu(self.layer2_block4_bn1(self.layer2_block4_conv1(x)))
        current_layer = 13
        if self.target_layer == 13:
            if self.quant_switch == 1:
                x_exit_1 =  self.block4_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block4_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[12])
            if ratio >= 1:
                self.num_early_exit_list[12] += 1
                return 12, exit1

        x = self.layer2_block4_bn2(self.layer2_block4_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 14
        if self.target_layer == 14:
            if self.quant_switch == 1:
                x_exit_1 =  self.block4_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block4_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[13])
            if ratio >= 1:
                self.num_early_exit_list[13] += 1
                return 13, exit1

        temp2 = self.layer3_block1_bn3(self.layer3_block1_shortcut(x))
        x = F.relu(self.layer3_block1_bn1(self.layer3_block1_conv1(x)))
        current_layer = 15
        if self.target_layer == 15:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[14])
            if ratio >= 1:
                self.num_early_exit_list[14] += 1
                return 14, exit1

        x = self.layer3_block1_bn2(self.layer3_block1_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 16
        if self.target_layer == 16:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[15])
            if ratio >= 1:
                self.num_early_exit_list[15] += 1
                return 15, exit1

        temp2 = x
        x = F.relu(self.layer3_block2_bn1(self.layer3_block2_conv1(x)))
        current_layer = 17
        if self.target_layer == 17:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        x = self.layer3_block2_bn2(self.layer3_block2_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 18
        if self.target_layer == 18:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        temp2 = x
        x = F.relu(self.layer3_block3_bn1(self.layer3_block3_conv1(x)))
        current_layer = 19
        if self.target_layer == 19:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        x = self.layer3_block3_bn2(self.layer3_block3_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 20
        if self.target_layer == 20:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
        
        temp2 = x
        x = F.relu(self.layer3_block4_bn1(self.layer3_block4_conv1(x)))
        current_layer = 21
        if self.target_layer == 21:
            if self.quant_switch == 1:
                x_exit_1 =  self.block4_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block4_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        x = self.layer3_block4_bn2(self.layer3_block4_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 22
        if self.target_layer == 22:
            if self.quant_switch == 1:
                x_exit_1 =  self.block4_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block4_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        temp2 = x
        x = F.relu(self.layer3_block5_bn1(self.layer3_block5_conv1(x)))
        current_layer = 23
        if self.target_layer == 23:
            if self.quant_switch == 1:
                x_exit_1 =  self.block5_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block5_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        x = self.layer3_block5_bn2(self.layer3_block5_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 24
        if self.target_layer == 24:
            if self.quant_switch == 1:
                x_exit_1 =  self.block5_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block5_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        temp2 = x
        x = F.relu(self.layer3_block6_bn1(self.layer3_block6_conv1(x)))
        current_layer = 25
        if self.target_layer == 25:
            if self.quant_switch == 1:
                x_exit_1 =  self.block6_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block6_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        x = self.layer3_block6_bn2(self.layer3_block6_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 26
        if self.target_layer == 26:
            if self.quant_switch == 1:
                x_exit_1 =  self.block6_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block6_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1


        temp2 = self.layer4_block1_bn3(self.layer4_block1_shortcut(x))
        x = F.relu(self.layer4_block1_bn1(self.layer4_block1_conv1(x)))
        current_layer = 27
        if self.target_layer == 27:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        x = self.layer4_block1_bn2(self.layer4_block1_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 28
        if self.target_layer == 28:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        temp2 = x
        x = F.relu(self.layer4_block2_bn1(self.layer4_block2_conv1(x)))
        current_layer = 29
        if self.target_layer == 29:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        x = self.layer4_block2_bn2(self.layer4_block2_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 30
        if self.target_layer == 30:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        temp2 = x
        x = F.relu(self.layer4_block3_bn1(self.layer4_block3_conv1(x)))
        current_layer = 31
        if self.target_layer == 31:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        x = self.layer4_block3_bn2(self.layer4_block3_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 32
        if self.target_layer == 32:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        x = self.avg_pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        if self.quant_switch == 1:
            x = self.dequant(x)
        self.original += 1
        return 32, x

    def test_on_train_normal_forward(self, x):
        self.target_layer = self.start_layer
    
        if self.quant_switch == 1:
            x = self.quant(x)
        x = self.pre(x)
        temp2 = x
        x = F.relu(self.layer1_block1_bn1(self.layer1_block1_conv1(x)))
        current_layer = 1
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            self.count_store[3] += 32*64*64*32 + 256*127 + 64*19
            ratio = self._calculate_max_activation( exit1 ) / self.beta * self.activation_threshold_list[0]
            if ratio >= 1:
                self.num_early_exit_list[0] += 1
                return 0, exit1
            
        x = self.layer1_block1_bn2(self.layer1_block1_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 2
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[1])
            if ratio >= 1:
                self.num_early_exit_list[1] += 1
                return 1, exit1
            

        temp2 = x
        x = F.relu(self.layer1_block2_bn1(self.layer1_block2_conv1(x)))
        current_layer = 3
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[2])
            if ratio >= 1:
                self.num_early_exit_list[2] += 1
                return 2, exit1

        x = self.layer1_block2_bn2(self.layer1_block2_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 4
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[3])
            if ratio >= 1:
                self.num_early_exit_list[3] += 1
                return 3, exit1
            
        temp2 = x
        x = F.relu(self.layer1_block3_bn1(self.layer1_block3_conv1(x)))
        current_layer = 5
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[4])
            if ratio >= 1:
                self.num_early_exit_list[4] += 1
                return 4, exit1

        x = self.layer1_block3_bn2(self.layer1_block3_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 6
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_0(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_0(x)
            x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
            exit1 = self.layer1_exit_1_fc(x_exit_1)
            self.count_store[3] += 32*64*64*32 + 256*127 + 64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            if ratio >= 1:
                self.num_early_exit_list[5] += 1
                return 5, exit1

        temp2 = self.layer2_block1_bn3(self.layer2_block1_shortcut(x))
        x = F.relu(self.layer2_block1_bn1(self.layer2_block1_conv1(x)))
        current_layer = 7
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[6])
            if ratio >= 1:
                self.num_early_exit_list[6] += 1
                return 6, exit1

        x = self.layer2_block1_bn2(self.layer2_block1_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 8
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            self.count_store[3] += 32*32*128*64 + 127*64+ 64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[7])
            if ratio >= 1:
                self.num_early_exit_list[7] += 1
                return 7, exit1

        temp2 = x
        x = F.relu(self.layer2_block2_bn1(self.layer2_block2_conv1(x)))
        current_layer = 9
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            self.count_store[3] += 32*32*128*64 + 127*64+ 64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[8])
            if ratio >= 1:
                self.num_early_exit_list[8] += 1
                return 8, exit1

        x = self.layer2_block2_bn2(self.layer2_block2_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 10
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            self.count_store[3] += 32*32*128*64 + 127*64+ 64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[9])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        temp2 = x
        x = F.relu(self.layer2_block3_bn1(self.layer2_block3_conv1(x)))
        self.count_store[3] += 128*32*128*32
        current_layer = 11
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            self.count_store[3] += 32*32*128*64 + 127*64+ 64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        x = self.layer2_block3_bn2(self.layer2_block3_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 12
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            self.count_store[3] += 32*32*128*64 + 127*64+ 64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[11])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        temp2 = x
        x = F.relu(self.layer2_block4_bn1(self.layer2_block4_conv1(x)))
        current_layer = 13
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block4_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block4_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[12])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        x = self.layer2_block4_bn2(self.layer2_block4_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 14
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block4_exit_1(self.dequant(x))
            else:
                x_exit_1 = self.block4_exit_1(x)
            x_exit_1 = self.layer2_exit_0_fc(x_exit_1)
            exit1 = self.layer2_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[13])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1


        temp2 = self.layer3_block1_bn3(self.layer3_block1_shortcut(x))
        x = F.relu(self.layer3_block1_bn1(self.layer3_block1_conv1(x)))
        current_layer = 15
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[14])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        x = self.layer3_block1_bn2(self.layer3_block1_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 16
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            self.count_store[3] += 32*32*256*64 + 256 * (2*64-1) + 64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[15])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        temp2 = x
        x = F.relu(self.layer3_block2_bn1(self.layer3_block2_conv1(x)))
        current_layer = 17
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        x = self.layer3_block2_bn2(self.layer3_block2_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 18
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        temp2 = x
        x = F.relu(self.layer3_block3_bn1(self.layer3_block3_conv1(x)))
        current_layer = 19
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        x = self.layer3_block3_bn2(self.layer3_block3_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 20
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1
        
        temp2 = x
        x = F.relu(self.layer3_block4_bn1(self.layer3_block4_conv1(x)))
        current_layer = 21
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block4_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block4_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        x = self.layer3_block4_bn2(self.layer3_block4_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 22
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block4_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block4_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            self.count_store[3] += 32*32*256*64 + 256 * (2*64-1) + 64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        temp2 = x
        x = F.relu(self.layer3_block5_bn1(self.layer3_block5_conv1(x)))
        current_layer = 23
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block5_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block5_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        x = self.layer3_block5_bn2(self.layer3_block5_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 24
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block5_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block5_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        temp2 = x
        x = F.relu(self.layer3_block6_bn1(self.layer3_block6_conv1(x)))
        current_layer = 25
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block6_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block6_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        x = self.layer3_block6_bn2(self.layer3_block6_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 26
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block6_exit_2(self.dequant(x))
            else:
                x_exit_1 = self.block6_exit_2(x)
            x_exit_1 = self.layer3_exit_0_fc(x_exit_1)
            exit1 = self.layer3_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1


        temp2 = self.layer4_block1_bn3(self.layer4_block1_shortcut(x))
        x = F.relu(self.layer4_block1_bn1(self.layer4_block1_conv1(x)))
        current_layer = 27
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        x = self.layer4_block1_bn2(self.layer4_block1_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 28
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block1_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block1_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            self.count_store[3] += 512 * (2*64-1) + 64 * 19 + 32*32*512*64
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        temp2 = x
        x = F.relu(self.layer4_block2_bn1(self.layer4_block2_conv1(x)))
        current_layer = 29
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        x = self.layer4_block2_bn2(self.layer4_block2_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 30
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block2_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block2_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        temp2 = x
        x = F.relu(self.layer4_block3_bn1(self.layer4_block3_conv1(x)))
        current_layer = 31
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        x = self.layer4_block3_bn2(self.layer4_block3_conv2(x))
        if self.quant_switch == 1:
            x = self.dequant(x)
            temp2 = self.dequant(temp2)
        x = F.relu(x + temp2)
        if self.quant_switch == 1:
            x = self.quant(x)
        current_layer = 32
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.block3_exit_3(self.dequant(x))
            else:
                x_exit_1 = self.block3_exit_3(x)
            x_exit_1 = self.layer4_exit_0_fc(x_exit_1)
            exit1 = self.layer4_exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[current_layer-1])
            if ratio >= 1:
                self.num_early_exit_list[current_layer-1] += 1
                return current_layer-1, exit1

        x = self.avg_pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        self.original += 1
        if self.quant_switch == 1:
            x = self.dequant(x)
        return 32, x

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
class ResNet_exits_train(ResNet_exits_eval):
    def __init__(self, num_classes):
        super().__init__(num_classes)
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
    
    def forward_exits( self, x ):
        x = self.pre(x)
        temp2 = x
        x = F.relu(self.layer1_block1_bn1(self.layer1_block1_conv1(x)))
        x_exit_1 = self.block1_exit_0(x)
        x_exit_1 = self.layer1_exit_0_fc(x_exit_1)
        exit1 = self.layer1_exit_1_fc(x_exit_1)

        x = F.relu(self.layer1_block1_bn2(self.layer1_block1_conv2(x))+temp2)
        x_exit_2 = self.block1_exit_0(x)
        x_exit_2 = self.layer1_exit_0_fc(x_exit_2)
        exit2 = self.layer1_exit_1_fc(x_exit_2)

        temp2 = x
        x = F.relu(self.layer1_block2_bn1(self.layer1_block2_conv1(x)))
        x_exit_3 = self.block2_exit_0(x)
        x_exit_3 = self.layer1_exit_0_fc(x_exit_3)
        exit3 = self.layer1_exit_1_fc(x_exit_3)

        x = F.relu(self.layer1_block2_bn2(self.layer1_block2_conv2(x))+temp2)
        x_exit_4 = self.block2_exit_0(x)
        x_exit_4 = self.layer1_exit_0_fc(x_exit_4)
        exit4 = self.layer1_exit_1_fc(x_exit_4)
        
        temp2 = x
        x = F.relu(self.layer1_block3_bn1(self.layer1_block3_conv1(x)))
        x_exit_5 = self.block3_exit_0(x)
        x_exit_5 = self.layer1_exit_0_fc(x_exit_5)
        exit5 = self.layer1_exit_1_fc(x_exit_5)

        x = F.relu(self.layer1_block3_bn2(self.layer1_block3_conv2(x))+temp2)
        x_exit_6 = self.block3_exit_0(x)
        x_exit_6 = self.layer1_exit_0_fc(x_exit_6)
        exit6 = self.layer1_exit_1_fc(x_exit_6)

        temp2 = self.layer2_block1_bn3(self.layer2_block1_shortcut(x))
        x = F.relu(self.layer2_block1_bn1(self.layer2_block1_conv1(x)))
        x_exit_7 = self.block1_exit_1(x)
        x_exit_7 = self.layer2_exit_0_fc(x_exit_7)
        exit7 = self.layer2_exit_1_fc(x_exit_7)

        x = self.layer2_block1_bn2(self.layer2_block1_conv2(x))
        x = F.relu(x + temp2)
        x_exit_8 = self.block1_exit_1(x)
        x_exit_8 = self.layer2_exit_0_fc(x_exit_8)
        exit8 = self.layer2_exit_1_fc(x_exit_8)

        temp2 = x
        x = F.relu(self.layer2_block2_bn1(self.layer2_block2_conv1(x)))
        x_exit_9 = self.block2_exit_1(x)
        x_exit_9 = self.layer2_exit_0_fc(x_exit_9)
        exit9 = self.layer2_exit_1_fc(x_exit_9)

        x = F.relu(self.layer2_block2_bn2(self.layer2_block2_conv2(x))+temp2)
        x_exit_10 = self.block2_exit_1(x)
        x_exit_10 = self.layer2_exit_0_fc(x_exit_10)
        exit10 = self.layer2_exit_1_fc(x_exit_10)

        temp2 = x
        x = F.relu(self.layer2_block3_bn1(self.layer2_block3_conv1(x)))
        x_exit_11 = self.block3_exit_1(x)
        x_exit_11 = self.layer2_exit_0_fc(x_exit_11)
        exit11 = self.layer2_exit_1_fc(x_exit_11)

        x = F.relu(self.layer2_block3_bn2(self.layer2_block3_conv2(x))+temp2)
        x_exit_12 = self.block3_exit_1(x)
        x_exit_12 = self.layer2_exit_0_fc(x_exit_12)
        exit12 = self.layer2_exit_1_fc(x_exit_12)

        temp2 = x
        x = F.relu(self.layer2_block4_bn1(self.layer2_block4_conv1(x)))
        x_exit_13 = self.block4_exit_1(x)
        x_exit_13 = self.layer2_exit_0_fc(x_exit_13)
        exit13 = self.layer2_exit_1_fc(x_exit_13)

        x = F.relu(self.layer2_block4_bn2(self.layer2_block4_conv2(x))+ temp2)
        x_exit_14 = self.block4_exit_1(x)
        x_exit_14 = self.layer2_exit_0_fc(x_exit_14)
        exit14 = self.layer2_exit_1_fc(x_exit_14)

        temp2 = self.layer3_block1_bn3(self.layer3_block1_shortcut(x))
        x = F.relu(self.layer3_block1_bn1(self.layer3_block1_conv1(x)))
        x_exit_15 = self.block1_exit_2(x)
        x_exit_15 = self.layer3_exit_0_fc(x_exit_15)
        exit15 = self.layer3_exit_1_fc(x_exit_15)


        x = self.layer3_block1_bn2(self.layer3_block1_conv2(x))
        x = F.relu(x + temp2)
        x_exit_16 = self.block1_exit_2(x)
        x_exit_16 = self.layer3_exit_0_fc(x_exit_16)
        exit16 = self.layer3_exit_1_fc(x_exit_16)

        temp2 = x
        x = F.relu(self.layer3_block2_bn1(self.layer3_block2_conv1(x)))
        x_exit_17 = self.block2_exit_2(x)
        x_exit_17 = self.layer3_exit_0_fc(x_exit_17)
        exit17 = self.layer3_exit_1_fc(x_exit_17)

        x = F.relu(self.layer3_block2_bn2(self.layer3_block2_conv2(x))+temp2)
        x_exit_18 = self.block2_exit_2(x)
        x_exit_18 = self.layer3_exit_0_fc(x_exit_18)
        exit18 = self.layer3_exit_1_fc(x_exit_18)

        temp2 = x
        x = F.relu(self.layer3_block3_bn1(self.layer3_block3_conv1(x)))
        x_exit_19 = self.block3_exit_2(x)
        x_exit_19 = self.layer3_exit_0_fc(x_exit_19)
        exit19 = self.layer3_exit_1_fc(x_exit_19)

        x = F.relu(self.layer3_block3_bn2(self.layer3_block3_conv2(x))+temp2)
        x_exit_20 = self.block3_exit_2(x)
        x_exit_20 = self.layer3_exit_0_fc(x_exit_20)
        exit20 = self.layer3_exit_1_fc(x_exit_20)
        
        temp2 = x
        x = F.relu(self.layer3_block4_bn1(self.layer3_block4_conv1(x)))
        x_exit_21 = self.block4_exit_2(x)
        x_exit_21 = self.layer3_exit_0_fc(x_exit_21)
        exit21 = self.layer3_exit_1_fc(x_exit_21)

        x = F.relu(self.layer3_block4_bn2(self.layer3_block4_conv2(x))+temp2)
        x_exit_22 = self.block4_exit_2(x)
        x_exit_22 = self.layer3_exit_0_fc(x_exit_22)
        exit22 = self.layer3_exit_1_fc(x_exit_22)

        temp2 = x
        x = F.relu(self.layer3_block5_bn1(self.layer3_block5_conv1(x)))
        x_exit_23 = self.block5_exit_2(x)
        x_exit_23 = self.layer3_exit_0_fc(x_exit_23)
        exit23 = self.layer3_exit_1_fc(x_exit_23)

        x = F.relu(self.layer3_block5_bn2(self.layer3_block5_conv2(x))+temp2)
        x_exit_24 = self.block5_exit_2(x)
        x_exit_24 = self.layer3_exit_0_fc(x_exit_24)
        exit24 = self.layer3_exit_1_fc(x_exit_24)

        temp2 = x
        x = F.relu(self.layer3_block6_bn1(self.layer3_block6_conv1(x)))
        x_exit_25 = self.block6_exit_2(x)
        x_exit_25 = self.layer3_exit_0_fc(x_exit_25)
        exit25 = self.layer3_exit_1_fc(x_exit_25)

        x = F.relu(self.layer3_block6_bn2(self.layer3_block6_conv2(x))+temp2)
        x_exit_26 = self.block6_exit_2(x)
        x_exit_26 = self.layer3_exit_0_fc(x_exit_26)
        exit26 = self.layer3_exit_1_fc(x_exit_26)

        temp2 = self.layer4_block1_bn3(self.layer4_block1_shortcut(x))
        x = F.relu(self.layer4_block1_bn1(self.layer4_block1_conv1(x)))
        x_exit_27 = self.block1_exit_3(x)
        x_exit_27 = self.layer4_exit_0_fc(x_exit_27)
        exit27 = self.layer4_exit_1_fc(x_exit_27)

        x = self.layer4_block1_bn2(self.layer4_block1_conv2(x))
        x = F.relu(x + temp2)
        x_exit_28 = self.block1_exit_3(x)
        x_exit_28 = self.layer4_exit_0_fc(x_exit_28)
        exit28 = self.layer4_exit_1_fc(x_exit_28)

        temp2 = x
        x = F.relu(self.layer4_block2_bn1(self.layer4_block2_conv1(x)))
        x_exit_29 = self.block2_exit_3(x)
        x_exit_29 = self.layer4_exit_0_fc(x_exit_29)
        exit29 = self.layer4_exit_1_fc(x_exit_29)

        x = F.relu(self.layer4_block2_bn2(self.layer4_block2_conv2(x))+temp2)
        x_exit_30 = self.block2_exit_3(x)
        x_exit_30 = self.layer4_exit_0_fc(x_exit_30)
        exit30 = self.layer4_exit_1_fc(x_exit_30)

        temp2 = x
        x = F.relu(self.layer4_block3_bn1(self.layer4_block3_conv1(x)))
        x_exit_31 = self.block3_exit_3(x)
        x_exit_31 = self.layer4_exit_0_fc(x_exit_31)
        exit31 = self.layer4_exit_1_fc(x_exit_31)

        x = F.relu(self.layer4_block3_bn2(self.layer4_block3_conv2(x))+temp2)
        x_exit_32 = self.block3_exit_3(x)
        x_exit_32 = self.layer4_exit_0_fc(x_exit_32)
        exit32 = self.layer4_exit_1_fc(x_exit_32)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return (exit1, exit2, exit3, exit4, exit5, exit6, exit7, exit8, exit9, exit10, exit11, exit12, exit13, exit14,\
                 exit15, exit16, exit17, exit18, exit19, exit20, exit21, exit22, exit23, exit24, exit25, \
                  exit26, exit27, exit28, exit29 ,exit30, exit31, exit32, x)

    def forward_original(self,x):
        x = self.pre(x)

        temp2 = x
        x = F.relu(self.layer1_block1_bn1(self.layer1_block1_conv1(x)))
        x = F.relu(self.layer1_block1_bn2(self.layer1_block1_conv2(x))+temp2)

        temp2 = x
        x = F.relu(self.layer1_block2_bn1(self.layer1_block2_conv1(x)))
        x = F.relu(self.layer1_block2_bn2(self.layer1_block2_conv2(x))+temp2)
        
        temp2 = x
        x = F.relu(self.layer1_block3_bn1(self.layer1_block3_conv1(x)))
        x = F.relu(self.layer1_block3_bn2(self.layer1_block3_conv2(x))+temp2)

        temp2 = self.layer2_block1_bn3(self.layer2_block1_shortcut(x))
        x = F.relu(self.layer2_block1_bn1(self.layer2_block1_conv1(x)))
        x = self.layer2_block1_bn2(self.layer2_block1_conv2(x))
        x = F.relu(x + temp2)

        temp2 = x
        x = F.relu(self.layer2_block2_bn1(self.layer2_block2_conv1(x)))
        x = F.relu(self.layer2_block2_bn2(self.layer2_block2_conv2(x))+temp2)

        temp2 = x
        x = F.relu(self.layer2_block3_bn1(self.layer2_block3_conv1(x)))
        x = F.relu(self.layer2_block3_bn2(self.layer2_block3_conv2(x))+temp2)

        temp2 = x
        x = F.relu(self.layer2_block4_bn1(self.layer2_block4_conv1(x)))
        x = F.relu(self.layer2_block4_bn2(self.layer2_block4_conv2(x))+temp2)

        temp2 = self.layer3_block1_bn3(self.layer3_block1_shortcut(x))
        x = F.relu(self.layer3_block1_bn1(self.layer3_block1_conv1(x)))
        x = self.layer3_block1_bn2(self.layer3_block1_conv2(x))
        x = F.relu(x + temp2)

        temp2 = x
        x = F.relu(self.layer3_block2_bn1(self.layer3_block2_conv1(x)))
        x = F.relu(self.layer3_block2_bn2(self.layer3_block2_conv2(x))+temp2)

        temp2 = x
        x = F.relu(self.layer3_block3_bn1(self.layer3_block3_conv1(x)))
        x = F.relu(self.layer3_block3_bn2(self.layer3_block3_conv2(x))+temp2)
        
        temp2 = x
        x = F.relu(self.layer3_block4_bn1(self.layer3_block4_conv1(x)))
        x = F.relu(self.layer3_block4_bn2(self.layer3_block4_conv2(x))+temp2)

        temp2 = x
        x = F.relu(self.layer3_block5_bn1(self.layer3_block5_conv1(x)))
        x = F.relu(self.layer3_block5_bn2(self.layer3_block5_conv2(x))+temp2)

        temp2 = x
        x = F.relu(self.layer3_block6_bn1(self.layer3_block6_conv1(x)))
        x = F.relu(self.layer3_block6_bn2(self.layer3_block6_conv2(x))+temp2)

        temp2 = self.layer4_block1_bn3(self.layer4_block1_shortcut(x))
        x = F.relu(self.layer4_block1_bn1(self.layer4_block1_conv1(x)))
        x = self.layer4_block1_bn2(self.layer4_block1_conv2(x))
        x = F.relu(x + temp2)

        temp2 = x
        x = F.relu(self.layer4_block2_bn1(self.layer4_block2_conv1(x)))
        x = F.relu(self.layer4_block2_bn2(self.layer4_block2_conv2(x))+temp2)

        temp2 = x
        x = F.relu(self.layer4_block3_bn1(self.layer4_block3_conv1(x)))
        x = F.relu(self.layer4_block3_bn2(self.layer4_block3_conv2(x))+temp2)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x