import torch.nn as nn
import torch.nn.functional as F
import torch.nn.quantized.functional as F_quantized
import torch
from nikolaos.bof_utils import LogisticConvBoF
import time
Jump_step = [3,4,5]
data_type = 10
class vgg19_exits_eval_jump_mnist( nn.Module ):
    def __init__(self):
        super(vgg19_exits_eval_jump_mnist, self).__init__()
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
        self.exit_0_fc = nn.Linear(576,64)
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
        self.target_layer = 4

        self.ratio_store = []
        self.jumpstep_store = []
        self.entropy_store = []
    
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
    
    def output( self ):
        return self.ratio_store, self.jumpstep_store, self.entropy_store
       # return self.ratio_store, self.jumpstep_store

    def _calculate_cross_entropy(self, param):

        A = torch.abs(param)
        temp = torch.sum(A[0])
        for i in range(data_type):
            A[0][i] /= temp
        a = -((A * torch.log2(A)).sum())
        return a 

    '''
    def forward( self, x ):
        flag = 0

        x = F.relu(self.bn1(self.conv1(x)))
        current_layer = 1
        if self.target_layer == 1:
            x_exit_1 = self.exit_0(x)
  #   #   x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[0])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[0] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 0, exit1
            self.ratio_store.append(ratio.item())
            self.entropy_store.append(self._calculate_cross_entropy(exit1).item())
            self.total[flag] = self.total[flag] + 1
        else:
            x_exit_1 = self.exit_0(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[0])
            if ratio >= 1:
                self.statics[flag][current_layer-self.target_layer]= self.statics[flag][current_layer-self.target_layer] + 1
                self.num_early_exit_list[0] += 1
                return 0, exit1

        x = F.relu(self.bn2(self.conv2(x)))
        current_layer = 2
        if self.target_layer == 2:
            x_exit_1 = self.exit_0(x)
  #   #   x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[1])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[1] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 1, exit1
            self.total[flag] = self.total[flag] + 1
        else:
            x_exit_1 = self.exit_0(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[1])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.statics[flag][current_layer-self.target_layer]= self.statics[flag][current_layer-self.target_layer] + 1
                self.num_early_exit_list[1] += 1
                return 1, exit1

        x = F.max_pool2d(x, 1, 1)
        current_layer = 3
        if self.target_layer == 3:
            x_exit_1 = self.exit_0(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[2])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[2] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 2, exit1
            self.total[flag] = self.total[flag] + 1
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
        x_exit_1 = self.exit_1(x)
        x_exit_1 = self.exit_0_fc(x_exit_1)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[3])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]= self.statics[flag][current_layer-self.target_layer] + 1
            self.num_early_exit_list[3] += 1
            return 3, exit1

        x = F.relu(self.bn4(self.conv4(x)))
        current_layer = 5
        x_exit_1 = self.exit_1(x)
        x_exit_1 = self.exit_0_fc(x_exit_1)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[4])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]= self.statics[flag][current_layer-self.target_layer] + 1
            self.num_early_exit_list[4] += 1
            return 4, exit1

        x = F.max_pool2d(x, 2, 2)
        current_layer = 6
        if self.target_layer == 6:
            x_exit_1 = self.exit_1(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[5] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 5, exit1
            self.total[flag] = self.total[flag] + 1
        else:
            x_exit_1 = self.exit_1(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.statics[flag][current_layer-self.target_layer]= self.statics[flag][current_layer-self.target_layer] + 1
                self.num_early_exit_list[5] += 1
                return 5, exit1

        x = F.relu(self.bn5(self.conv5(x)))
        current_layer = 7
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
###############################################################################################################################################
#      x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.view(-1, 512)
        x = self.fc(x)
        self.original += 1
        current_layer += 1
        self.statics[flag][current_layer-self.target_layer]+= 1
        return 20, x
    '''
    def forward( self, x ):
        x = F.relu(self.bn1(self.conv1(x)))
        current_layer = 1
        if self.target_layer == 1:
            x_exit_1 = self.exit_0(x)
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
            x_exit_1 = self.exit_0(x)
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
            x_exit_1 = self.exit_0(x)
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
            x_exit_1 = self.exit_1(x)
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
            x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[4])
            if ratio >= 1:
                self.num_early_exit_list[4] += 1
                return 4, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)

        x = F.max_pool2d(x, 2, 2)
        current_layer = 6
        if self.target_layer == 6:
            x_exit_1 = self.exit_1(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            if ratio >= 1:
                self.num_early_exit_list[5] += 1
                return 5, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)

        x = F.relu(self.bn5(self.conv5(x)))
        current_layer =7
        if self.target_layer == 7:
            x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[6])
            if ratio >= 1:
                self.num_early_exit_list[6] += 1
                return 6, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)

        x = F.relu(self.bn6(self.conv6(x)))
        current_layer = 8
        if self.target_layer == 8:
            x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[7])
            if ratio >= 1:
                self.num_early_exit_list[7] += 1
                return 7, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer) 

        x = F.relu(self.bn7(self.conv7(x)))
        current_layer = 9
        if self.target_layer == 9:
            x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[8])
            if ratio >= 1:
                self.num_early_exit_list[8] += 1
                return 8, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
        
        x = F.relu(self.bn8(self.conv8(x)))
        current_layer = 10
        if self.target_layer == 10:
            x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[9])
            if ratio >= 1:
                self.num_early_exit_list[9] += 1
                return 9, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer) 

        x = F.max_pool2d(x, 2, 2)
        current_layer = 11
        if self.target_layer == 11:
            x_exit_1 = self.exit_2(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
            if ratio >= 1:
                self.num_early_exit_list[10] += 1
                return 10, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
        
        x = F.relu(self.bn9(self.conv9(x)))
        current_layer = 12
        if self.target_layer == 12:
            x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[11])
            if ratio >= 1:
                self.num_early_exit_list[11] += 1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num 

        x = F.relu(self.bn10(self.conv10(x)))
        current_layer = 13
        if self.target_layer == 13:
            x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[12])
            if ratio >= 1:
                self.num_early_exit_list[12] += 1
                return 12, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
             
        x = F.relu(self.bn11(self.conv11(x)))
        current_layer = 14
        if self.target_layer == 14:
            x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[13])
            if ratio >= 1:
                self.num_early_exit_list[13] += 1
                return 13, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn12(self.conv12(x)))
        current_layer = 15
        if self.target_layer == 15:
            x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[14])
            if ratio >= 1:
                self.num_early_exit_list[14] += 1
                return 14, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.max_pool2d(x, 2, 2)
        current_layer = 16
        if self.target_layer == 16:
            x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[15])
            if ratio >= 1:
                self.num_early_exit_list[15] += 1
                return 15, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn13(self.conv13(x)))
        current_layer = 17
        if self.target_layer == 17:
            x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[16])
            if ratio >= 1:
                self.num_early_exit_list[16] += 1
                return 16, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn14(self.conv14(x)))
        current_layer = 18
        if self.target_layer == 18:
            x_exit_1 = self.exit_3(x)
         #   x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[17])
            if ratio >= 1:
                self.num_early_exit_list[17] += 1
                return 17, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn15(self.conv15(x)))
        current_layer = 19
        if self.target_layer == 19:
            x_exit_1 = self.exit_3(x)
       #     x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[18])
            if ratio >= 1:
                self.num_early_exit_list[18] += 1
                return 18, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
        
        x = F.relu(self.bn16(self.conv16(x)))
        current_layer = 20
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
    

class vgg19_exits_train_jump_mnist( vgg19_exits_eval_jump_mnist ):
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
##########################################################################################################################################################
    #    x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x

    def forward_exits( self, x ):
        x = F.relu(self.bn1(self.conv1(x)))
        x_exit_1 = self.exit_0(x)
  #   #   x_exit_1 = self.exit_0_fc(x_exit_1)
        exit1 = self.exit_1_fc(x_exit_1)


        x = F.relu(self.bn2(self.conv2(x)))
        x_exit_2 = self.exit_0(x)
  #   #   x_exit_2 = self.exit_0_fc(x_exit_2)
        exit2 = self.exit_1_fc(x_exit_2)


   #   #  x = F.max_pool2d(x, 2, 2)
        x = F.max_pool2d(x, 1, 1)
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
    #    exit2 = (x_exit_1 + x_exit_2) / 2
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
###                            Vgg 19 STL-10                               ###
##############################################################################


class vgg19_exits_eval_jump_stl10( nn.Module ):
    def __init__(self):
        super(vgg19_exits_eval_jump_stl10, self).__init__()
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
        self.fc = nn.Linear(512*9, data_type)
        # early exits
        self.exit_0 = LogisticConvBoF(64, 64, split_horizon=16)
        self.exit_1 = LogisticConvBoF(128, 64, split_horizon=8)
        self.exit_2 = LogisticConvBoF(256, 64, split_horizon=4)
        self.exit_3 = LogisticConvBoF(512, 64, split_horizon=2)
        self.exit_0_fc = nn.Linear(256*9,64)
        self.exit_1_fc = nn.Linear(64, data_type)
        self.exit_2_fc = nn.Linear(576, 64)
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
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.statics = [([0] * 20) for i in range(11)]
        self.total = [0] * 11

        # the beta coefficient used for accuracy-speed trade-off, the higher the more accurate
        self.beta = 0
        self.target_layer = 6
        self.start_layer = 6
        self.count_store = [0]*4
        self.layer_store = [0]*20
        self.jumpstep_store = []
        self.prediction_store = []
        
        self.quant_switch = 0
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        # The test forward normal to choose 'normal_forward', 'accuracy_forward', 'quant_forward'
        self.forward_mode = 'normal_forward'
        self.possible_layer = []
    
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

    def set_possible_layer(self, temp):
        self.possible_layer = temp

    def get_specific_exit_number(self, iterate):
        return self.num_early_exit_list[iterate]

    def _calculate_max_activation( self, param ):
        '''
        return the maximum activation item in [param]
        '''
        return torch.max( torch.abs( torch.max( param ) ), torch.abs( torch.min( param ) ) )

    def output( self ):
        return (self.count_store, self.layer_store), self.jumpstep_store, self.prediction_store


    def simple_conv1d(self, param):
        copy = param.copy()
        temp = []
        copy.insert(0,0)
        copy.append(0)
        for i in range(len(copy)-2):
            temp.append(copy[i]+copy[i+1]+copy[i+2])
        return temp
        
    def prediction(self, param, layer):
        round = 5 
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

    def settings( self, layer, forward_mode, p):
        self.target_layer = layer
        self.start_layer = layer
        self.forward_mode = forward_mode
        self.quant_switch = p

    def forward( self, x):
        if self.forward_mode == 'accuracy_forward':
            return self.accuracy_forward(x)
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
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[0])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[0] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 0, exit1
            flag = 1
            self.total[flag] = self.total[flag] + 1
        #    self.count_store.append(ratio.item())
            self.prediction_store.append(self.prediction(exit1, current_layer))
           # self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_0(self.dequant(x))
            else:
                x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[0])
            if ratio >= 1:
                self.statics[flag][current_layer-self.target_layer]= self.statics[flag][current_layer-self.target_layer] + 1
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
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[1] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 1, exit1
        #    self.count_store.append(ratio.item())
            self.total[flag] = self.total[flag] + 1
            self.prediction_store.append(self.prediction(exit1, current_layer))
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
                self.statics[flag][current_layer-self.target_layer]= self.statics[flag][current_layer-self.target_layer] + 1
                self.num_early_exit_list[1] += 1
                return 1, exit1
    

        x = F.max_pool2d(x, 2, 2)
        current_layer = 3
        if self.target_layer == 3:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_0(self.dequant(x))
            else:
                x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_2_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[2])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[2] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 2, exit1
        #    self.count_store.append(ratio.item())
            self.total[flag] = self.total[flag] + 1
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_0(self.dequant(x))
            else:
                x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_2_fc(x_exit_1)
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
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 3, exit1
        #    self.count_store.append(ratio.item())
            self.total[flag] = self.total[flag] + 1
            self.prediction_store.append(self.prediction(exit1, current_layer))    
     #       self.prediction_store.append(self.prediction(exit1, current_layer))     
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
                self.statics[flag][current_layer-self.target_layer]= self.statics[flag][current_layer-self.target_layer] + 1
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
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[4] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 4, exit1
        #    self.count_store.append(ratio.item())
            self.total[flag] = self.total[flag] + 1
            self.prediction_store.append(self.prediction(exit1, current_layer))           
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
                self.statics[flag][current_layer-self.target_layer]= self.statics[flag][current_layer-self.target_layer] + 1
                self.num_early_exit_list[4] += 1
                return 4, exit1

        x = F.max_pool2d(x, 2, 2)
        current_layer = 6
        if self.target_layer == 6:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_2_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[5] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 5, exit1
        #    self.count_store.append(ratio.item())
            self.total[flag] = self.total[flag] + 1
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_2_fc(x_exit_1)
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
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 6, exit1   
            flag = 1    
        #    self.count_store.append(ratio.item())
            self.total[flag] = self.total[flag] + 1
            self.prediction_store.append(self.prediction(exit1, current_layer))
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
                self.num_early_exit_list[6] += 1
                self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
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
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 7, exit1
            flag = 1
        #    self.count_store.append(ratio.item())
            self.total[flag] = self.total[flag] + 1
            self.prediction_store.append(self.prediction(exit1, current_layer))
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
                self.num_early_exit_list[7] += 1
                self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
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
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[8] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 8, exit1
        #    self.count_store.append(ratio.item())
            self.total[flag] = self.total[flag] + 1
            self.prediction_store.append(self.prediction(exit1, current_layer))
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
                self.num_early_exit_list[8] += 1
                self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
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
            self.total[flag] = self.total[flag] + 1
            self.prediction_store.append(self.prediction(exit1, current_layer))
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
                self.statics[flag][current_layer-self.target_layer]= self.statics[flag][current_layer-self.target_layer] + 1
                self.num_early_exit_list[9] += 1
                return 9, exit1            
       
        x = F.max_pool2d(x, 2, 2)
        current_layer = 11
        if self.target_layer == 11:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_2_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
            if ratio >= 1:
                self.num_early_exit_list[10] += 1
                return 10, exit1
            self.total[flag] = self.total[flag] + 1
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_2_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.statics[flag][current_layer-self.target_layer]= self.statics[flag][current_layer-self.target_layer] + 1
                self.num_early_exit_list[10] += 1
                return 10, exit1                            
        
        x = F.relu(self.bn9(self.conv9(x)))
        current_layer = 12
        if self.target_layer == 12:        
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[11])
            if ratio >= 1:
                self.num_early_exit_list[11] += 1
                return 11, exit1
            self.total[flag] = self.total[flag] + 1
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
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
        current_layer = 13
        if self.target_layer == 13:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[12])
            if ratio >= 1:
                self.num_early_exit_list[12] += 1
                return 12, exit1
            self.prediction_store.append(self.prediction(exit1, current_layer))
            self.total[flag] = self.total[flag] + 1
        elif self.target_layer <= current_layer:
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
        current_layer = 14
        if self.target_layer == 14:     
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[13])
            if ratio >= 1:
                self.num_early_exit_list[13] += 1
                return 13, exit1
            self.prediction_store.append(self.prediction(exit1, current_layer))
            self.total[flag] = self.total[flag] + 1        
        elif self.target_layer <= current_layer:   
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
        current_layer = 15
        if self.target_layer == 15:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[14])
            if ratio >= 1:
                self.num_early_exit_list[14] += 1
                return 14, exit1
            self.prediction_store.append(self.prediction(exit1, current_layer))
            self.total[flag] = self.total[flag] + 1                        
        elif self.target_layer <= current_layer: 
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
        current_layer = 16
        if self.target_layer == 16:         
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_2_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[15])
            if ratio >= 1:
                self.num_early_exit_list[15] += 1
                return 15, exit1
            self.prediction_store.append(self.prediction(exit1, current_layer))
            self.total[flag] = self.total[flag] + 1             
        elif self.target_layer <= current_layer:         
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_2_fc(x_exit_1)
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
         #   x_exit_1 = self.exit_0_fc(x_exit_1)
        x_exit_1 = self.exit_2_fc(x_exit_1)
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
        x_exit_1 = self.exit_2_fc(x_exit_1)
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
        x_exit_1 = self.exit_2_fc(x_exit_1)
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
        x_exit_1 = self.exit_2_fc(x_exit_1)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[19])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]+= 1
            self.num_early_exit_list[19] += 1
            return 19, exit1
    
        x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.reshape(-1, 512*9)
        x = self.fc(x)
        if self.quant_switch == 1:
            x = self.dequant(x)
        self.original += 1
        current_layer += 1
        self.statics[flag][current_layer-self.target_layer]+= 1
        return 20, x


    def normal_forward( self, x ):
        self.target_layer = self.start_layer
        count = 0
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
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                count += 1

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
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                count+=1
            
        x = F.max_pool2d(x, 2, 2)
        current_layer = 3
        if self.target_layer == 3:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_0(self.dequant(x))
            else:
                x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_2_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[2])
            if ratio >= 1:
                self.num_early_exit_list[2] += 1
                return 2, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                count+=1

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
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                count += 1 
            
        x = F.relu(self.bn4(self.conv4(x)))
        current_layer = 5
        if self.target_layer == 5:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[4])
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[4] += 1
                return 4, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer) 
                count += 1

        x = F.max_pool2d(x, 2, 2) 
        current_layer = 6
        if self.target_layer == 6:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_2_fc(x_exit_1) # 576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            if ratio >= 1:
                self.num_early_exit_list[5] += 1
                return 5, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                count += 1

        x = F.relu(self.bn5(self.conv5(x))) #128*256*24*24
        current_layer = 7
        if self.target_layer == 7:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[6])
            if ratio >= 1:
                self.num_early_exit_list[6] += 1
                return 6, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                count += 1 

        x = F.relu(self.bn6(self.conv6(x))) #256*256*24*24*9
        current_layer = 8
        if self.target_layer == 8:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[7])
            if ratio >= 1:
                self.num_early_exit_list[7] += 1
                return 7, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer) 
                count += 1

        x = F.relu(self.bn7(self.conv7(x))) #256*256*24*24
        current_layer = 9
        if self.target_layer == 9:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[8])
            if ratio >= 1:
                self.num_early_exit_list[8] += 1
                return 8, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                count += 1 
        
        x = F.relu(self.bn8(self.conv8(x))) #256*256*24*24
        current_layer = 10
        if self.target_layer == 10:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[9])
            if ratio >= 1:
                self.num_early_exit_list[9] += 1
                return 9, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer) 
                count += 1

        x = F.max_pool2d(x, 2, 2)
        current_layer = 11
        if self.target_layer == 11:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
            if ratio >= 1:
                self.num_early_exit_list[10] += 1
                return 10, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer) 
                count += 1
        
        x = F.relu(self.bn9(self.conv9(x))) #256*512*12*12
        current_layer = 12
        if self.target_layer == 12:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[11])
            if ratio >= 1:
                self.num_early_exit_list[11] += 1
                return 11, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num 
                count += 1

        x = F.relu(self.bn10(self.conv10(x))) #512*512*12*12
        current_layer = 13
        if self.target_layer == 13:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[12])
            if ratio >= 1:
                self.num_early_exit_list[12] += 1
                return 12, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
                count += 1
             
        x = F.relu(self.bn11(self.conv11(x))) #512*512*12*12
        current_layer = 14
        if self.target_layer == 14:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[13])
            if ratio >= 1:
                self.num_early_exit_list[13] += 1
                return 13, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
                count += 1

        x = F.relu(self.bn12(self.conv12(x))) #512*512*12*12
        current_layer =  15
        if self.target_layer == 15:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[14])
            if ratio >= 1:
                self.num_early_exit_list[14] += 1
                return 14, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
                count += 1

        x = F.max_pool2d(x, 2, 2) 
        current_layer = 16
        if self.target_layer == 16:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[15])
            if ratio >= 1:
                self.num_early_exit_list[15] += 1
                return 15, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
                count +=1

        x = F.relu(self.bn13(self.conv13(x))) #512*512*6*6
        current_layer = 17
        if self.target_layer == 17:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[16])
            if ratio >= 1:
                self.num_early_exit_list[16] += 1
                return 16, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
                count += 1

        x = F.relu(self.bn14(self.conv14(x))) #512*512*6*6
        current_layer = 18
        if self.target_layer == 18:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[17])
            if ratio >= 1:
                self.num_early_exit_list[17] += 1
                return 17, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
                count +=1

        x = F.relu(self.bn15(self.conv15(x)))  #512*512*6*6
        current_layer = 19
        if self.target_layer == 19:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[18])
            if ratio >= 1:
                self.num_early_exit_list[18] += 1
                return 18, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
                count +=1
        
        x = F.relu(self.bn16(self.conv16(x))) #512*512*6*6
        current_layer = 20
        if self.target_layer == 20:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[19])
            if ratio >= 1:
                self.num_early_exit_list[19] += 1
                return 19, exit1
            count += 1
    
        x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.reshape(-1, 512*9)
        x = self.fc(x)
        if self.quant_switch == 1:
            x = self.dequant(x)
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
            x_exit_1 = self.exit_2_fc(x_exit_1)
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
            x_exit_1 = self.exit_1(x_exit_1) #48*48*128*64
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[4])
            count += 48*48*128*64 + 256*9*127 + 64*19
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[4] += 1
                return 4, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer) 

        x = F.max_pool2d(x, 2, 2) 
        ################################################################################
        '''
        x_exit_1 = self.exit_1(x) #24*24*128*64
        x_exit_1 = self.exit_2_fc(x_exit_1) # 576*127
        exit1 = self.exit_1_fc(x_exit_1) #64*19
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
        count += 24*24*128*64 + 576*127 + 64*19
        if ratio >= 1:
            self.count_store.append(count)
            self.num_early_exit_list[5] += 1
            return 5, exit1        
        '''
        ################################################################################
        current_layer = 6
        if self.target_layer == 6:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_1(x_exit_1) #24*24*128*64
            x_exit_1 = self.exit_2_fc(x_exit_1) # 576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            count += 24*24*128*64 + 576*127 + 64*19
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[5] += 1
                return 5, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)

        x = F.relu(self.bn5(self.conv5(x))) #128*256*24*24
        current_layer = 7
        if self.target_layer == 7:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_2(x_exit_1) #24*24*256*64
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[6])
            count += 128*256*24*24 + 256*9*127 + 64*19 + 128*256*24*24
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[6] += 1
                return 6, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer) 

        x = F.relu(self.bn6(self.conv6(x))) #256*256*24*24
        current_layer = 8
        if self.target_layer == 8:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_2(x_exit_1) #24*24*256*64
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[7])
            count += 256*256*24*24 + 24*24*256*64 + 256*9*127 + 64*19
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[7] += 1
                return 7, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer) 

        x = F.relu(self.bn7(self.conv7(x))) #256*256*24*24
        current_layer = 9
        if self.target_layer == 9:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_2(x_exit_1) #24*24*256*64
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[8])
            count += 256*256*24*24 + 24*24*256*64 + 256*9*127 + 64*19
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[8] += 1
                return 8, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer) 
        
        x = F.relu(self.bn8(self.conv8(x))) #256*256*24*24
        current_layer = 10
        if self.target_layer == 10:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_2(x_exit_1) #24*24*256*64
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[9])
            count += 256*256*24*24 + 24*24*256*64 + 256*9*127 + 64*19
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[9] += 1
                return 9, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer) 

        x = F.max_pool2d(x, 2, 2)
        ##################################################################################
        '''
        x_exit_1 = self.exit_2(x) #12*12*256*64
        x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
        exit1 = self.exit_1_fc(x_exit_1) #64*19
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
        count += 12*12*256*64 + 576*127 + 64*19 + 128*256*24*24 + 256*256*24*24*3
        if ratio >= 1:
            self.count_store.append(count)
            self.num_early_exit_list[10] += 1
            return 10, exit1
        '''
        ##################################################################################
        current_layer = 11
        if self.target_layer == 11:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_2(x_exit_1) #12*12*256*64
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
            count += 12*12*256*64 + 576*127 + 64*19
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[10] += 1
                return 10, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer) 
        
        x = F.relu(self.bn9(self.conv9(x))) #256*512*12*12
        current_layer = 12
        if self.target_layer == 12:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_3(x_exit_1) #512*64*12*12
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[11])
            count += 256*512*12*12 + 512*64*12*12 + 256*9*127 + 64*19
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[11] += 1
                return 11, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num 

        x = F.relu(self.bn10(self.conv10(x))) #512*512*12*12
        current_layer = 13
        if self.target_layer == 13:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_3(x_exit_1) #512*64*12*12
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[12])
            count += 512*512*12*12 + 256*9*127 + 64*19 + 512*64*12*12
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[12] += 1
                return 12, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
             
        x = F.relu(self.bn11(self.conv11(x))) #512*512*12*12
        current_layer = 14
        if self.target_layer == 14:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_3(x_exit_1) #512*64*12*12
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[13])
            count += 512*512*12*12 + 512*64*12*12 + 256*9*127 + 64*19
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[13] += 1
                return 13, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn12(self.conv12(x))) #512*512*12*12
        current_layer =  15
        if self.target_layer == 15:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_3(x_exit_1) #512*64*12*12
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[14])
            count += 512*512*12*12 + 512*64*12*12 + 256*9*127 + 64*19
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[14] += 1
                return 14, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.max_pool2d(x, 2, 2) 
        ##################################################################################
        '''
        x_exit_1 = self.exit_3(x) # 512*64*6*6
        x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
        exit1 = self.exit_1_fc(x_exit_1) #64*19
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[15])
        count += 512*64*6*6 + 576*127 + 64*19 + 512*512*12*12*3 + 256*512*12*12 + 12*12*256*64 + 576*127 + 64*19 + 128*256*24*24 + 256*256*24*24*3 + 24*24*128*64 + 576*127 + 64*19
        if ratio >= 1:
            self.count_store.append(count)
            self.num_early_exit_list[15] += 1
            return 15, exit1
        '''
        ###################################################################################
        current_layer = 16
        if self.target_layer == 16:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_3(x_exit_1) # 512*64*6*6
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[15])
            count += 512*64*6*6 + 576*127 + 64*19
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[15] += 1
                return 15, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn13(self.conv13(x))) #512*512*6*6
        current_layer = 17
        if self.target_layer == 17:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_3(x_exit_1)  # 512*64*6*6
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[16])
            count += 512*512*6*6 + 512*64*6*6 + 576*127 + 64*19
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[16] += 1
                return 16, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn14(self.conv14(x))) #512*512*6*6
        current_layer = 18
        if self.target_layer == 18:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_3(x_exit_1) # 512*64*6*6
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[17])
            count += 512*512*6*6 + 512*64*6*6 + 576*127 + 64*19
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[17] += 1
                return 17, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn15(self.conv15(x)))  #512*512*6*6
        current_layer = 19
        if self.target_layer == 19:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_3(x_exit_1) # 512*64*6*6
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[18])
            count += 512*512*6*6 + 512*64*6*6 + 576*127 + 64*19
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[18] += 1
                return 18, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
        
        x = F.relu(self.bn16(self.conv16(x))) #512*512*6*6
        current_layer = 20
        if self.target_layer == 20:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_3(x_exit_1) # 512*64*6*6
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[19])
            count += 512*512*6*6 + 512*64*6*6 + 576*127 + 64*19
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[19] += 1
                return 19, exit1
    
        x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.reshape(-1, 512*9)
        x = self.fc(x)
        count += 512*512*6*6 + 512*512*6*6 + 512*512*6*6 + 512*64*6*6 +  512*512*6*6
        self.count_store.append(count)
        self.original += 1
        x = self.dequant(x)
        return 20, x


    def quant_accuracy_forward( self, x ):
        flag = 0
        x = self.quant(x)
        x = F.relu(self.bn1(self.conv1(x)))
        current_layer = 1
        if self.target_layer == 1:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_0(x_exit_1)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[0])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[0] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 0, exit1
            self.total[flag] = self.total[flag] + 1
            self.count_store.append(ratio.item())
           # self.prediction_store.append(1)
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_0(x_exit_1)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[0])
            if ratio >= 1:
                self.statics[flag][current_layer-self.target_layer]= self.statics[flag][current_layer-self.target_layer] + 1
                self.num_early_exit_list[0] += 1
                return 0, exit1
        

        x = F.relu(self.bn2(self.conv2(x)))
        current_layer = 2
        if self.target_layer == 2:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_0(x_exit_1)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[1])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[1] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 1, exit1
            self.count_store.append(ratio.item())
            self.total[flag] = self.total[flag] + 1
            self.prediction_store.append(1)
        elif self.target_layer <= current_layer:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_0(x_exit_1)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[1])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.statics[flag][current_layer-self.target_layer]= self.statics[flag][current_layer-self.target_layer] + 1
                self.num_early_exit_list[1] += 1
                return 1, exit1
    

        x = F.max_pool2d(x, 2, 2)
        current_layer = 3
        if self.target_layer == 3:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_0(x_exit_1)
            x_exit_1 = self.exit_2_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[2])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[2] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 2, exit1
            self.count_store.append(ratio.item())
            self.total[flag] = self.total[flag] + 1
            self.prediction_store.append(1)
        elif self.target_layer <= current_layer:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_0(x_exit_1)
            x_exit_1 = self.exit_2_fc(x_exit_1)
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
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_1(x_exit_1)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[3])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[3] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 3, exit1
            self.count_store.append(ratio.item())
            self.total[flag] = self.total[flag] + 1
            self.prediction_store.append(self.prediction(exit1, current_layer))    
     #       self.prediction_store.append(self.prediction(exit1, current_layer))     
        elif self.target_layer <= current_layer:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_1(x_exit_1)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[3])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.statics[flag][current_layer-self.target_layer]= self.statics[flag][current_layer-self.target_layer] + 1
                self.num_early_exit_list[3] += 1
                return 3, exit1
        
        
        x = F.relu(self.bn4(self.conv4(x)))
        current_layer = 5
        if self.target_layer == 5:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_1(x_exit_1)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[4])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[4] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 4, exit1
            self.count_store.append(ratio.item())
            self.total[flag] = self.total[flag] + 1
            self.prediction_store.append(1)           
        elif self.target_layer <= current_layer:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_1(x_exit_1)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[4])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.statics[flag][current_layer-self.target_layer]= self.statics[flag][current_layer-self.target_layer] + 1
                self.num_early_exit_list[4] += 1
                return 4, exit1

        x = F.max_pool2d(x, 2, 2)
        current_layer = 6
        if self.target_layer == 6:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_1(x_exit_1)
            x_exit_1 = self.exit_2_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[5] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 5, exit1
            self.count_store.append(ratio.item())
            self.total[flag] = self.total[flag] + 1
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_1(x_exit_1)
            x_exit_1 = self.exit_2_fc(x_exit_1)
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
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_2(x_exit_1)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[6])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[6] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 6, exit1
            self.count_store.append(ratio.item())
            self.total[flag] = self.total[flag] + 1
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_2(x_exit_1)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[6])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[6] += 1
                self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 6, exit1


        x = F.relu(self.bn6(self.conv6(x)))
        current_layer = 8
        if self.target_layer == 8:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_2(x_exit_1)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[7])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[7] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 7, exit1
            self.count_store.append(ratio.item())
            self.total[flag] = self.total[flag] + 1
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_2(x_exit_1)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[7])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[7] += 1
                self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 7, exit1

  
        x = F.relu(self.bn7(self.conv7(x)))
        current_layer = 9
        if self.target_layer == 9:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_2(x_exit_1)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[8])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[8] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 8, exit1
            self.count_store.append(ratio.item())
            self.total[flag] = self.total[flag] + 1
            self.prediction_store.append(self.prediction(exit1, current_layer))
        elif self.target_layer <= current_layer:
            x_exit_1 = self.dequant(x)
            x_exit_1 = self.exit_2(x_exit_1)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[8])
            if ratio >= 1:
                self.jumpstep_store.append(current_layer-self.target_layer)
                self.num_early_exit_list[8] += 1
                self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 8, exit1
 
        
        x = F.relu(self.bn8(self.conv8(x)))
        current_layer += 1
        x_exit_1 = self.dequant(x)
        x_exit_1 = self.exit_2(x_exit_1)
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
        x_exit_1 = self.dequant(x)
        x_exit_1 = self.exit_2(x_exit_1)
        x_exit_1 = self.exit_2_fc(x_exit_1)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]+= 1
            self.num_early_exit_list[10] += 1
            return 10, exit1
        
        
        x = F.relu(self.bn9(self.conv9(x)))
        current_layer += 1
        x_exit_1 = self.dequant(x)
        x_exit_1 = self.exit_3(x_exit_1)
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
        x_exit_1 = self.dequant(x)
        x_exit_1 = self.exit_3(x_exit_1)
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
        x_exit_1 = self.dequant(x)
        x_exit_1 = self.exit_3(x_exit_1)
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
        x_exit_1 = self.dequant(x)
        x_exit_1 = self.exit_3(x_exit_1)
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
        x_exit_1 = self.dequant(x)
        x_exit_1 = self.exit_3(x_exit_1)
        x_exit_1 = self.exit_2_fc(x_exit_1)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[15])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]+= 1
            self.num_early_exit_list[15] += 1
            return 15, exit1
    

        x = F.relu(self.bn13(self.conv13(x)))
        current_layer += 1
        x_exit_1 = self.dequant(x)
        x_exit_1 = self.exit_3(x_exit_1)
        x_exit_1 = self.exit_2_fc(x_exit_1)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[16])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]+= 1
            self.num_early_exit_list[16] += 1
            return 16, exit1

        x = F.relu(self.bn14(self.conv14(x)))
        current_layer += 1
        x_exit_1 = self.dequant(x)
        x_exit_1 = self.exit_3(x_exit_1)
        x_exit_1 = self.exit_2_fc(x_exit_1)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[17])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]+= 1
            self.num_early_exit_list[17] += 1
            return 17, exit1
            

        x = F.relu(self.bn15(self.conv15(x)))
        current_layer += 1
        x_exit_1 = self.dequant(x)
        x_exit_1 = self.exit_3(x_exit_1)
        x_exit_1 = self.exit_2_fc(x_exit_1)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[18])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]+= 1
            self.num_early_exit_list[18] += 1
            return 18, exit1
            
        
        x = F.relu(self.bn16(self.conv16(x)))
        current_layer += 1
        x_exit_1 = self.dequant(x)
        x_exit_1 = self.exit_3(x_exit_1)
        x_exit_1 = self.exit_2_fc(x_exit_1)
        exit1 = self.exit_1_fc(x_exit_1)
        ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[19])
        if ratio >= 1:
            self.jumpstep_store.append(current_layer-self.target_layer)
            self.statics[flag][current_layer-self.target_layer]+= 1
            self.num_early_exit_list[19] += 1
            return 19, exit1
    
        x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.reshape(-1, 512*9)
        x = self.fc(x)
        x = self.dequant(x)
        self.original += 1
        current_layer += 1
        self.statics[flag][current_layer-self.target_layer]+= 1
        return 20, x


    def test_on_train_forward( self, x):
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
            else:
                self.target_layer +=  self.prediction(exit1, current_layer)
            
        x = F.max_pool2d(x, 2, 2)
        current_layer = 3
        if self.target_layer == 3:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_0(self.dequant(x))
            else:
                x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_2_fc(x_exit_1)
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
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
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
            x_exit_1 = self.exit_2_fc(x_exit_1) # 576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            if ratio >= 1:
                self.num_early_exit_list[5] += 1
                return 5, exit1

        x = F.relu(self.bn5(self.conv5(x))) #128*256*24*24
        current_layer = 7
        if self.target_layer == 7:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[6])
            if ratio >= 1:
                self.num_early_exit_list[6] += 1
                return 6, exit1

        x = F.relu(self.bn6(self.conv6(x))) #256*256*24*24*9
        current_layer = 8
        if self.target_layer == 8:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[7])
            if ratio >= 1:
                self.num_early_exit_list[7] += 1
                return 7, exit1

        x = F.relu(self.bn7(self.conv7(x))) #256*256*24*24
        current_layer = 9
        if self.target_layer == 9:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[8])
            if ratio >= 1:
                self.num_early_exit_list[8] += 1
                return 8, exit1
        
        x = F.relu(self.bn8(self.conv8(x))) #256*256*24*24
        current_layer = 10
        if self.target_layer == 10:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[9])
            if ratio >= 1:
                self.num_early_exit_list[9] += 1
                return 9, exit1

        x = F.max_pool2d(x, 2, 2)
        current_layer = 11
        if self.target_layer == 11:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
            if ratio >= 1:
                self.num_early_exit_list[10] += 1
                return 10, exit1
            else:
                self.target_layer +=  self.prediction(exit1, current_layer) 
        
        x = F.relu(self.bn9(self.conv9(x))) #256*512*12*12
        current_layer = 12
        if self.target_layer == 12:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[11])
            if ratio >= 1:
                self.num_early_exit_list[11] += 1
                return 11, exit1

        x = F.relu(self.bn10(self.conv10(x))) #512*512*12*12
        current_layer = 13
        if self.target_layer == 13:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[12])
            if ratio >= 1:
                self.num_early_exit_list[12] += 1
                return 12, exit1
             
        x = F.relu(self.bn11(self.conv11(x))) #512*512*12*12
        current_layer = 14
        if self.target_layer == 14:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[13])
            if ratio >= 1:
                self.num_early_exit_list[13] += 1
                return 13, exit1

        x = F.relu(self.bn12(self.conv12(x))) #512*512*12*12
        current_layer =  15
        if self.target_layer == 15:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[14])
            if ratio >= 1:
                self.num_early_exit_list[14] += 1
                return 14, exit1

        x = F.max_pool2d(x, 2, 2) 
        current_layer = 16
        if self.target_layer == 16:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[15])
            if ratio >= 1:
                self.num_early_exit_list[15] += 1
                return 15, exit1

        x = F.relu(self.bn13(self.conv13(x))) #512*512*6*6
        current_layer = 17
        if self.target_layer == 17:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[16])
            if ratio >= 1:
                self.num_early_exit_list[16] += 1
                return 16, exit1


        x = F.relu(self.bn14(self.conv14(x))) #512*512*6*6
        current_layer = 18
        if self.target_layer == 18:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[17])
            if ratio >= 1:
                self.num_early_exit_list[17] += 1
                return 17, exit1

        x = F.relu(self.bn15(self.conv15(x)))  #512*512*6*6
        current_layer = 19
        if self.target_layer == 19:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[18])
            if ratio >= 1:
                self.num_early_exit_list[18] += 1
                return 18, exit1
        
        x = F.relu(self.bn16(self.conv16(x))) #512*512*6*6
        current_layer = 20
        if self.target_layer == 20:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[19])
            if ratio >= 1:
                self.num_early_exit_list[19] += 1
                return 19, exit1
    
        x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.reshape(-1, 512*9)
        x = self.fc(x)
        if self.quant_switch == 1:
            x = self.dequant(x)
        self.original += 1
        return 20, x


    def test_on_train_normal_forward( self, x ):
        self.target_layer = self.start_layer
        count = 0
        if self.quant_switch == 1:
            x = self.quant(x)
        x = F.relu(self.bn1(self.conv1(x)))
        self.count_store[3] += 3*32*64*32*9
        current_layer = 1
        if current_layer in self.possible_layer:
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

            
        x = F.max_pool2d(x, 2, 2)
        current_layer = 3
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_0(self.dequant(x))
            else:
                x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_2_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[2])
            if ratio >= 1:
                self.num_early_exit_list[2] += 1
                return 2, exit1

        x = F.relu(self.bn3(self.conv3(x)))
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
            
        x = F.relu(self.bn4(self.conv4(x)))
        current_layer = 5
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[4])
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[4] += 1
                return 4, exit1

        x = F.max_pool2d(x, 2, 2) 
        current_layer = 6
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_2_fc(x_exit_1) # 576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            if ratio >= 1:
                self.num_early_exit_list[5] += 1
                return 5, exit1

        x = F.relu(self.bn5(self.conv5(x))) #128*256*24*24
        current_layer = 7
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[6])
            if ratio >= 1:
                self.num_early_exit_list[6] += 1
                return 6, exit1

        x = F.relu(self.bn6(self.conv6(x))) #256*256*24*24*9
        current_layer = 8
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[7])
            if ratio >= 1:
                self.num_early_exit_list[7] += 1
                return 7, exit1

        x = F.relu(self.bn7(self.conv7(x))) #256*256*24*24
        current_layer = 9
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[8])
            if ratio >= 1:
                self.num_early_exit_list[8] += 1
                return 8, exit1
        
        x = F.relu(self.bn8(self.conv8(x))) #256*256*24*24
        current_layer = 10
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[9]) 
            if ratio >= 1:
                self.num_early_exit_list[9] += 1
                return 9, exit1

        x = F.max_pool2d(x, 2, 2)
        current_layer = 11
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
            if ratio >= 1:
                self.num_early_exit_list[10] += 1
                return 10, exit1
        
        x = F.relu(self.bn9(self.conv9(x))) #256*512*12*12
        current_layer = 12
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[11])
            if ratio >= 1:
                self.num_early_exit_list[11] += 1
                return 11, exit1

        x = F.relu(self.bn10(self.conv10(x))) #512*512*12*12
        current_layer = 13
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[12])
            if ratio >= 1:
                self.num_early_exit_list[12] += 1
                return 12, exit1
             
        x = F.relu(self.bn11(self.conv11(x))) #512*512*12*12
        current_layer = 14
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[13])
            if ratio >= 1:
                self.num_early_exit_list[13] += 1
                return 13, exit1

        x = F.relu(self.bn12(self.conv12(x))) #512*512*12*12
        current_layer =  15
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) #256*9*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[14])
            if ratio >= 1:
                self.num_early_exit_list[14] += 1
                return 14, exit1

        x = F.max_pool2d(x, 2, 2) 
        current_layer = 16
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[15])  
            if ratio >= 1:
                self.num_early_exit_list[15] += 1
                return 15, exit1

        x = F.relu(self.bn13(self.conv13(x))) #512*512*6*6
        current_layer = 17
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[16])
            if ratio >= 1:
                self.num_early_exit_list[16] += 1
                return 16, exit1

        x = F.relu(self.bn14(self.conv14(x))) #512*512*6*6
        current_layer = 18
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[17])
            if ratio >= 1:
                self.num_early_exit_list[17] += 1
                return 17, exit1

        x = F.relu(self.bn15(self.conv15(x)))  #512*512*6*6
        current_layer = 19
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[18])
            if ratio >= 1:
                self.num_early_exit_list[18] += 1
                return 18, exit1

        
        x = F.relu(self.bn16(self.conv16(x))) #512*512*6*6
        current_layer = 20
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_2_fc(x_exit_1) #576*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[19])
            if ratio >= 1:
                self.num_early_exit_list[19] += 1
                return 19, exit1
            count += 1
    
        x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.reshape(-1, 512*9)
        x = self.fc(x)
        if self.quant_switch == 1:
            x = self.dequant(x) 
        self.original += 1
        return 20, x

class vgg19_exits_train_jump_stl10( vgg19_exits_eval_jump_stl10):
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
        x = x.view(-1, 512*9)
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
        x_exit_3 = self.exit_2_fc(x_exit_3)
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
        x_exit_6 = self.exit_2_fc(x_exit_6)
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
        x_exit_11 = self.exit_2_fc(x_exit_11)
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
        x_exit_16 = self.exit_2_fc(x_exit_16)
        exit16 = self.exit_1_fc(x_exit_16)

        x = F.relu(self.bn13(self.conv13(x)))
        x_exit_17 = self.exit_3(x)
        x_exit_17 = self.exit_2_fc(x_exit_17)
        exit17 = self.exit_1_fc(x_exit_17)

        x = F.relu(self.bn14(self.conv14(x)))
        x_exit_18 = self.exit_3(x)
        x_exit_18 = self.exit_2_fc(x_exit_18)
        exit18 = self.exit_1_fc(x_exit_18)

        x = F.relu(self.bn15(self.conv15(x)))
        x_exit_19 = self.exit_3(x)
        x_exit_19 = self.exit_2_fc(x_exit_19)
        exit19 = self.exit_1_fc(x_exit_19)

        x = F.relu(self.bn16(self.conv16(x)))
        x_exit_20 = self.exit_3(x)
        x_exit_20 = self.exit_2_fc(x_exit_20)
        exit20 = self.exit_1_fc(x_exit_20)

        x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.view(-1, 512*9)
        x = self.fc(x)

        return (exit1, exit2, exit3, exit4, exit5, exit6, exit7, exit8, exit9, exit10, exit11, exit12, exit13, exit14, exit15, exit16, exit17, exit18, exit19, exit20, x)  



##############################################################################
###                            Vgg 19 SVHN                                 ###
##############################################################################


class vgg19_exits_eval_jump_svhn( nn.Module ):
    def __init__(self):
        super(vgg19_exits_eval_jump_svhn, self).__init__()
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
        self.count_store = [0]*4
        self.layer_store = [0]*20
        self.jumpstep_store = []
        self.prediction_store = []


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

    def get_specific_exit_number(self, iterate):
        return self.num_early_exit_list[iterate]

    def set_possible_layer(self, temp):
        self.possible_layer = temp
    
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
        return (self.count_store,self.layer_store), self.jumpstep_store, self.prediction_store
      #  return self.ratio_store, self.jumpstep_store
    
    
    def settings( self, layer, forward_mode, p = 0 ):
        self.target_layer = layer
        self.start_layer = layer
        self.forward_mode = forward_mode
        self.quant_switch = p

    def forward( self, x):
        if self.forward_mode == 'accuracy_forward':
            return self.accuracy_forward(x)
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
            self.count_store.append(ratio.item())
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
         #   self.count_store.append(ratio.item())
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
          #  self.count_store.append(ratio.item())
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
         #   self.count_store.append(ratio.item())
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
          #  self.count_store.append(ratio.item())
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
          #  self.count_store.append(ratio.item())
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
          #  self.count_store.append(ratio.item())
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
        self.count_store[3] += 3*32*64*32
        current_layer = 1
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_0(self.dequant(x))
            else:
                x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 256*127
            exit1 = self.exit_1_fc(x_exit_1) #64*19
            ratio = self._calculate_max_activation( exit1 ) / self.beta * self.activation_threshold_list[0]
            time2 = time.perf_counter()  #Time Sign
            self.count_store[3] += 32*64*64*32 + 256*127 + 64*19
            layer_end = current_layer          
            self.count_store[0] += time2 - time1
            self.layer_store[0] += layer_end - layer_st
            time1 = time.perf_counter()
            layer_st = current_layer     
            if ratio >= 1:
                self.num_early_exit_list[0] += 1
                return 0, exit1


        x = F.relu(self.bn2(self.conv2(x))) #32*32*64*64
        self.count_store[3] += 32*32*64*64
        current_layer = 2
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_0(self.dequant(x))
            else:
                x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[1])
            time2 = time.perf_counter()  #Time Sign
            self.count_store[3] += 32*64*64*32 + 256*127 + 64*19
            layer_end = current_layer         
            self.count_store[0] += time2 - time1
            self.layer_store[0] += layer_end - layer_st
            time1 = time.perf_counter()
            layer_st = current_layer    
            if ratio >= 1:
                self.num_early_exit_list[1] += 1
                return 1, exit1
            
        x = F.max_pool2d(x, 2, 2) #32*32*64*64
        self.count_store[3] += 32*32*64*64
        current_layer = 3
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_0(self.dequant(x))
            else:
                x_exit_1 = self.exit_0(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[2])
            time2 = time.perf_counter()  #Time Sign
            self.count_store[3] += 32*64*64*32 + 64*19
            layer_end = current_layer          
            self.count_store[0] += time2 - time1
            self.layer_store[0] += layer_end - layer_st
            time1 = time.perf_counter()
            layer_st = current_layer    
            if ratio >= 1:
                self.num_early_exit_list[2] += 1
                return 2, exit1    

        x = F.relu(self.bn3(self.conv3(x))) #16*16*128*128
        self.count_store[3] += 16*128*16*128
        current_layer = 4
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[3])
            time2 = time.perf_counter()  #Time Sign
            self.count_store[3] += 16*16*128*64 + 127*64+ 64*19
            layer_end = current_layer           
            self.count_store[0] += time2 - time1
            self.layer_store[0] += layer_end - layer_st
            time1 = time.perf_counter()
            layer_st = current_layer    
            if ratio >= 1:
                self.num_early_exit_list[3] += 1
                return 3, exit1

            
        x = F.relu(self.bn4(self.conv4(x))) #16*16*128*128
        self.count_store[3] += 16*16*128*128
        current_layer = 5
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[4])
            time2 = time.perf_counter()  #Time Sign
            self.count_store[3] += 16*16*128*64 + 127*64+ 64*19
            layer_end = current_layer        
            self.count_store[0] += time2 - time1
            self.layer_store[0] += layer_end - layer_st
            time1 = time.perf_counter()
            layer_st = current_layer                
            if ratio >= 1:
                self.num_early_exit_list[4] += 1
                return 4, exit1


        x = F.max_pool2d(x, 2, 2) #16*16*128*128
        self.count_store[3] += 16*16*128*128
        current_layer = 6
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_1(self.dequant(x))
            else:
                x_exit_1 = self.exit_1(x)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * 19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            time2 = time.perf_counter()  #Time Sign
            self.count_store[3] +=  8*8*128*64 + 64 * 19
            layer_end = current_layer           
            self.count_store[0] += time2 - time1
            self.layer_store[0] += layer_end - layer_st
            time1 = time.perf_counter()
            layer_st = current_layer    
            if ratio >= 1:
                self.num_early_exit_list[5] += 1
                return 5, exit1

        x = F.relu(self.bn5(self.conv5(x))) # 8*8*128*256
        self.count_store[3] += 8*8*128*256
        current_layer = 7
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 256 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[6])
            time2 = time.perf_counter()  #Time Sign
            self.count_store[3] += 8*8*256*64 + 256 * (2*64-1) + 64*19
            layer_end = current_layer          
            self.count_store[0] += time2 - time1
            self.layer_store[0] += layer_end - layer_st
            time1 = time.perf_counter()
            layer_st = current_layer    
            if ratio >= 1:
                self.num_early_exit_list[6] += 1
                return 6, exit1

        x = F.relu(self.bn6(self.conv6(x))) # 8*8*256*256
        self.count_store[3] += 8*8*256*256
        current_layer = 8
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 256 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64*19
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[7])
            time2 = time.perf_counter()  #Time Sign
            self.count_store[3] +=8*8*256*64 + 256 * (2*64-1) + 64*19
            layer_end = current_layer          
            self.count_store[0] += time2 - time1
            self.layer_store[0] += layer_end - layer_st
            time1 = time.perf_counter()
            layer_st = current_layer    
            if ratio >= 1:
                self.num_early_exit_list[7] += 1
                return 7, exit1


        x = F.relu(self.bn7(self.conv7(x))) # 8*8*256*256
        self.count_store[3] += 8*8*256*256
        current_layer = 9
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 256 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[8])
            time2 = time.perf_counter()  #Time Sign
            self.count_store[3] += 8*8*256*64 + 256 * (2*64-1) + 64*19
            layer_end = current_layer          
            self.count_store[0] += time2 - time1
            self.layer_store[0] += layer_end - layer_st
            time1 = time.perf_counter()
            layer_st = current_layer    
            if ratio >= 1:
                self.num_early_exit_list[8] += 1
                return 8, exit1

        
        x = F.relu(self.bn8(self.conv8(x))) # 8*8*256*256
        self.count_store[3] += 8*8*256*256
        current_layer = 10
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)# 256 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[9])
            time2 = time.perf_counter()  #Time Sign
            self.count_store[3] += 8*8*256*64 + 256 * (2*64-1) + 64*19
            layer_end = current_layer     
            self.count_store[0] += time2 - time1
            self.layer_store[0] += layer_end - layer_st
            time1 = time.perf_counter()
            layer_st = current_layer    
            if ratio >= 1:
                self.num_early_exit_list[9] += 1
                return 9, exit1


        x = F.max_pool2d(x, 2, 2) # 4*4*256*256
        self.count_store[3] += 4*4*256*256
        current_layer = 11
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_2(self.dequant(x))
            else:
                x_exit_1 = self.exit_2(x) 
            exit1 = self.exit_1_fc(x_exit_1)  # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
            time2 = time.perf_counter()  #Time Sign
            self.count_store[3] += 4*4*256*64 + 64 * 19
            layer_end = current_layer         
            self.count_store[0] += time2 - time1
            self.layer_store[0] += layer_end - layer_st
            time1 = time.perf_counter()
            layer_st = current_layer    
            if ratio >= 1:
                self.num_early_exit_list[10] += 1
                return 10, exit1

        
        x = F.relu(self.bn9(self.conv9(x))) # 256*512*4*4
        self.count_store[3] += 256*512*4*4
        current_layer = 12
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 512 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[11])
            time2 = time.perf_counter()  #Time Sign
            self.count_store[3] += 4*4*512*64 + 512 * (2*64-1) + 64 * 19
            layer_end = current_layer           
            self.count_store[0] += time2 - time1
            self.layer_store[0] += layer_end - layer_st
            time1 = time.perf_counter()
            layer_st = current_layer    
            if ratio >= 1:
                self.num_early_exit_list[11] += 1
                return 11, exit1

        x = F.relu(self.bn10(self.conv10(x))) #512*512*4*4
        self.count_store[3] += 512*512*4*4
        current_layer = 13
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 512 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[12])
            time2 = time.perf_counter()  #Time Sign
            self.count_store[3] += 512 * (2*64-1) + 64 * 19 + 4*4*512*64
            layer_end = current_layer         
            self.count_store[0] += time2 - time1
            self.layer_store[0] += layer_end - layer_st
            time1 = time.perf_counter()
            layer_st = current_layer    
            if ratio >= 1:
                self.num_early_exit_list[12] += 1
                return 12, exit1
             
        x = F.relu(self.bn11(self.conv11(x))) #512*512*4*4
        self.count_store[3] += 512*512*4*4
        current_layer = 14
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 512 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1)  # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[13])
            time2 = time.perf_counter()  #Time Sign
            self.count_store[3] += 512 * (2*64-1) + 64 * 19 + 4*4*512*64
            layer_end = current_layer          
            self.count_store[0] += time2 - time1
            self.layer_store[0] += layer_end - layer_st
            time1 = time.perf_counter()
            layer_st = current_layer    
            if ratio >= 1:
                self.num_early_exit_list[13] += 1
                return 13, exit1

        x = F.relu(self.bn12(self.conv12(x))) #512*512*4*4
        self.count_store[3] += 512*512*4*4
        current_layer = 15
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1) # 512 * (2*64-1)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[14])
            time2 = time.perf_counter()  #Time Sign
            self.count_store[3] += 512 * (2*64-1) + 64 * 19 + 4*4*512*64
            layer_end = current_layer           
            self.count_store[0] += time2 - time1
            self.layer_store[0] += layer_end - layer_st
            time1 = time.perf_counter()
            layer_st = current_layer    
            if ratio >= 1:
                self.num_early_exit_list[14] += 1
                return 14, exit1

        x = F.max_pool2d(x, 2, 2) #512*512*2*2
        self.count_store[3] += 512*512*2*2
        current_layer = 16
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[15])
            time2 = time.perf_counter()  #Time Sign
            self.count_store[3] += 2*2*512*64 + 64 * 19
            layer_end = current_layer         
            self.count_store[0] += time2 - time1
            self.layer_store[0] += layer_end - layer_st
            time1 = time.perf_counter()
            layer_st = current_layer    
            if ratio >= 1:
                self.num_early_exit_list[15] += 1
                return 15, exit1

        x = F.relu(self.bn13(self.conv13(x))) #512*512*2*2
        self.count_store[3] += 512*512*2*2
        current_layer = 17
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1)  # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[16])
            time2 = time.perf_counter()  #Time Sign
            self.count_store[3] += 2*2*512*64 + 64 * 19
            layer_end = current_layer          
            self.count_store[0] += time2 - time1
            self.layer_store[0] += layer_end - layer_st
            time1 = time.perf_counter()
            layer_st = current_layer    
            if ratio >= 1:
                self.num_early_exit_list[16] += 1
                return 16, exit1

        x = F.relu(self.bn14(self.conv14(x))) #512*512*2*2
        self.count_store[3] += 512*512*2*2
        current_layer = 18
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1)  # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[17])
            time2 = time.perf_counter()  #Time Sign
            self.count_store[3] += 2*2*512*64 + 64 * 19
            layer_end = current_layer          
            self.count_store[0] += time2 - time1
            self.layer_store[0] += layer_end - layer_st
            time1 = time.perf_counter()
            layer_st = current_layer    
            if ratio >= 1:
                self.num_early_exit_list[17] += 1
                return 17, exit1

        x = F.relu(self.bn15(self.conv15(x))) #512*512*2*2
        self.count_store[3] += 512*512*2*2
        current_layer = 19
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[18])
            time2 = time.perf_counter()  #Time Sign
            self.count_store[3] += 2*2*512*64 + 64 * 19
            layer_end = current_layer
            if count == 0:           
                self.count_store[0] += time2 - time1
                self.layer_store[0] += layer_end - layer_st
            time1 = time.perf_counter()
            layer_st = current_layer    
            if ratio >= 1:
                self.num_early_exit_list[18] += 1
                return 18, exit1
        
        x = F.relu(self.bn16(self.conv16(x))) #512*512*2*2
        self.count_store[3] += 512*512*2*2
        current_layer = 20
        if current_layer in self.possible_layer:
            if self.quant_switch == 1:
                x_exit_1 =  self.exit_3(self.dequant(x))
            else:
                x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1) # 64 * (2*10-1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[19])
            time2 = time.perf_counter()  #Time Sign
            self.count_store[3] += 2*2*512*64 + 64 * 19
            layer_end = current_layer         
            self.count_store[0] += time2 - time1
            self.layer_store[0] += layer_end - layer_st
            time1 = time.perf_counter()
            layer_st = current_layer    
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
            count += 512*512*2*2 + 2*2*512*64 + 64 * (2*10-1)
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
            count += 512*512*2*2 + 2*2*512*64 + 64 * (2*10-1)
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
            count += 512*512*2*2 + 2*2*512*64 + 64 * (2*10-1)
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
            count += 512*512*2*2 +  2*2*512*64 + 64 * (2*10-1)
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
            count += 512*512*2*2 + 2*2*512*64 + 64 * (2*10-1)
            if ratio >= 1:
                self.count_store.append(count)
                self.num_early_exit_list[19] += 1
                return 19, exit1
    
        x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.reshape(-1, 512)
        x = self.fc(x)
        count += 512*512*2*2*4 + 512*512*2*2 + 2*2*512*64 + 64 * (2*10-1) + 512*512*4*4*4 + 4*4*256*256 + 4*4*256*64 + 64 * (2*10-1) + 8*8*256*256*4 + 8*8*128*64 + 64 * 19
        self.count_store.append(count)
        self.original += 1
        x = self.dequant(x)
        return 20, x


class vgg19_exits_train_jump_svhn( vgg19_exits_eval_jump_svhn ):
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