import torch.nn as nn
import torch.nn.functional as F
import torch
from nikolaos.bof_utils import LogisticConvBoF
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
        self.target_layer = 1

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
        if self.target_layer == 1:
            x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / self.beta * self.activation_threshold_list[0]
            if ratio >= 1:
                self.num_early_exit_list[0] += 1
                return 0, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 

        x = F.relu(self.bn2(self.conv2(x)))
        if self.target_layer == 2:
            x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[1])
            if ratio >= 1:
                self.num_early_exit_list[1] += 1
                return 1, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2]
            
        x = F.max_pool2d(x, 2, 2)
        if self.target_layer == 3:
            x_exit_1 = self.exit_0(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[2])
            if ratio >= 1:
                self.num_early_exit_list[2] += 1
                return 2, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 

        x = F.relu(self.bn3(self.conv3(x)))
        if self.target_layer == 4:
            x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[3])
            if ratio >= 1:
                self.num_early_exit_list[3] += 1
                return 3, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 
            
        x = F.relu(self.bn4(self.conv4(x)))
        if self.target_layer == 5:
            x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[4])
            if ratio >= 1:
                self.num_early_exit_list[4] += 1
                return 4, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 

        x = F.max_pool2d(x, 2, 2)
        if self.target_layer == 6:
            x_exit_1 = self.exit_1(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            if ratio >= 1:
                self.num_early_exit_list[5] += 1
                return 5, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 

        x = F.relu(self.bn5(self.conv5(x)))
        if self.target_layer == 7:
            x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[6])
            if ratio >= 1:
                self.num_early_exit_list[6] += 1
                return 6, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 

        x = F.relu(self.bn6(self.conv6(x)))
        if self.target_layer == 8:
            x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[7])
            if ratio >= 1:
                self.num_early_exit_list[7] += 1
                return 7, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 

        x = F.relu(self.bn7(self.conv7(x)))
        if self.target_layer == 9:
            x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[8])
            if ratio >= 1:
                self.num_early_exit_list[8] += 1
                return 8, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 
        
        x = F.relu(self.bn8(self.conv8(x)))
        if self.target_layer == 10:
            x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[9])
            if ratio >= 1:
                self.num_early_exit_list[9] += 1
                return 9, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 

        x = F.max_pool2d(x, 2, 2)
        if self.target_layer == 11:
            x_exit_1 = self.exit_2(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
            if ratio >= 1:
                self.num_early_exit_list[10] += 1
                return 10, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 
        
        x = F.relu(self.bn9(self.conv9(x)))
        if self.target_layer == 12:
            x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[11])
            if ratio >= 1:
                self.num_early_exit_list[11] += 1
                return 11, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num 

        x = F.relu(self.bn10(self.conv10(x)))
        if self.target_layer == 13:
            x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[12])
            if ratio >= 1:
                self.num_early_exit_list[12] += 1
                return 12, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
             
        x = F.relu(self.bn11(self.conv11(x)))
        if self.target_layer == 14:
            x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[13])
            if ratio >= 1:
                self.num_early_exit_list[13] += 1
                return 13, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn12(self.conv12(x)))
        if self.target_layer == 15:
            x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[14])
            if ratio >= 1:
                self.num_early_exit_list[14] += 1
                return 14, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.max_pool2d(x, 2, 2)
        if self.target_layer == 16:
            x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[15])
            if ratio >= 1:
                self.num_early_exit_list[15] += 1
                return 15, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn13(self.conv13(x)))
        if self.target_layer == 17:
            x_exit_1 = self.exit_3(x)
         #   x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[16])
            if ratio >= 1:
                self.num_early_exit_list[16] += 1
                return 16, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn14(self.conv14(x)))
        if self.target_layer == 18:
            x_exit_1 = self.exit_3(x)
         #   x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[17])
            if ratio >= 1:
                self.num_early_exit_list[17] += 1
                return 17, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn15(self.conv15(x)))
        if self.target_layer == 19:
            x_exit_1 = self.exit_3(x)
       #     x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[18])
            if ratio >= 1:
                self.num_early_exit_list[18] += 1
                return 18, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
        
        x = F.relu(self.bn16(self.conv16(x)))
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
    '''

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

        self.statics = [([0] * 20) for i in range(11)]
        self.total = [0] * 11

        # the beta coefficient used for accuracy-speed trade-off, the higher the more accurate
        self.beta = 0
        self.target_layer = 1
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

    
    def forward( self, x ):
        flag = 0
        x = F.relu(self.bn1(self.conv1(x)))
        current_layer = 1
        if self.target_layer == 1:
            x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[0])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[0] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 0, exit1
            self.total[flag] = self.total[flag] + 1
            self.ratio_store.append(ratio.item())
            self.entropy_store.append(self._calculate_cross_entropy(exit1).item())
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
            x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[1])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[1] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 0, exit1
            self.ratio_store.append(ratio.item())
            self.total[flag] = self.total[flag] + 1
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
            x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_2_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[2])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[2] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 2, exit1
            self.ratio_store.append(ratio.item())
            self.total[flag] = self.total[flag] + 1
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
            x_exit_1 = self.exit_2_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            flag = int(ratio/0.1)
            if ratio >= 1:
                self.num_early_exit_list[5] += 1
              #  self.statics[flag][current_layer-self.target_layer] = self.statics[flag][current_layer-self.target_layer] + 1
                return 5, exit1
            self.ratio_store.append(ratio.item())
            self.total[flag] = self.total[flag] + 1
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
        x_exit_1 = self.exit_3(x)
         #   x_exit_1 = self.exit_0_fc(x_exit_1)
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
        x_exit_1 = self.exit_3(x)
       #     x_exit_1 = self.exit_0_fc(x_exit_1)
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
        x_exit_1 = self.exit_3(x)
       #     x_exit_1 = self.exit_0_fc(x_exit_1)
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
        x = x.view(-1, 512*9)
        x = self.fc(x)
        self.original += 1
        current_layer += 1
        self.statics[flag][current_layer-self.target_layer]+= 1
        return 20, x
    '''
    def forward( self, x ):
        x = F.relu(self.bn1(self.conv1(x)))
        if self.target_layer == 1:
            x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / self.beta * self.activation_threshold_list[0]
            if ratio >= 1:
                self.num_early_exit_list[0] += 1
                return 0, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 

        x = F.relu(self.bn2(self.conv2(x)))
        if self.target_layer == 2:
            x_exit_1 = self.exit_0(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[1])
            if ratio >= 1:
                self.num_early_exit_list[1] += 1
                return 1, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2]
            
        x = F.max_pool2d(x, 2, 2)
        if self.target_layer == 3:
            x_exit_1 = self.exit_0(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[2])
            if ratio >= 1:
                self.num_early_exit_list[2] += 1
                return 2, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 

        x = F.relu(self.bn3(self.conv3(x)))
        if self.target_layer == 4:
            x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[3])
            if ratio >= 1:
                self.num_early_exit_list[3] += 1
                return 3, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 
            
        x = F.relu(self.bn4(self.conv4(x)))
        if self.target_layer == 5:
            x_exit_1 = self.exit_1(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[4])
            if ratio >= 1:
                self.num_early_exit_list[4] += 1
                return 4, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 

        x = F.max_pool2d(x, 2, 2)
        if self.target_layer == 6:
            x_exit_1 = self.exit_1(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[5])
            if ratio >= 1:
                self.num_early_exit_list[5] += 1
                return 5, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 

        x = F.relu(self.bn5(self.conv5(x)))
        if self.target_layer == 7:
            x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[6])
            if ratio >= 1:
                self.num_early_exit_list[6] += 1
                return 6, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 

        x = F.relu(self.bn6(self.conv6(x)))
        if self.target_layer == 8:
            x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[7])
            if ratio >= 1:
                self.num_early_exit_list[7] += 1
                return 7, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 

        x = F.relu(self.bn7(self.conv7(x)))
        if self.target_layer == 9:
            x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[8])
            if ratio >= 1:
                self.num_early_exit_list[8] += 1
                return 8, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 
        
        x = F.relu(self.bn8(self.conv8(x)))
        if self.target_layer == 10:
            x_exit_1 = self.exit_2(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[9])
            if ratio >= 1:
                self.num_early_exit_list[9] += 1
                return 9, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 

        x = F.max_pool2d(x, 2, 2)
        if self.target_layer == 11:
            x_exit_1 = self.exit_2(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[10])
            if ratio >= 1:
                self.num_early_exit_list[10] += 1
                return 10, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2] 
        
        x = F.relu(self.bn9(self.conv9(x)))
        if self.target_layer == 12:
            x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[11])
            if ratio >= 1:
                self.num_early_exit_list[11] += 1
                return 11, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num 

        x = F.relu(self.bn10(self.conv10(x)))
        if self.target_layer == 13:
            x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[12])
            if ratio >= 1:
                self.num_early_exit_list[12] += 1
                return 12, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
             
        x = F.relu(self.bn11(self.conv11(x)))
        if self.target_layer == 14:
            x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[13])
            if ratio >= 1:
                self.num_early_exit_list[13] += 1
                return 13, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn12(self.conv12(x)))
        if self.target_layer == 15:
            x_exit_1 = self.exit_3(x)
            x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[14])
            if ratio >= 1:
                self.num_early_exit_list[14] += 1
                return 14, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.max_pool2d(x, 2, 2)
        if self.target_layer == 16:
            x_exit_1 = self.exit_3(x)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[15])
            if ratio >= 1:
                self.num_early_exit_list[15] += 1
                return 15, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn13(self.conv13(x)))
        if self.target_layer == 17:
            x_exit_1 = self.exit_3(x)
         #   x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[16])
            if ratio >= 1:
                self.num_early_exit_list[16] += 1
                return 16, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn14(self.conv14(x)))
        if self.target_layer == 18:
            x_exit_1 = self.exit_3(x)
         #   x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[17])
            if ratio >= 1:
                self.num_early_exit_list[17] += 1
                return 17, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num

        x = F.relu(self.bn15(self.conv15(x)))
        if self.target_layer == 19:
            x_exit_1 = self.exit_3(x)
       #     x_exit_1 = self.exit_0_fc(x_exit_1)
            exit1 = self.exit_1_fc(x_exit_1)
            ratio = self._calculate_max_activation( exit1 ) / (self.beta * self.activation_threshold_list[18])
            if ratio >= 1:
                self.num_early_exit_list[18] += 1
                return 18, exit1
            elif ratio > 0.7:
                self.target_layer+=Jump_step[0]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            elif ratio > 0.4:
                self.target_layer+=Jump_step[1]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
            else:
                self.target_layer+=Jump_step[2]
                if self.target_layer > self.exit_num:
                    self.target_layer = self.exit_num
        
        x = F.relu(self.bn16(self.conv16(x)))
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
    '''

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

        return (exit1, exit2, exit3, exit4, exit5, exit6, exit7, exit8, exit9, exit10, exit11, exit12, exit13, exit14, exit15, exit16, exit17, exit18, exit19, exit20)        