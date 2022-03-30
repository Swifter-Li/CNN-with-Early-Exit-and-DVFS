from torch.nn import parameter
import torch.nn.functional as F
import torch
import torch.nn as nn



class LogisticConvBoF(nn.Module):
    
    def __init__(self, input_features, n_codewords, split_horizon=2):
        '''
        usually, 
            [input_features] = [the number of channels of the prescedent parameters]
            [n_codewords] = [the number of codewords, usually 64]
            [split_horizon] is the input image size
        '''
        super(LogisticConvBoF, self).__init__()
        self.input_features = input_features
        self.n_codewords = n_codewords
        self.codebook = nn.Conv2d(input_features, n_codewords, kernel_size=1)

        self.a = nn.Parameter(torch.FloatTensor(data=[1]))
        self.c = nn.Parameter(torch.FloatTensor(data=[0]))
        self.split_horizon = split_horizon

        self.n1 = nn.Parameter(torch.FloatTensor(data=[n_codewords]))
        self.n2 = nn.Parameter(torch.FloatTensor(data=[split_horizon]))

    def forward(self, input, eps=5e-16 ):
        # Step 1: Measure the similarity with each codeword
        '''
        note: 
            1. [codeword_length] == [feature_length]
            2. [image_size]^2 is the number of features for that image
        variables:
            1. [codebook's parameter]: [codeword_length] x [codeword_num] x [1] x [1]
                e.g. 32 x 64 x 1 x 1: 64 codewords, each with length 32 bits
            2. [input]:                [1] x [feature_length] x [image_size] x [image_size]
                e.g. 1 x 32 x 5 x 5: 25 image features, each with length 32 bits
            3. [x = codebook(input)]:  [1] x [codeword_num] x [image_size] x [image_size]
                e.g. 1 x 64 x 5 x 5: 64 x 25 products, which records the one-to-one product
                beteen 64 codewords and 25 features
            4. [x = F.tanh(...); x = (x+1)/2.0; x = (x / ...); ]: calculate the kernel and 
                calculate the normalized u_ijk
            5. [x = F.avg_pool2d]: instead of summing up u_ijk along dim1, only sum up some
                of them specified by kernel size
            6. [split_horizon] is the kernel size, as well as the stride of average pooling
                if the channel size is not a multiple of kernel size, the remainder will be discarded
                e.g. if channel size is 28, when applied by kernel of 6 * 6, the output channel
                size is 4 * 4
        '''

        x = self.codebook(input)

        # Step 2: Scale to ensure that the resulting value encodes the similarity
        x = F.tanh(self.a.expand_as(x) * x + self.c.expand_as(x))
        x = (x + 1) / 2.0

        # Step 3: Create the similarity vectors for each of the input feature vectors
        x = (x / (torch.sum(x, dim=1, keepdim=True)+ eps)) * self.n1

        # Step 4: Perform temporal pooling
        x = F.avg_pool2d(x, self.split_horizon) * self.n2
        x = x.reshape((x.size(0), -1))

        return x


if __name__ == '__main__':
    model = LogisticConvBoF( 256,64,1 )
    for name, parameter in model.named_parameters():
        print( name )