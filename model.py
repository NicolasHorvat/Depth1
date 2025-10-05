import torch
import torch.nn as nn

'''
# todo trying different architectures

Models I could/should try:
            - Encoder/Decoder
                -> U-Net:               (https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
                -> ResNet based

            - FullyConvolutionalNetwork (FCN)

            - Transformers

'''


# class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
# class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

class UNet_3channels(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        Encoder:
        -> conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        -> max-pool 2x2
        -> conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        -> max-pool 2x2
        -> conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        -> max-pool 2x2

        # Bottleneck
        -> conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2

        Decoder:
        -> up-conv 2x2
        -> conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        -> up-conv 2x2
        -> conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        Out:


        en1                     de1
            en2             de2
                en3     de3
                    bn

        '''

        input_channels=3
        num_features_1 = input_channels * 16  #  3 * 16  = 48
        num_features_2 = num_features_1 * 2   # 48 *  2  = 96
        num_features_3 = num_features_2 * 2   # 96 *  2  = 192
        num_features_bn = num_features_3 * 2   # 192 *  2  = 384
        output_channels = 1

        
        # Encoder:
        # conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        self.enc1 = nn.Sequential( 
            nn.Conv2d(input_channels, num_features_1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features_1, num_features_1, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        self.enc2 = nn.Sequential(
            nn.Conv2d(num_features_1, num_features_2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features_2, num_features_2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        self.enc3 = nn.Sequential(
            nn.Conv2d(num_features_2, num_features_3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features_3, num_features_3, kernel_size=3, padding=1),
            nn.ReLU()
        )


        # max-pool 2x2
        self.max_pool = nn.MaxPool2d(2, 2)


        # Bottleneck
        self.bn = nn.Sequential(
            nn.Conv2d(num_features_3, num_features_bn, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features_bn, num_features_bn, kernel_size=3, padding=1),
            nn.ReLU()
        )


        # Decoder:
        # conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        self.up3 = nn.ConvTranspose2d(num_features_bn, num_features_3, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(num_features_bn, num_features_3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features_3, num_features_3, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(num_features_3, num_features_2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(num_features_3, num_features_2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features_2, num_features_2, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(num_features_2, num_features_1, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(num_features_2, num_features_1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features_1, num_features_1, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Final conv to get depth map
        self.out = nn.Conv2d(num_features_1, output_channels, kernel_size=1)


    def forward(self, x):
            
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.max_pool(e1))
        e3 = self.enc3(self.max_pool(e2))

        # Bottleneck
        bn = self.bn(self.max_pool(e3))

        # Decoder
        d3 = self.up3(bn)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))  # connection 3

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # connection 2

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # connection 1

        # Output
        output = self.out(d1)
        return output
    
class UNet_4channels(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        Encoder:
        -> conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        -> max-pool 2x2
        -> conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        -> max-pool 2x2
        -> conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        -> max-pool 2x2

        # Bottleneck
        -> conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2

        Decoder:
        -> up-conv 2x2
        -> conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        -> up-conv 2x2
        -> conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        Out:


        en1                     de1
            en2             de2
                en3     de3
                    bn

        '''

        input_channels=4
        num_features_1 = input_channels * 16  #  4 * 16  = 64
        num_features_2 = num_features_1 * 2   # 48 *  2  = 128
        num_features_3 = num_features_2 * 2   # 96 *  2  = 256
        num_features_bn = num_features_3 * 2   # 192 *  2  = 512
        output_channels = 1

        
        # Encoder:
        # conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        self.enc1 = nn.Sequential( 
            nn.Conv2d(input_channels, num_features_1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features_1, num_features_1, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        self.enc2 = nn.Sequential(
            nn.Conv2d(num_features_1, num_features_2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features_2, num_features_2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        self.enc3 = nn.Sequential(
            nn.Conv2d(num_features_2, num_features_3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features_3, num_features_3, kernel_size=3, padding=1),
            nn.ReLU()
        )


        # max-pool 2x2
        self.max_pool = nn.MaxPool2d(2, 2)


        # Bottleneck
        self.bn = nn.Sequential(
            nn.Conv2d(num_features_3, num_features_bn, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features_bn, num_features_bn, kernel_size=3, padding=1),
            nn.ReLU()
        )


        # Decoder:
        # conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        self.up3 = nn.ConvTranspose2d(num_features_bn, num_features_3, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(num_features_bn, num_features_3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features_3, num_features_3, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(num_features_3, num_features_2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(num_features_3, num_features_2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features_2, num_features_2, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(num_features_2, num_features_1, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(num_features_2, num_features_1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features_1, num_features_1, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Final conv to get depth map
        self.out = nn.Conv2d(num_features_1, output_channels, kernel_size=1)


    def forward(self, x):
            
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.max_pool(e1))
        e3 = self.enc3(self.max_pool(e2))

        # Bottleneck
        bn = self.bn(self.max_pool(e3))

        # Decoder
        d3 = self.up3(bn)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))  # connection 3

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # connection 2

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # connection 1

        # Output
        output = self.out(d1)
        return output
