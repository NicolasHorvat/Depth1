import torch
import torch.nn as nn
import torchvision.models as models

'''
# todo trying different architectures

Models I could/should try:
            - Encoder/Decoder
                -> U-Net:               (https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
                -> ResNet based

            - FullyConvolutionalNetwork (FCN)

            - Transformers

'''


class UNetWithBins(nn.Module):
    """
    U-Net with adaptive depth bins and predicted depth range
    """
    def __init__(
        self,
        input_channels=3,
        num_layers=2,
        base_channels=12,
        num_bins=64
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_bins = num_bins

        # ---------- Encoder ------------
        self.encoders = nn.ModuleList() 
        in_en = input_channels
        out_en = base_channels

        for _ in range(num_layers):
            self.encoders.append(self.conv_relu_conv_relu(in_en, out_en))
            in_en = out_en
            out_en = in_en * 2

        self.max_pooling = nn.MaxPool2d(2, 2)

        # ---------- Bottleneck ----------
        self.bottleneck = self.conv_relu_conv_relu(in_en, out_en)

        # ---------- Decoder -------------
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        in_de = out_en
        out_de = in_en

        for _ in range(num_layers - 1, -1, -1):
            self.upconvs.append(nn.ConvTranspose2d(in_de, out_de, kernel_size=2, stride=2))
            self.decoders.append(self.conv_relu_conv_relu(in_de, out_de))
            in_de = out_de
            out_de = in_de // 2

        # ---------- Output layers ----------
        # Bin logits per pixel
        self.out_bins = nn.Conv2d(base_channels, num_bins, kernel_size=1)
        # Depth range (single scalar per image)
        self.out_range = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # global avg pooling -> [B, C, 1, 1]
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.ReLU()  # ensure positive range
        )

    def conv_relu_conv_relu(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc_feats = []

        # --- Encoder ---
        for enc in self.encoders:
            x = enc(x)
            enc_feats.append(x)
            x = self.max_pooling(x)

        # --- Bottleneck ---
        x = self.bottleneck(x)

        # --- Decoder ---
        for i in range(self.num_layers - 1, -1, -1):
            x = self.upconvs[self.num_layers - 1 - i](x)
            x = torch.cat([x, enc_feats[i]], dim=1)
            x = self.decoders[self.num_layers - 1 - i](x)

        # --- Outputs ---
        bin_logits = self.out_bins(x)  # [B, num_bins, H, W]
        depth_range = self.out_range(x).squeeze(-1).squeeze(-1)  # [B]

        # --- Convert bins to depth ---
        bin_probs = torch.softmax(bin_logits, dim=1)  # [B, num_bins, H, W]
        bin_centers = torch.linspace(0, 1, self.num_bins, device=x.device).view(1, self.num_bins, 1, 1)
        depth_map = (bin_probs * bin_centers).sum(dim=1, keepdim=True)
        depth_map = depth_map * depth_range.view(-1, 1, 1, 1)

        return depth_map
    

class UNetWithBins3chs(UNetWithBins):
    def __init__(self):
        super().__init__(input_channels=3, num_layers=2, base_channels=12, num_bins=64)

class UNetWithBins4chs(UNetWithBins):
    def __init__(self):
        super().__init__(input_channels=4, num_layers=2, base_channels=12, num_bins=64)



# class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
# class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)


class UNet(nn.Module):
    '''
    Modular U-net
    '''
    def __init__(
        self,
        input_channels = 3,
        output_channels = 1,
        num_layers = 2,              # number of encoder blocks
        base_channels = 3 * 4        # first block channel size
    ):
        super().__init__()
        self.num_layers = num_layers

        '''

        in_ch_1 -> out_ch_1                             up_in_1 -> out_ch_1
            in_ch_2 -> out_ch_2                     up_in_2 -> out_ch_2
                ...                             ...
                    in_ch_n -> out_ch_n     up_in_n -> out_ch_n
                                in_ch_bn -> out_ch_bn

        '''

        # ---------- Encoder ------------
        self.encoders = nn.ModuleList() 

        in_en = input_channels  # eg. 3
        out_en = base_channels  # eg. 12

        for _ in range(num_layers):
            self.encoders.append(self.conv_relu_conv_relu(in_en, out_en)) # add encoder block 
            in_en = out_en
            out_en = in_en * 2 

        self.max_pooling = nn.MaxPool2d(2, 2)

        # ---------- Bottleneck ----------
        self.bottleneck = self.conv_relu_conv_relu(in_en, out_en) # 18 -> 36


        # ---------- Decoder -------------
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        in_de = out_en # 36
        out_de = in_en # 18

        for _ in range(num_layers - 1, -1, -1):
            self.upconvs.append(nn.ConvTranspose2d(in_de, out_de, kernel_size=2, stride=2))
            self.decoders.append(self.conv_relu_conv_relu(in_de, out_de))
            in_de = out_de
            out_de = in_de // 2

        # --- Final output ---
        self.out_conv = nn.Conv2d(base_channels, output_channels, kernel_size=1)

    # conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2 -> maxPool 2x2
    def conv_relu_conv_relu(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): # tensor: [bs, Ch, H, W]
        enc_feats = []

        # --- Encoder path ---
        # print("Start output:", x.shape)                         # [4, 3, 608, 968]

        for enc in self.encoders:

            x = enc(x)
            enc_feats.append(x)
            # print("Encoder output:", x.shape)                   # 1: [4, 9, 608, 968], 2: [4, 18, 304, 484] *

            x = self.max_pooling(x)
            # print("Max Pooling:", x.shape)                      # 1: [4, 9, 304, 484], 2: [4, 18, 152, 242]

        # --- Bottleneck ---
        x = self.bottleneck(x)
        #print("Bottleneck output:", x.shape)                    # [4, 36, 152, 242]

        # --- Decoder path ---
        for i in range(self.num_layers - 1, -1, -1):            # backwards n n-1 n-2

            x = self.upconvs[self.num_layers - 1 - i](x)
            #print("After upconv:", x.shape)                     # [4, 18, 304, 484] * , 2: [4, 9, 608, 968]

            x = torch.cat([x, enc_feats[i]], dim=1)             
            #print("After concat:", x.shape)                     # [4, 36, 304, 484]   , 2: [4, 18, 608, 968]

            x = self.decoders[self.num_layers - 1 - i](x)
            #print("After decoder:", x.shape)                    # [4, 18, 304, 484]   , 2: [4, 9, 608, 968]

        return self.out_conv(x)
    

# ------------- 3Channels --------------
class UNet_3inChs_1L_12bc(UNet):
    def __init__(self):
        super().__init__(input_channels = 3, output_channels = 1, num_layers = 1, base_channels = 12)

class UNet_3inChs_2L_12bc(UNet):
    def __init__(self):
        super().__init__(input_channels = 3, output_channels = 1, num_layers = 2, base_channels = 12)

class UNet_3inChs_3L_12bc(UNet):
    def __init__(self):
        super().__init__(input_channels = 3, output_channels = 1, num_layers = 3, base_channels = 12)


class UNet_3inChs_1L_24bf(UNet):
    def __init__(self):
        super().__init__(input_channels = 3, output_channels = 1, num_layers = 1, base_channels = 24)

class UNet_3inChs_2L_24bf(UNet):
    def __init__(self):
        super().__init__(input_channels = 3, output_channels = 1, num_layers = 2, base_channels = 24)

class UNet_3inChs_3L_24bf(UNet):
    def __init__(self):
        super().__init__(input_channels = 3, output_channels = 1, num_layers = 3, base_channels = 24)

# ------------- 4Channels --------------

class UNet_4inChs_1L_12bc(UNet):
    def __init__(self):
        super().__init__(input_channels = 4, output_channels = 1, num_layers = 1, base_channels = 12)

class UNet_4inChs_2L_12bc(UNet):
    def __init__(self):
        super().__init__(input_channels = 4, output_channels = 1, num_layers = 2, base_channels = 12)

class UNet_4inChs_3L_12bc(UNet):
    def __init__(self):
        super().__init__(input_channels = 4, output_channels = 1, num_layers = 3, base_channels = 12)


class UNet_4inChs_1L_24bf(UNet):
    def __init__(self):
        super().__init__(input_channels = 4, output_channels = 1, num_layers = 1, base_channels = 24)

class UNet_4inChs_2L_24bf(UNet):
    def __init__(self):
        super().__init__(input_channels = 4, output_channels = 1, num_layers = 2, base_channels = 24)

class UNet_4inChs_3L_24bf(UNet):
    def __init__(self):
        super().__init__(input_channels = 4, output_channels = 1, num_layers = 3, base_channels = 24)





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
        num_channels_1 = input_channels * 16  #  3 * 16  = 48
        num_channels_2 = num_channels_1 * 2   # 48 *  2  = 96
        num_channels_3 = num_channels_2 * 2   # 96 *  2  = 192
        num_features_bn = num_channels_3 * 2   # 192 *  2  = 384
        output_channels = 1

        
        # Encoder:
        # conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        self.enc1 = nn.Sequential( 
            nn.Conv2d(input_channels, num_channels_1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels_1, num_channels_1, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        self.enc2 = nn.Sequential(
            nn.Conv2d(num_channels_1, num_channels_2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels_2, num_channels_2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        self.enc3 = nn.Sequential(
            nn.Conv2d(num_channels_2, num_channels_3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels_3, num_channels_3, kernel_size=3, padding=1),
            nn.ReLU()
        )


        # max-pool 2x2
        self.max_pool = nn.MaxPool2d(2, 2)


        # Bottleneck
        self.bn = nn.Sequential(
            nn.Conv2d(num_channels_3, num_features_bn, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features_bn, num_features_bn, kernel_size=3, padding=1),
            nn.ReLU()
        )


        # Decoder:
        # conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        self.up3 = nn.ConvTranspose2d(num_features_bn, num_channels_3, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(num_features_bn, num_channels_3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels_3, num_channels_3, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(num_channels_3, num_channels_2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(num_channels_3, num_channels_2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels_2, num_channels_2, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(num_channels_2, num_channels_1, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(num_channels_2, num_channels_1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels_1, num_channels_1, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Final conv to get depth map
        self.out = nn.Conv2d(num_channels_1, output_channels, kernel_size=1)


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
    
class UNet_4channels_256(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        Encoder:
        -> conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        -> max-pool 2x2
        -> conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        -> max-pool 2x2

        # Bottleneck
        -> conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2

        Decoder:
        -> up-conv 2x2
        -> conv 3x3 -> relu 2x2 -> conv 3x3 -> relu 2x2
        Out:


        en1                     de1
            en2             de2
                    bn

        '''

        input_channels=4
        num_features_1 = input_channels * 16  #  4 * 16  = 64
        num_features_2 = num_features_1 * 2   # 48 *  2  = 128
        num_features_bn = num_features_2 * 2   # 192 *  2  = 256
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


        # max-pool 2x2
        self.max_pool = nn.MaxPool2d(2, 2)


        # Bottleneck
        self.bn = nn.Sequential(
            nn.Conv2d(num_features_2, num_features_bn, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features_bn, num_features_bn, kernel_size=3, padding=1),
            nn.ReLU()
        )


        # Decoder:

        self.up2 = nn.ConvTranspose2d(num_features_bn, num_features_2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(num_features_bn, num_features_2, kernel_size=3, padding=1),
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

        # Bottleneck
        bn = self.bn(self.max_pool(e2))

        # Decoder
        d2 = self.up2(bn)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # connection 2

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # connection 1

        # Output
        output = self.out(d1)
        return output

class UNet_4channels_512(nn.Module):
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





## not working yet
class ResNetUNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False

        # Split ResNet into blocks for skip connections
        self.initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.enc1 = resnet.layer1
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4

        # Decoder with upsampling and concatenation
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(64+64, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x0 = self.initial(x)
        x1 = self.maxpool(x0)
        e1 = self.enc1(x1)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d4 = self.up4(e4)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x0], dim=1))

        out = self.out(d1)
        return out
