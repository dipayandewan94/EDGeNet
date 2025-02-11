import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List
from torchvision.ops import StochasticDepth
from ptflops import get_model_complexity_info
from pytorch_tcn import TCN, TemporalBlock, CausalConv1d # type: ignore

class PixelShuffle1D(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x
    
class DiConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DiConvModule, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.kernel_size = kernel_size

    def forward(self, x):
        input_size = x.size(-1)
        output_size = input_size//2
        dilation_rate = ((input_size - 1) - (output_size - 1) * 1 + self.conv.kernel_size[0] - 1)//(self.conv.kernel_size[0] - 1)-1
        out = nn.functional.conv1d(x, self.conv.weight, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=self.conv.padding, dilation=dilation_rate)
        return out

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ConvBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, shortcut=False, pool=True):
        '''
        pool_mode can be False (no pooling) or True ('maxpool')
        '''
        super(ConvBlock, self).__init__()
        if shortcut==True: self.shortcut = nn.Sequential(conv1x1(in_c, out_c), nn.BatchNorm1d(out_c))
        else: self.shortcut=False
        pad = (k_sz - 1) // 2

        block = []
        if pool: self.pool = nn.MaxPool1d(kernel_size=2)
        else: self.pool = False

        block.append(nn.Conv1d(in_c, out_c, kernel_size=k_sz, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm1d(out_c))

        block.append(nn.Conv1d(out_c, out_c, kernel_size=k_sz, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm1d(out_c))

        self.block = nn.Sequential(*block)
    def forward(self, x):
        if self.pool: x = self.pool(x)
        out = self.block(x)
        if self.shortcut: return out + self.shortcut(x)
        else: return out
        
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, k_sz=3):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_c)
        #self.relu = nn.ReLU(inplace=True)
        self.mish = nn.GELU()
        self.conv1 = nn.Conv1d(in_c, out_c // 4 , 1)
        self.bn2 = nn.BatchNorm1d(out_c // 4)
        #self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_c // 4, out_c // 4, kernel_size=k_sz, padding='same')
        # self.conv3 = nn.Conv2d(out_c * 4, out_c, 3, padding='same', dilation=dilation[1])
        # self.conv4 = nn.Conv2d(out_c * 4, out_c, 3, padding='same', dilation=dilation[2])
        
        
        #self.conv2 = nn.Conv2d(output_channels/4, output_channels/4, 3, stride, padding = 1, bias = False)
        self.dropout = nn.Dropout(0.2)
        self.bn3 = nn.BatchNorm1d(out_c // 4)
        #self.relu = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv1d(out_c // 4, out_c, 1, 1, bias = False)
        self.conv6 = nn.Conv1d(in_c, out_c , 1, 1, padding='same', bias = False)
        
    def forward(self, x):
        #residual = x
        #out = self.bn1(x)
        #out = self.mish(out)
        out = self.conv1(x)
        #out = self.bn2(out)
        out = self.mish(out)
        out = self.conv2(out)
        # out2 = self.conv3(out)
        # out3 = self.conv4(out)
        # out = torch.add(torch.add(out1,out2),out3)
        #out =  self.conv2(out)+ self.conv3(out)+ self.conv4(out)
        out = self.bn3(out)
        out = self.dropout(out)
        out = self.mish(out)
        out = self.conv5(out)
        
       
        #if (self.input_channels != self.output_channels) or (self.stride !=1 ):
        residual = self.conv6(x)
        out += residual
        return out

class Trunk(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, residual= True, causal=True):
        '''
        pool_mode can be False (no pooling) or True ('maxpool')
        '''
        super(Trunk, self).__init__()
        if residual:
            if causal:
                self.conv = TCN(in_c,[out_c,out_c],kernel_size=k_sz)
                '''self.conv = nn.Sequential(
                    TemporalBlock(in_c, out_c, kernel_size=k_sz, dilation = 2, causal= True),
                    TemporalBlock(out_c, out_c, kernel_size=k_sz, dilation = 2, causal= True),
                )'''
            else:    
                self.conv = nn.Sequential(
                    ResBlock(in_c, out_c, k_sz=k_sz),
                    ResBlock(out_c, out_c, k_sz=k_sz),
                )
        else:
            if causal:
                self.conv =  nn.Sequential(
                    CausalConv1d(in_c, out_c, kernel_size=k_sz),
                    nn.BatchNorm1d(out_c),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    CausalConv1d(out_c, out_c, kernel_size=k_sz),
                    nn.BatchNorm1d(out_c),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                )
            else:    
                self.conv = nn.Sequential(
                    nn.Conv1d(in_c, out_c, kernel_size=k_sz, padding='same'),
                    nn.BatchNorm1d(out_c),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Conv1d(out_c, out_c, kernel_size=k_sz, padding='same'),
                    nn.BatchNorm1d(out_c),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                )
    def forward(self, x):
        #print(x.shape)
        out = self.conv(x)
        #out = F.dropout1d(out, 0.1)
        return out
            
class AttConvBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, shortcut=False, pool='pool', attention=False, residual=True, causal=True, conv_type='normal'):
        '''
        pool_mode can be False (no pooling) or True ('maxpool')
        '''
        super(AttConvBlock, self).__init__()
        self.conv_type = conv_type
        if shortcut==True: self.shortcut = nn.Sequential(conv1x1(in_c, out_c), nn.BatchNorm1d(out_c))
        else: self.shortcut=False
        pad = (k_sz - 1) // 2

        if pool=='pool':
            self.pool = nn.MaxPool1d(kernel_size=2)
        elif pool=='conv':
            self.pool = nn.Conv1d(in_c, in_c, kernel_size=2, stride=2)
        else:
            self.pool = False

        self.conv = Trunk(in_c, out_c, k_sz=k_sz, residual=residual, causal=causal)
        
        if attention==True:
            self.mpool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

            self.softmax1_blocks = self._get_conv_layer(in_c, out_c, k_sz, dilation=6)

            self.skip1_connection_residual_block = self._get_conv_layer(out_c, out_c, k_sz)

            self.mpool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

            self.softmax2_blocks = self._get_conv_layer(out_c, out_c, k_sz, dilation=4)

            self.skip2_connection_residual_block = self._get_conv_layer(out_c, out_c, k_sz)

            self.mpool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

            self.softmax3_blocks = nn.Sequential(
                self._get_conv_layer(out_c, out_c, k_sz, dilation=2),
                self._get_conv_layer(out_c, out_c, k_sz, dilation=2)
            )

            self.interpolation3 = nn.Upsample(scale_factor=2)

            self.softmax4_blocks = self._get_conv_layer(out_c, out_c, k_sz, dilation=4)

            self.interpolation2 = nn.Upsample(scale_factor=2)

            self.softmax5_blocks = self._get_conv_layer(out_c, out_c, k_sz, dilation=6)

            self.interpolation1 = nn.Upsample(scale_factor=2)

            self.softmax6_blocks = nn.Sequential(
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
                self._get_conv_layer(out_c, out_c, k_sz=1),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
                self._get_conv_layer(out_c, out_c, k_sz=1),
                nn.Sigmoid()
            )

            self.last_blocks = self._get_conv_layer(out_c, out_c, k_sz)

    def _get_conv_layer(self, in_c, out_c, k_sz, dilation=1):
        if self.conv_type == 'depthwise':
            return nn.Conv1d(in_c, out_c, kernel_size=k_sz, padding='same', dilation=dilation, groups=in_c)
        else:
            return nn.Conv1d(in_c, out_c, kernel_size=k_sz, padding='same', dilation=dilation)

    def forward(self, x, attention=False):
        input_size = x.size(-1)
        dilation_rate = input_size // 4
        if self.pool: x = self.pool(x)
        out_trunk = self.conv(x)
        if attention==True:
            out_mpool1 = self.mpool1(x)
            out_softmax1 = self.softmax1_blocks(out_mpool1)
            out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
            out_mpool2 = self.mpool2(out_softmax1)
            out_softmax2 = self.softmax2_blocks(out_mpool2)
            out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
            out_mpool3 = self.mpool3(out_softmax2)
            out_softmax3 = self.softmax3_blocks(out_mpool3)
            out_interp3 = self.interpolation3(out_softmax3)
            out = torch.add(out_interp3, out_skip2_connection)
            out_softmax4 = self.softmax4_blocks(out)
            out_interp2 = self.interpolation2(out_softmax4)
            out = torch.add(out_interp2, out_skip1_connection)
            out_softmax5 = self.softmax5_blocks(out)
            out_interp1 = self.interpolation1(out_softmax5)
            out_softmax6 = self.softmax6_blocks(out_interp1)
            out = torch.multiply((1 + out_softmax6), out_trunk)
            out = self.last_blocks(out)
        else:
            out = out_trunk
        if self.shortcut: return out + self.shortcut(x)
        else: return out
        
        
class UpsampleBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, up_mode='transp_conv', upsample = True):
        super(UpsampleBlock, self).__init__()
        block = []
        if upsample:
            if up_mode == 'transp_conv':
                block.append(nn.ConvTranspose1d(in_c, out_c, kernel_size=2, stride=2))
            elif up_mode == 'up_conv':
                block.append(nn.Upsample(scale_factor=2))
                block.append(nn.Conv1d(in_c, out_c, kernel_size=1))
            elif up_mode == 'pixelshuffle':
                block.append(nn.Conv1d(in_c, 2*out_c, kernel_size=1))
                block.append(PixelShuffle1D(2))
            else:
                raise Exception('Upsampling mode not supported')
        else:
            block.append(nn.Conv1d(in_c, out_c, kernel_size=1))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out
    
class DoubleAttBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, shortcut= True, pool = 'pool', attention = True, residual= True, causal=True, conv_encoder='depthwise'):
        super(DoubleAttBlock, self).__init__()
        
        self.attention = attention
        self.block1 = AttConvBlock(in_c, in_c, k_sz=k_sz,
                              shortcut=shortcut, pool=False, attention=False, residual= residual, causal=causal, conv_type=conv_encoder)
        self.block2 = AttConvBlock(in_c, out_c, k_sz=k_sz,
                              shortcut=shortcut, pool=pool, attention=self.attention, residual= residual, causal=causal, conv_type=conv_encoder)

    def forward(self, x):
        out = self.block1(x, attention = False)
        out = self.block2(out, attention = self.attention)
        return out

class ConvBridgeBlock(torch.nn.Module):
    def __init__(self, channels, k_sz=3):
        super(ConvBridgeBlock, self).__init__()
        # pad = (k_sz - 1) // 2
        # block=[]

        # block.append(nn.Conv2d(channels, channels, kernel_size=k_sz, padding=pad))
        # block.append(nn.ReLU())
        # block.append(nn.BatchNorm2d(channels))

        # self.block = nn.Sequential(*block)
        self.block = ResBlock(channels, channels)

    def forward(self, x):
        out = self.block(x)
        return out

class UpConvBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, up_mode='up_conv', conv_bridge=False, shortcut=False, 
                 skip_conn=True, residual= True, causal=True, upsample=True):
        super(UpConvBlock, self).__init__()
        self.conv_bridge = conv_bridge
        self.skip_conn=skip_conn
        self.up_layer = UpsampleBlock(in_c, out_c, up_mode=up_mode, upsample=upsample)
        self.conv_layer1 = AttConvBlock(out_c, out_c, k_sz=k_sz, shortcut=shortcut, pool=False, attention= True, residual= residual, causal=causal)
        self.conv_layer2 = AttConvBlock(2 * out_c, out_c, k_sz=k_sz, shortcut=shortcut, pool=False, attention= True, residual= residual, causal=causal)
        if self.conv_bridge:
            self.conv_bridge_layer = ConvBridgeBlock(out_c, k_sz=k_sz)

    def forward(self, x, skip=None):
        #print(x.shape)
        up = self.up_layer(x)
        #print(up.shape)
        up = self.conv_layer1(up, attention = True)
        #print(up.shape)
        #skip = torch.multiply((1 + up), skip)
        if skip is not None and self.skip_conn:
            if self.conv_bridge:
                skip = self.conv_bridge_layer(skip)
                #print(skip.shape, up.shape)
                skip = torch.multiply((1 + up), skip)
                out = torch.cat([up, skip], dim=1) 
            else:
                skip = torch.multiply((1 + up), skip)
                out = torch.cat([up, skip], dim=1)
            out = self.conv_layer2(out, attention = True)
        else:
            out=up
        return out
    
class EDGeNetEncoder(nn.Module):
    def __init__(
        self,
        in_c: int,
        k_sz: int,
        layers: List[int],
        shortcut: bool = True,
        pool = 'pool',
        residual= True,
        causal=True,
        conv_encoder='depthwise',
        downsample: List[bool] = None
    ):
        super().__init__()
        self.first = ConvBlock(in_c=in_c, out_c=layers[0], k_sz=k_sz,
                               shortcut=shortcut, pool=False)
        
        # in_out_widths = list(zip(layers, layers[1:]))
        # create drop paths probabilities (one for each stage)
        # drop_probs = [x.item() for x in torch.linspace(0, drop_p, sum(depths))]
        
        self.down_path = nn.ModuleList()
        for i in range(len(layers) - 1):
            if i <= 7:
                block = DoubleAttBlock(in_c=layers[i], out_c=layers[i + 1], k_sz=k_sz,
                                shortcut=shortcut,pool=pool if downsample[i] else False, attention=True, residual= residual, causal=causal, conv_encoder=conv_encoder)
            else:
                block = DoubleAttBlock(in_c=layers[i], out_c=layers[i + 1], k_sz=k_sz,
                                shortcut=shortcut,pool=pool if downsample[i] else False, attention=False, residual= residual, causal=causal, conv_encoder=conv_encoder)
            self.down_path.append(block)
        
    def forward(self, x):
        x = self.first(x)
        down_activations = []
        for i, down in enumerate(self.down_path):
            down_activations.append(x)
            #print(x.shape)
            x = down(x)
            #print(x.shape)
        down_activations.reverse()
        return x, down_activations

class EDGeNetDecoder(nn.Module):
    def __init__(
        self,
        n_classes: int,
        k_sz: int,
        layers: List[int],
        upsample: List[bool],
        up_mode='up_conv',
        conv_bridge:bool = True,
        shortcut:bool = True,
        skip_conn:bool = True,
        residual= True, 
        causal=True,
    ):
        super().__init__()
        
        self.up_path = nn.ModuleList()
        #print(layers)
        reversed_layers = list(reversed(layers))
        for i in range(len(layers) - 1):
            block = UpConvBlock(in_c=reversed_layers[i], out_c=reversed_layers[i + 1], k_sz=k_sz,
                                up_mode=up_mode, conv_bridge=conv_bridge, shortcut=shortcut, skip_conn=skip_conn, residual= residual, causal=causal, upsample=upsample[i])
            self.up_path.append(block)
            
        self.final = nn.Conv1d(layers[0], n_classes, kernel_size=1)
        

    def forward(self, x, down_activations):
        for i, up in enumerate(self.up_path):
            x = up(x, down_activations[i])
        return self.final(x)


class EDGeNet(nn.Module):
    def __init__(self, in_c, n_classes, layers, downsample, k_sz=3, up_mode='pixelshuffle', pool='pool', 
                 conv_bridge=True, shortcut=True, skip_conn=True, residual_enc= False, residual_dec = True, 
                 causal_enc=False, causal_dec=True, conv_encoder='depthwise'):
        super(EDGeNet, self).__init__()
        #self.n_classes = n_classes
        upsample = list(reversed(downsample))
        self.encoder = EDGeNetEncoder(in_c, k_sz, layers, pool=pool, residual= residual_enc,
                                      causal=causal_enc, conv_encoder=conv_encoder, downsample=downsample)
        #self.latent = nn.Conv1d(widths[-1],latent_features,1)
        self.decoder = EDGeNetDecoder(n_classes, k_sz, layers, upsample, up_mode, conv_bridge, shortcut,skip_conn, 
                                      residual = residual_dec, causal = causal_dec)

    def forward(self, x):
        x, down_activations = self.encoder(x)
        #x= self.encoder(x)
        #print(x.size())
        x = self.decoder(x, down_activations)
        return x
# from ptflops import get_model_complexity_info
# from torchinfo import summary
# from torchstat import stat

# def test_model():
#     mod = EDGeNet(in_c=8, n_classes=8, layers=[8,8], downsample=[False], k_sz=5, up_mode='pixelshuffle', pool='conv', 
#                     conv_bridge=True, shortcut=True, skip_conn=True)
#     #mod = ConvNextForTSPrediction(in_channels=1, stem_features=4, depths=[2,2,6,2], widths=[4,8,12,16])
#     #mod = sumnet_auto()
#     x = torch.randn((1, 8, 64))
#     #mod.cuda()
#     pred = mod(x)
#     #print(stat(mod, input_size=(1, 1024)))
#     print(summary(mod, input_size=(1, 8, 64)))
#     #macs, params = get_model_complexity_info(mod, (1, 2048),
#     #                                         as_strings=True,
#     #                                         print_per_layer_stat=False)
#     #print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#     #print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
#     print(pred.shape)

# test_model()