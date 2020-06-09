import torch
import torch.nn as nn
from util import make_1ch, make_3ch
import torch.nn.functional as F

class Generator_UP(nn.Module):
    def __init__(self, channels=3, layers=8, features=64, scale_factor=2):
        super(Generator_UP, self).__init__()
        self.scale_factor = scale_factor
        
        model = [nn.Conv2d(channels, features, kernel_size=3, stride=1, padding=1),
                 nn.ReLU(True)]
        
        for i in range(1, layers - 1):
            model += [nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
                      nn.ReLU(True)]
        
        model += [nn.Conv2d(features, channels, kernel_size=3, stride=1, padding=1)]  
        
        self.model = nn.Sequential(*model)
        
        self.bilinear_kernel = torch.FloatTensor([[[[9/16, 3/16], [3/16, 1/16]]],
                                                  [[[3/16, 9/16], [1/16, 3/16]]],
                                                  [[[3/16, 1/16], [9/16, 3/16]]],
                                                  [[[1/16, 3/16], [3/16, 9/16]]]]).cuda()
    
    def bilinear_upsample(self, x):
        x = torch.cat([x[:,:,:1,:], x, x[:,:,-1:,:]], dim=2)
        x = torch.cat([x[:,:,:,:1], x, x[:,:,:,-1:]], dim=3)        
        x = make_1ch(x)
        x = F.conv2d(x, self.bilinear_kernel)
        x = F.pixel_shuffle(x, 2)
        x = make_3ch(x)
        x = x[..., 1:-1, 1:-1]
        return x
        
    def forward(self, x):
        x = self.bilinear_upsample(x)
        out = x + self.model(x)  # add skip connections
        return out

        
class Generator_DN(nn.Module):
    def __init__(self, features=64):
        super(Generator_DN, self).__init__()
        struct = [7, 5, 3, 1, 1, 1]
        self.G_kernel_size = 13
        # First layer
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=features, kernel_size=struct[0], stride=1, bias=False)

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            if struct[layer] == 3: # Downsample on the first layer with kernel_size=1
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], stride=2, bias=False)]
            else:
                feature_block += [nn.Conv2d(in_channels=features, out_channels=features, kernel_size=struct[layer], bias=False)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=struct[-1], bias=False)

    def forward(self, x):
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        x = make_1ch(x)
        x = self.first_layer(x)
        x = self.feature_block(x)
        out = self.final_layer(x)
        return make_3ch(out)


class Discriminator_DN(nn.Module):

    def __init__(self, layers=7, features=64, D_kernel_size=7):
        super(Discriminator_DN, self).__init__()

        # First layer - Convolution (with no ReLU)
        self.first_layer = nn.utils.spectral_norm(nn.Conv2d(in_channels=3, out_channels=features, kernel_size=D_kernel_size, bias=True))
        feature_block = []  # Stacking layers with 1x1 kernels
        for _ in range(1, layers - 1):
            feature_block += [nn.utils.spectral_norm(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=1, bias=True)),
                              nn.BatchNorm2d(features),
                              nn.ReLU(True)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(in_channels=features, out_channels=1, kernel_size=1, bias=True)),
                                         nn.Sigmoid())
        
        # Calculate number of pixels shaved in the forward pass
        self.forward_shave = 128 - self.forward(torch.FloatTensor(torch.ones([1, 3, 128, 128]))).shape[-1]
        
    def forward(self, x):
        x = self.first_layer(x)
        x = self.feature_block(x)
        out = self.final_layer(x)
        return out


def weights_init_D_DN(m):
    """ initialize weights of the discriminator """
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif class_name.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_init_G_DN(m):
    """ initialize weights of the generator """
    if m.__class__.__name__.find('Conv') != -1:
        n = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
        m.weight.data.normal_(1/n, 1/n)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)

def weights_init_G_UP(m):
    """ initialize weights of the generator """
    if m.__class__.__name__.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)