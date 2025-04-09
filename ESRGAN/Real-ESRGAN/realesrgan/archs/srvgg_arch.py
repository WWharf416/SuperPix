import torch
from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn as nn
from torch.nn import functional as F
from torchvision.transforms.functional import equalize
try:
    import skimage.exposure
    import numpy as np
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


@ARCH_REGISTRY.register()
class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.

    It is a compact network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_conv (int): Number of convolution layers in the body network. Default: 16.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
        enhancement_type (str): Enhancement type applied after SR. Options: 'none', 'he', 'clahe'. Default: 'none'.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu', enhancement_type='none'):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        if enhancement_type not in ['none', 'he', 'clahe']:
            raise ValueError("enhancement_type must be one of 'none', 'he', 'clahe'")
        if enhancement_type == 'clahe' and not HAS_SKIMAGE:
            raise ImportError("Please install scikit-image to use CLAHE enhancement. `pip install scikit-image`")
        self.enhancement_type = enhancement_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            # activation
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        # Apply selected enhancement to the input if enabled
        if self.enhancement_type == 'he':
            # Histogram Equalization Preprocessing
            processed_channels = []
            for i in range(x.size(1)): # Iterate through channels (B C H W)
                channel = x[:, i:i+1, :, :].clamp(0, 1) # Clamp input to [0, 1]
                channel_uint8 = (channel * 255).to(torch.uint8)
                equalized_channel_uint8 = equalize(channel_uint8)
                equalized_channel_float = equalized_channel_uint8.float() / 255.0
                processed_channels.append(equalized_channel_float)
            net_input = torch.cat(processed_channels, dim=1)

        elif self.enhancement_type == 'clahe':
            # CLAHE Preprocessing
            processed_batch = []
            for b in range(x.size(0)): # Iterate through batch
                img_batch_item = x[b].clamp(0, 1)
                img_np = img_batch_item.permute(1, 2, 0).cpu().numpy()
                img_clahe_np = np.zeros_like(img_np)
                for i in range(img_np.shape[2]):
                    # Apply CLAHE channel-wise, ensure input is in [0, 1] for skimage
                    channel_data = img_np[:, :, i]
                    # Handle potential non-[0,1] float inputs for skimage
                    if np.issubdtype(channel_data.dtype, np.floating) and (channel_data.min() < 0 or channel_data.max() > 1):
                        channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-6) # Normalize to [0, 1]
                    img_clahe_np[:, :, i] = skimage.exposure.equalize_adapthist(channel_data, clip_limit=0.01)

                img_clahe_tensor = torch.from_numpy(img_clahe_np).permute(2, 0, 1).to(x.device)
                processed_batch.append(img_clahe_tensor)
            net_input = torch.stack(processed_batch, dim=0)

        else: # enhancement_type == 'none'
            net_input = x

        # Original network forward pass starts here, using potentially enhanced input
        out = net_input
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # Add the nearest upsampled ORIGINAL image for residual learning
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base

        return out
