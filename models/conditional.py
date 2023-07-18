import torch
import torch.nn as nn
import torch.nn.functional as F

from .didn import DIDN

class ConditionalInstanceNorm(nn.Module):
    def __init__(self, in_channel, latent_dim=64):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)

        self.style = nn.Linear(latent_dim, in_channel * 2)

        self.style.bias.data[:in_channel] = 0
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, latent_code):
        # style [batch_size, in_channels*2] => [batch_size, in_channels*2, 1, 1, 1]
        style = self.style(latent_code).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(0)
        # print(style.shape, len(style))      
        gamma, beta = style.chunk(2, dim=1)

        out = self.norm(input)
        # out = input
        out = (1. + gamma) * out + beta

        return out
    
    
class PreActBlock_Conditional(nn.Module):
    expansion = 1
    def __init__(
        self,
        in_channel,
        num_chans,
        stride,
        bias=False,
        latent_dim=64,
        mapping_fmaps=64,
        ):
        super().__init__()
        
        self.adain1 = ConditionalInstanceNorm(in_channel, latent_dim)
        self.conv1 = nn.Conv2d(in_channel, num_chans, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.adain2 = ConditionalInstanceNorm(in_channel, latent_dim)
        self.conv2 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        
        self.mapping = nn.Sequential(
            nn.Linear(1, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, latent_dim),
            nn.LeakyReLU(0.2)
        )

        if stride != 1 or in_channel != self.expansion*num_chans:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, self.expansion*num_chans, kernel_size=1, stride=stride, bias=bias)
            )

    def forward(self, x, lambda_):
        # print(x.shape, x.dtype, lambda_.shape, lambda_.dim())

        latent_fea = self.mapping(lambda_)
        out = F.leaky_relu(self.adain1(x, latent_fea), negative_slope=0.2)

        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)

        out = self.conv2(F.leaky_relu(self.adain2(out, latent_fea), negative_slope=0.2))

        out += shortcut
        return out
    

class ConditionalResidualBlock(nn.Module):
    def __init__(
        self,
        num_chans=64):
        super().__init__()
        
        bias = True
        # res 1
        self.conv1 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu4 = nn.PReLU()
        self.cond1 = PreActBlock_Conditional(num_chans, num_chans, 1, bias=bias)
        
        # concat 1
        self.conv5 = nn.Conv2d(num_chans, num_chans * 2, kernel_size=3, stride=2, padding=1, bias=bias)
        self.relu6 = nn.PReLU()
        
        # res 2
        self.conv7 = nn.Conv2d(num_chans * 2, num_chans * 2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu8 = nn.PReLU()
        self.cond2 = PreActBlock_Conditional(num_chans * 2, num_chans * 2, 1, bias=bias)
        
        # concat 2
        self.conv9 = nn.Conv2d(num_chans * 2, num_chans * 4, kernel_size=3, stride=2, padding=1, bias=bias)
        self.relu10 = nn.PReLU()
        
        # res 3
        self.conv11 = nn.Conv2d(num_chans * 4, num_chans * 4, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu12 = nn.PReLU()
        self.cond3 = PreActBlock_Conditional(num_chans * 4, num_chans * 4, 1, bias=bias)

        self.conv13 = nn.Conv2d(num_chans * 4, num_chans * 8, kernel_size=1, stride=1, padding=0, bias=bias)
        self.up14 = nn.PixelShuffle(2)
        
        # concat 2 
        self.conv15 = nn.Conv2d(num_chans * 4, num_chans * 2, kernel_size=1, stride=1, padding=0, bias=bias)
        # res 4
        self.conv16 = nn.Conv2d(num_chans * 2, num_chans * 2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu17 = nn.PReLU()
        self.cond4 = PreActBlock_Conditional(num_chans * 2, num_chans * 2, 1, bias=bias)
        
        # res 4
        self.conv18 = nn.Conv2d(num_chans * 2, num_chans * 4, kernel_size=1, stride=1, padding=0, bias=bias)
        self.up19 = nn.PixelShuffle(2)
        
        # concat 1
        self.conv20 = nn.Conv2d(num_chans * 2, num_chans, kernel_size=1, stride=1, padding=0, bias=bias)
        
        # res 5
        self.conv21 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu22 = nn.PReLU()
        self.conv23 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu24 = nn.PReLU()
        self.cond5 = PreActBlock_Conditional(num_chans, num_chans, 1, bias=bias)
        
        self.conv25 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        
    def forward(self, x, lambda_):
        res1 = x
        out = self.cond1(self.relu4(self.conv3(self.relu2(self.conv1(x)))), lambda_)
        out = torch.add(out, res1)
        cat1 = out
        
        out = self.relu6(self.conv5(out))
        res2 = out
        out = self.cond2(self.relu8(self.conv7(out)), lambda_)
        out = torch.add(out, res2)
        cat2 = out
        
        out = self.relu10(self.conv9(out))
        res3 = out
        
        out = self.cond3(self.relu12(self.conv11(out)), lambda_)
        out = torch.add(out, res3)
        
        out = self.up14(self.conv13(out))
        
        out = torch.cat([out, cat2], 1)
        out = self.conv15(out)
        res4 = out
        out = self.cond4(self.relu17(self.conv16(out)), lambda_)
        out = torch.add(out, res4)
        
        out = self.up19(self.conv18(out))
        
        out = torch.cat([out, cat1], 1)
        out = self.conv20(out)
        res5 = out
        out = self.cond5(self.relu24(self.conv23(self.relu22(self.conv21(out)))), lambda_)
        out = torch.add(out, res5)
        
        out = self.conv25(out)
        out = torch.add(out, res1)
        
        return out
    

class Cond_DIDN(DIDN):
    def __init__(
        self,
        in_chans,
        out_chans,
        num_chans=64,
        pad_data=True,
        global_residual=True,
        n_res_blocks=6,
    ):
        super().__init__(in_chans, out_chans, num_chans, pad_data, global_residual, n_res_blocks)    
        self.pad_data = pad_data
        self.global_residual = global_residual
        bias = True
        self.conv_input = nn.Conv2d(in_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu1 = nn.PReLU()
        self.conv_down = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=2, padding=1, bias=bias)
        self.relu2 = nn.PReLU()
        
        self.n_res_blocks = n_res_blocks
        recursive = []
        
        for i in range(self.n_res_blocks):
            recursive.append(ConditionalResidualBlock(num_chans=num_chans))
        
        self.recursive = torch.nn.ModuleList(recursive)
        
        self.conv_mid = nn.Conv2d(num_chans * self.n_res_blocks, num_chans, kernel_size=1, stride=1, padding=0, bias=bias)
        self.relu3 = nn.PReLU()
        self.conv_mid2 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu4 = nn.PReLU()
        
        self.subpixel = nn.PixelShuffle(2)
        self.conv_output = nn.Conv2d(num_chans // 4, out_chans, kernel_size=3, stride=1, padding=1, bias=bias)
        
    def forward(self, x, lambda_):
        if self.pad_data:
            orig_shape2d = x.shape[-2:]
            p2d = self.calculate_downsampling_padding2d(x, 3)
            x = self.pad2d(x, p2d)

        residual = x
        out = self.relu1(self.conv_input(x))
        out = self.relu2(self.conv_down(out))

        recons = []
        for i in range(self.n_res_blocks):
            out = self.recursive[i](out, lambda_)
            recons.append(out)

        out = torch.cat(recons, 1)

        out = self.relu3(self.conv_mid(out))
        residual2 = out
        out = self.relu4(self.conv_mid2(out))
        out = torch.add(out, residual2)

        out = self.subpixel(out)
        out = self.conv_output(out)

        if self.global_residual:
            out = torch.add(out, residual)

        if self.pad_data:
            out = self.unpad2d(out, orig_shape2d)

        return out
    
    
# if __name__ == '__main__':
#     a = PreActBlock_Conditional(64, 64, 1).cuda()
#     input = torch.randn(24, 64, 64, 64).cuda()
#     lambda_ = torch.tensor([0.5]).float().cuda()
#     out = a(input, lambda_)
    
    