# full assembly of the sub-parts to form the complete net
import torch.nn.functional as F

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        # return F.sigmoid(x)
        return x

class UNet_dsc(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_dsc, self).__init__()
        self.inc = inconv_dsc(n_channels, 64)
        self.down1 = down_dsc(64, 128)
        self.down2 = down_dsc(128, 256)
        self.down3 = down_dsc(256, 512)
        self.down4 = down_dsc(512, 512)
        self.up1 = up_dsc(1024, 256)
        self.up2 = up_dsc(512, 128)
        self.up3 = up_dsc(256, 64)
        self.up4 = up_dsc(128, 64)
        self.outc = outconv_dsc(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
        # return F.sigmoid(x)

class SE_UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SE_UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.se_block1 = SE_block(64, 16)
        self.down1 = down(64, 128)
        self.se_block2 = SE_block(128, 16)
        self.down2 = down(128, 256)
        self.se_block3 = SE_block(256, 16)
        self.down3 = down(256, 512)
        self.se_block4 = SE_block(512, 16)
        self.down4 = down(512, 512)
        self.se_block5 = SE_block(512, 16)

        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.se_block1(x1)

        x2 = self.down1(x1)
        x2 = self.se_block2(x2)
        
        x3 = self.down2(x2)
        x3 = self.se_block3(x3)

        x4 = self.down3(x3)
        x4 = self.se_block4(x4)

        x5 = self.down4(x4)
        x5 = self.se_block5(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


# if __name__ == '__mian__':
#     model = SE_UNet(8, 2)
#     dummy_input = torch.randn(1,8, 480, 480)
#     out = model(dummy_input)
#     print(out.shape)