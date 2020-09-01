# full assembly of the sub-parts to form the complete net


from .unet_parts import *
torch.set_default_tensor_type('torch.FloatTensor')

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
        return F.softmax(x, dim=1)
        # return x

class NestedUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(NestedUNet, self).__init__()


        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = up_fit()

        self.conv0_0 = VGGBlock(n_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        # if self.args.deepsupervision:
        #     self.final1 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
        #     self.final2 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
        #     self.final3 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
        #     self.final4 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
        # else:
        #     self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
        self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(self.up(x1_0, x0_0))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(self.up(x2_0, x1_0))
        x0_2 = self.conv0_2(torch.cat([x0_0, self.up(x1_1, x0_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(self.up(x3_0, x2_0))
        x1_2 = self.conv1_2(torch.cat([x1_0, self.up(x2_1, x1_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, self.up(x1_2, x0_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(self.up(x4_0, x3_0))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1, x2_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, self.up(x2_2, x1_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, self.up(x1_3, x0_3)], 1))

        # if self.args.deepsupervision:
        #     output1 = self.final1(x0_1)
        #     output2 = self.final2(x0_2)
        #     output3 = self.final3(x0_3)
        #     output4 = self.final4(x0_4)
        #     return [output1, output2, output3, output4]

        # else:
        #     output = self.final(x0_4)
        #     return output


        output = self.final(x0_4)
        # return F.sigmoid(output)
        return F.softmax(output, dim=1)

class SCSENestedUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SCSENestedUNet, self).__init__()


        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = up_fit()

        self.conv0_0 = SCSEVGGBlock(n_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = SCSEVGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = SCSEVGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = SCSEVGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = SCSEVGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = SCSEVGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = SCSEVGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = SCSEVGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = SCSEVGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = SCSEVGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = SCSEVGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = SCSEVGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = SCSEVGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = SCSEVGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = SCSEVGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        # if self.args.deepsupervision:
        #     self.final1 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
        #     self.final2 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
        #     self.final3 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
        #     self.final4 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
        # else:
        #     self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
        self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(self.up(x1_0, x0_0))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(self.up(x2_0, x1_0))
        x0_2 = self.conv0_2(torch.cat([x0_0, self.up(x1_1, x0_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(self.up(x3_0, x2_0))
        x1_2 = self.conv1_2(torch.cat([x1_0, self.up(x2_1, x1_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, self.up(x1_2, x0_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(self.up(x4_0, x3_0))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1, x2_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, self.up(x2_2, x1_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, self.up(x1_3, x0_3)], 1))

        # if self.args.deepsupervision:
        #     output1 = self.final1(x0_1)
        #     output2 = self.final2(x0_2)
        #     output3 = self.final3(x0_3)
        #     output4 = self.final4(x0_4)
        #     return [output1, output2, output3, output4]

        # else:
        #     output = self.final(x0_4)
        #     return output


        output = self.final(x0_4)
        # return F.sigmoid(output)
        return F.softmax(output, dim=1)


class SCSE_UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SCSE_UNet, self).__init__()
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
        self.SCSE1 = SCSE(64)
        self.SCSE2 = SCSE(128)
        self.SCSE3 = SCSE(256)
        self.SCSE4 = SCSE(512)
        self.SCSE8 = SCSE(64)
        self.SCSE7 = SCSE(64)
        self.SCSE6 = SCSE(128)
        self.SCSE5 = SCSE(256)
        # self.SCSE4 = SCSE(512)
        # self.SCSE4 = SCSE(32)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.SCSE1(x1)
        x2 = self.down1(x1)
        x2 = self.SCSE2(x2)
        x3 = self.down2(x2)
        x3 = self.SCSE3(x3)
        x4 = self.down3(x3)
        x4 = self.SCSE4(x4)
        x5 = self.down4(x4)
        # x5 = self.SCSE4(x5)
        x = self.up1(x5, x4)
        x = self.SCSE5(x)
        x = self.up2(x, x3)
        x = self.SCSE6(x)
        x = self.up3(x, x2)
        x = self.SCSE7(x)
        x = self.up4(x, x1)
        x = self.SCSE8(x)
        x = self.outc(x)
        # x = self.SCSE1(x)
        # return F.sigmoid(x)
        return F.softmax(x, dim=1)
        # return x

class ConvActivation(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1,
                 activation=nn.ReLU(inplace=True)):
        super(ConvActivation, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class ConvBNActivation(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1,
                 activation=nn.ReLU(inplace=True)):
        super(ConvBNActivation, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1,
                 batch_norm=False, activation=nn.ReLU(inplace=True)):
        super(ConvBlock, self).__init__()
        conv = ConvBNActivation if batch_norm else ConvActivation
        self.block = nn.Sequential(
            conv(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, activation),
            conv(out_channels, out_channels, kernel_size,
                 stride, padding, dilation, activation)
        )

    def forward(self, x):
        out = self.block(x)
        return out


class UpBlockWithSkip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, up_mode='deconv',
                 batch_norm=False, activation=nn.ReLU(inplace=True)):
        assert up_mode in ('deconv', 'biupconv', 'nnupconv')
        super(UpBlockWithSkip, self).__init__()

        if up_mode == 'deconv':
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1)
        elif up_mode == 'biupconv':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2,
                            align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        elif up_mode == 'nnupconv':
            self.up = nn.Sequential(
                nn.Upsample(mode='nearest', scale_factor=2,
                            align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )

        self.conv_block = ConvBlock(
            out_channels * 2, out_channels, kernel_size,
            stride, padding, dilation, batch_norm, activation)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        out = torch.cat([x1, x2], 1)
        out = self.conv_block(out)

        return out


class DilatedUNet(nn.Module):

    def __init__(self, in_channels=3, classes=1, depth=3,
                 first_channels=44, padding=1,
                 bottleneck_depth=6, bottleneck_type='cascade',
                 batch_norm=True, up_mode='deconv',
                 activation=nn.ReLU(inplace=True)):

        assert bottleneck_type in ('cascade', 'parallel')
        super(DilatedUNet, self).__init__()

        self.depth = depth
        self.bottleneck_type = bottleneck_type

        conv = ConvBNActivation if batch_norm else ConvActivation

        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                ConvBlock(prev_channels, first_channels * 2**i, 3,
                          padding=padding, batch_norm=batch_norm,
                          activation=activation))
            prev_channels = first_channels * 2**i

        self.bottleneck_path = nn.ModuleList()
        for i in range(bottleneck_depth):
            bneck_in = prev_channels if i == 0 else prev_channels * 2
            self.bottleneck_path.append(
                conv(bneck_in, prev_channels * 2, 3,
                     dilation=2**i, padding=2**i, activation=activation))

        prev_channels *= 2

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up_path.append(
                UpBlockWithSkip(prev_channels, first_channels * 2**i, 3,
                                up_mode=up_mode, padding=padding,
                                batch_norm=batch_norm,
                                activation=activation))
            prev_channels = first_channels * 2**i

        self.last = nn.Conv2d(prev_channels, classes, kernel_size=1)

    def forward(self, x):
        bridges = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            bridges.append(x)
            x = F.avg_pool2d(x, 2)

        dilated_layers = []
        for i, bneck in enumerate(self.bottleneck_path):
            if self.bottleneck_type == 'cascade':
                x = bneck(x)
                dilated_layers.append(x.unsqueeze(-1))
            elif self.bottleneck_type == 'parallel':
                dilated_layers.append(bneck(x.unsqueeze(-1)))
        x = torch.cat(dilated_layers, dim=-1)
        x = torch.sum(x, dim=-1)

        for i, up in enumerate(self.up_path):
            x = up(x, bridges[-i-1])
        x = self.last(x)
        return F.softmax(x, dim=1)