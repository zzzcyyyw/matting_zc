import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from config import device

from collections import OrderedDict

""" ASPP模块 """


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, 1, 0, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x.type(torch.float32)))
        res = torch.cat(res, dim=1)
        return self.project(res)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class ImageEncoder(nn.Module):
    def __init__(self, layers, width=64):      # ori_res_encoder
        super(ImageEncoder, self).__init__()
        # 加载预训练resnet encoder
        # self.conv1 = ori_res_encoder.visual.conv1
        # self.bn1 = ori_res_encoder.visual.bn1
        # self.relu1 = ori_res_encoder.visual.relu1
        #
        # self.conv2 = ori_res_encoder.visual.conv2
        # self.bn2 = ori_res_encoder.visual.bn2
        # self.relu2 = ori_res_encoder.visual.relu2
        #
        # self.conv3 = ori_res_encoder.visual.conv3
        # self.bn3 = ori_res_encoder.visual.bn3
        # self.relu3 = ori_res_encoder.visual.relu3
        #
        # self.avgpool = ori_res_encoder.visual.avgpool
        # self.layer1 = ori_res_encoder.visual.layer1
        # self.layer2 = ori_res_encoder.visual.layer2
        # self.layer3 = ori_res_encoder.visual.layer3
        # self.layer4 = ori_res_encoder.visual.layer4
        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x_rgb):
        conv_out = [x_rgb]

        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x_rgb.type(self.conv1.weight.dtype)
        x = stem(x)
        conv_out.append(x)
        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        return conv_out


class TextEncoder(nn.Module):
    def __init__(self, ori_vit_encoder):
        super(TextEncoder, self).__init__()
        self.token_embedding = ori_vit_encoder.token_embedding
        self.positional_embedding = ori_vit_encoder.positional_embedding
        self.text_transformer = ori_vit_encoder.transformer
        self.ln_final = ori_vit_encoder.ln_final

    def forward(self, prompts):
        x_t = self.token_embedding(prompts)
        x_t = x_t + self.positional_embedding
        x_t = x_t.permute(1, 0, 2).to(torch.float16)
        x_t = self.text_transformer(x_t)
        x_t = x_t.permute(1, 0, 2)
        x_t = self.ln_final(x_t)

        return x_t

"""CLIP的ViT图像编码器部分"""
class DepthVision(nn.Module):
    def __init__(self, ori_vit_encoder, input_resolution, patch_size, width=768):
        super(DepthVision, self).__init__()
        self.depth_conv = nn.Conv2d(1, 3, kernel_size=1)

        self.conv1 = ori_vit_encoder.visual.conv1

        # scale = width ** -0.5
        # self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.class_embedding = ori_vit_encoder.visual.class_embedding
        # self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.positional_embedding = ori_vit_encoder.visual.positional_embedding
        self.ln_pre = ori_vit_encoder.visual.ln_pre
        self.depth_transformer = ori_vit_encoder.visual.transformer

    def forward(self, depth_map):
        depth_map_3 = self.depth_conv(depth_map)

        x_d = self.conv1(depth_map_3.to(torch.float16))
        x_d = x_d.reshape(x_d.shape[0], x_d.shape[1], -1)
        x_d = x_d.permute(0, 2, 1)
        x_d = torch.cat([self.class_embedding.to(x_d.dtype) + torch.zeros(x_d.shape[0], 1, x_d.shape[-1], dtype=x_d.dtype, device=x_d.device), x_d], dim=1)
        x_d = x_d + self.positional_embedding.to(x_d.dtype)
        x_d = self.ln_pre(x_d)

        x_d = x_d.permute(1, 0, 2)
        x_d = self.depth_transformer(x_d)
        x_d = x_d.permute(1, 0, 2)

        return x_d


"""深度图适配器(depth_map to text_embedding)"""
class DepthAdapter(nn.Module):
    def __init__(self):
        super(DepthAdapter, self).__init__()
        self.target_size = 101

        self.conv1d = nn.Conv1d(in_channels=1024, out_channels=768, kernel_size=1)
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        # self.fc = None
        self.fc = nn.Sequential(
            nn.Linear(257, 77),     # 224➡257, 320➡485
            nn.ReLU()
        )

    def forward(self, depth_embed):
        # batch_size, seq_len, embed_dim = depth_embed.size()
        depth_embed = depth_embed.to(torch.float32)
        depth_embed = self.conv1d(depth_embed.permute(0, 2, 1)).permute(0, 2, 1)

        depth_embed = depth_embed.permute(1, 0, 2)
        attn_out, _ = self.attention(depth_embed, depth_embed, depth_embed)
        attn_out = attn_out.permute(1, 0, 2)

        attn_out = attn_out.permute(0, 2, 1)
        text_embed = self.fc(attn_out)
        text_embed = text_embed.permute(0, 2, 1)

        return text_embed


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1d = nn.Conv1d(768, 512, kernel_size=1)       # 512 2048

        self.batch_norm = True
        middle_chn = 2048

        self.global_module = ASPP(middle_chn, atrous_rates=[6, 12, 18])
        en_chn = middle_chn + 512 + 256  # aspp_channel + depth_features_channel

        self.deconv6_1 = nn.Conv2d(en_chn, 512, kernel_size=1, bias=True)  # 512
        self.deconv5_1 = nn.Conv2d(1536, 512, kernel_size=5, padding=2, bias=True)  # 512
        self.deconv4_1 = nn.Conv2d(1024, 256, kernel_size=5, padding=2, bias=True)  # 512
        self.deconv3_1 = nn.Conv2d(512, 128, kernel_size=5, padding=2, bias=True)  # 256
        self.deconv2_1 = nn.Conv2d(128, 64, kernel_size=5, padding=2, bias=True)
        self.deconv1_1 = nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=True)

        self.deconv1 = nn.Conv2d(64, 1, kernel_size=5, padding=2, bias=True)

    def forward(self, conv_out, text_embed):
        conv5 = conv_out[-1]

        text_embed = text_embed.permute(0, 2, 1)
        text_embed = self.conv1d(text_embed)
        text_embed = nn.functional.interpolate(text_embed, size=(conv5.size(2) * conv5.size(3)))
        text_embed = text_embed.view(text_embed.size(0), 512, conv5.size(2), conv5.size(3))        # 512 2048

        # combined = conv5 + text_embed

        global_ctx = self.global_module(conv5)
        x = torch.cat([conv5, global_ctx, text_embed], 1)
        # x = torch.cat([combined, global_ctx], 1)
        x61d = F.relu(self.deconv6_1(x))

        x5d = F.interpolate(x61d, scale_factor=2, mode='bilinear', align_corners=False)
        x5d = torch.cat([x5d, conv_out[-2]], 1)
        x51d = F.relu(self.deconv5_1(x5d))

        x4d = F.interpolate(x51d, scale_factor=2, mode='bilinear', align_corners=False)
        x4d = torch.cat([x4d, conv_out[-3]], 1)
        x41d = F.relu(self.deconv4_1(x4d))

        x3d = F.interpolate(x41d, scale_factor=2, mode='bilinear', align_corners=False)
        x3d = torch.cat([x3d, conv_out[-4]], 1)
        x31d = F.relu(self.deconv3_1(x3d))

        x2d = F.interpolate(x31d, scale_factor=2, mode='bilinear', align_corners=False)
        # x2d = torch.cat([x2d, conv_out[-5]], 1)
        x21d = F.relu(self.deconv2_1(x2d))

        x1d = F.interpolate(x21d, scale_factor=2, mode='bilinear', align_corners=False)
        # x1d = torch.cat([x1d, conv_out[-6]], 1)
        x11d = F.relu(self.deconv1_1(x1d))

        raw_alpha = self.deconv1(x11d)
        pred_mattes = torch.sigmoid(raw_alpha)

        return pred_mattes


class ClipMattingModule(nn.Module):
    def __init__(self, img_enc, depth_vision, depth_adap, net_dec, text_enc, args):
        super(ClipMattingModule, self).__init__()
        # self.ori_res_encoder = ori_res_encoder
        self.stage = args.stage

        self.depth_vision = depth_vision
        self.depth_adapter = depth_adap
        self.text_encoder = text_enc

        # stage1：冻结clip的视觉编码器和文本编码器，只训练depth_adapter
        for param in self.depth_vision.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        # 只解冻self.depth_conv
        for param in self.depth_vision.depth_conv.parameters():
            param.requires_grad = True

        # stage2：冻结图像编码器和stage1训练好的depth_adapter，只训练Decoder
        if self.stage == 2:
            self.img_encoder = img_enc
            self.decoder = net_dec
            # for param in self.img_encoder.parameters():
            #     param.requires_grad = False
            for param in self.depth_adapter.parameters():
                param.requires_grad = False
            # 冻结stage1中DepthVision里训练的的self.depth_conv
            for param in self.depth_vision.depth_conv.parameters():
                param.requires_grad = False

    @property
    def dtype(self):
        return self.ori_res_encoder.visual.conv1.weight.dtype

    def forward(self, img, depth_map, prompts):
        depth_embed = self.depth_vision(depth_map)
        text_embed = self.depth_adapter(depth_embed)
        # prompts_embed = self.text_encoder(prompts)

        # 第一阶段训练depth_adapter
        if self.stage == 1:
            prompts_embed = self.text_encoder(prompts)
            return text_embed, prompts_embed

        conv_out = self.img_encoder(img)
        alpha_out = self.decoder(conv_out, text_embed)

        return alpha_out


def model_builder(weight_init, args):
    ori_res_encoder, _ = clip.load('RN50', device=device)
    ori_vit_encoder, _ = clip.load('ViT-L/14', device=device)
    text_encoder = TextEncoder(ori_vit_encoder)

    image_encoder = ImageEncoder((3, 4, 6, 3), 64)
    image_encoder.load_state_dict(ori_res_encoder.visual.state_dict(), strict=False)

    depth_vision = DepthVision(ori_vit_encoder, input_resolution=224, patch_size=14, width=1024)
    depth_adapter = DepthAdapter()
    net_decoder = Decoder()
    net_decoder.apply(weight_init)
    model = ClipMattingModule(image_encoder, depth_vision, depth_adapter, net_decoder, text_encoder, args)

    if args.stage == 2:
        depth_adapter_ckpt = torch.load('saved_model/adapter/train_adapter_221.pth')
        model.load_state_dict(depth_adapter_ckpt, strict=False)

    return model


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


if __name__ == "__main__":
    import numpy as np

    torch.manual_seed(7)
    np.random.seed(7)
    ori_vit_encoder, _ = clip.load('ViT-L/14', device=device)
    # model = DepthEncoder(ori_vit_encoder, input_resolution=224, patch_size=32, width=768)
    # image = torch.randn(32, 3, 224, 224).to(device)
    # depth = torch.randn(32, 1, 320, 320).to(device)
    # model.to(device)
    # model.train()
    # out = model(image.to(torch.float16))
    #
    # net = DepthAdapter()
    # net.to(device)
    # net.train()
    # text_out = net(out.to(torch.float32))
    h = TextEncoder(ori_vit_encoder)

    image = torch.randn(32, 3, 224, 224).to(device)
    depth = torch.randn(32, 1, 224, 224).to(device)

    model = model_builder(weight_init)
    model.to(device)
    model.train()

    out = model(image, depth)
    out = out.view(out.size(0), -1, 32, 32)

    print(out)
