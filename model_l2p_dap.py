import copy

import torch
import torch.nn as nn
from thop import profile
from timm import create_model
from torch.nn import init

from l2prompt import Prompt
from resnet import resnet50, resnet18, resnet34
from utils import initial_classifier#, print_model_structure
from IPython import embed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from vision_transformer import ViT
from sinet import SiNet

'''
    SIM_LAMBDA: 2.0
    NUM_DAP_TOKENS: 30
    TASK_EMB: 64
    NUM_TASKS_FOR_EMB: 10
    PROMPT_POOL: 10
    
    CURRENT_LAMBDA: 1.0
    SIM_LAMBDA: 0.1
    NUM_DAP_TOKENS: 10
    TASK_EMB: 16
'''
class ADPT_Block(nn.Module):
    def __init__(self):
        super(ADPT_Block, self).__init__()
        self.norm = nn.LayerNorm(768, eps=1e-6)

        self.NUM_DAP_TOKENS = 20
        self.hidden_size = 768
        self.TASK_EMB = 16
        self.prompt_length =4
        self.top_k = 5

        # domain-adaptive prompts
        # self.dap_config = dap_config
        self.dap_downsample = nn.Linear(251, self.NUM_DAP_TOKENS)
        nn.init.zeros_(self.dap_downsample.weight)
        nn.init.zeros_(self.dap_downsample.bias)
        self.dap_film = nn.Linear(self.TASK_EMB, self.hidden_size * 2)  # * 2
        self.dap_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)

    def forward(self, x, task_id_estimated_emb=None, isFirst = True):
        if isFirst: # first layer
            x_norm = self.dap_norm(x)
            x_tran = torch.transpose(x_norm, 2, 1)
            down = self.dap_downsample(x_tran)

            film = self.dap_film(task_id_estimated_emb)

            film = film[:, 0, :]

            gamma4 = film[:, :self.hidden_size]
            beta4 = film[:, self.hidden_size:]
            gamma_norm = gamma4.norm(p=2, dim=1, keepdim=True).detach()
            beta_norm = beta4.norm(p=2, dim=1, keepdim=True).detach()
            gamma4 = gamma4.div(gamma_norm).view(film.size(0), -1, 1)
            beta4 = beta4.div(beta_norm).view(film.size(0), -1, 1)
            down = gamma4 * down + beta4
            down = torch.transpose(down, 2, 1)

            x = torch.cat((
                x[:, :1, :],
                down,
                x[:, 1:, :]
            ), dim=1)
        else:
            x = torch.cat((
                x[:, :1, :],
                x[:, (1+self.NUM_DAP_TOKENS):, :] # self.NUM_DAP_TOKENS
            ), dim=1)

            x_norm = self.dap_norm(x)
            x_tran = torch.transpose(x_norm, 2, 1)
            down = self.dap_downsample(x_tran)

            film = self.dap_film(task_id_estimated_emb)

            film = film[:, 0, :]

            gamma4 = film[:, :self.hidden_size]
            beta4 = film[:, self.hidden_size:]
            gamma_norm = gamma4.norm(p=2, dim=1, keepdim=True).detach()
            beta_norm = beta4.norm(p=2, dim=1, keepdim=True).detach()

            gamma4 = gamma4.div(gamma_norm).view(film.size(0), -1, 1)
            beta4 = beta4.div(beta_norm).view(film.size(0), -1, 1)
            down = gamma4 * down + beta4
            down = torch.transpose(down, 2, 1)

            x = torch.cat((
                x[:, :1, :],
                down,
                x[:, 1:, :]
            ), dim=1)

        return x  #, weights


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                      padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = 18 * 9
        self.embedding = nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1)

    def forward(self, x):
        # [B,C,H,W] -> [B,embed_dim,H,W] ->[B,embed_dim,H*W] -> [B,H*W,embed_dim]
        x = self.embedding(x).flatten(2).transpose(1, 2)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [128, 211, 768]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpecificAttention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., fc=nn.Linear):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = fc(dim, dim, bias=qkv_bias)
        self.k = fc(dim, dim, bias=qkv_bias)
        self.v = fc(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = fc(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, token_nums, mask_heads=None, **kwargs):
        B, N, C = x.shape
        q = self.q(x[:, :token_nums]).reshape(B, token_nums, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if mask_heads is not None:
            mask_heads = mask_heads.expand(B, self.num_heads, -1, N)
            attn = attn * mask_heads

        specific = (attn @ v).transpose(1, 2).reshape(B, token_nums, C)
        specific = self.proj(specific)
        specific = self.proj_drop(specific)

        return specific  # , attn, v


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type=SpecificAttention):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attention_type(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, token_nums=2):
        if isinstance(self.attn, SpecificAttention):
            specific_tokens = x[:, :token_nums]
            x = specific_tokens + self.drop_path(self.attn(self.norm1(x), token_nums))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def print_model_structure(model):
    """
    打印模型的结构信息，包括模块名称和类型。

    Args:
        model (nn.Module): 要打印结构的模型实例。
    """
    print("Model Structure:")
    _print_module_structure(model, "")


def _print_module_structure(module, prefix):
    """
    递归地打印模块及其子模块的结构信息。

    Args:
        module (nn.Module): 要打印结构的模块实例。
        prefix (str): 当前模块在模型层次结构中的前缀，用于缩进显示。
    """
    print(prefix + f"{module.__class__.__name__}")
    for name, child in module.named_children():
        if isinstance(child, nn.Module):
            _print_module_structure(child, prefix + "    ")
        else:
            print(prefix + "    " + f"{name}: {child.__class__.__name__}")


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        if arch == 'resnet50':
            model_v = resnet50(pretrained=True,
                               last_conv_stride=1, last_conv_dilation=1)
        elif arch == 'resnet34':
            model_v = resnet34(pretrained=True,
                               last_conv_stride=1, last_conv_dilation=1)
        elif arch == 'resnet18':
            model_v = resnet18(pretrained=True,
                               last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        # x = self.visible.layer1(x)
        # x = self.visible.layer2(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        if arch == 'resnet50':
            model_t = resnet50(pretrained=True,
                               last_conv_stride=1, last_conv_dilation=1)
        elif arch == 'resnet34':
            model_t = resnet34(pretrained=True,
                               last_conv_stride=1, last_conv_dilation=1)
        elif arch == 'resnet18':
            model_t = resnet18(pretrained=True,
                               last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        # x = self.thermal.layer1(x)
        # x = self.thermal.layer2(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        if arch == 'resnet50':
            model_base = resnet50(pretrained=True,
                                  last_conv_stride=1, last_conv_dilation=1)
        elif arch == 'resnet34':
            model_base = resnet34(pretrained=True,
                                  last_conv_stride=1, last_conv_dilation=1)
        elif arch == 'resnet18':
            model_base = resnet18(pretrained=True,
                                  last_conv_stride=1, last_conv_dilation=1)

        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class embed_net(nn.Module):
    def __init__(self, class_num, arch='resnet50', non_local="on"):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        self.non_local = non_local
        # self.gm_pool = gm_pool

        if self.non_local == 'on':
            non_layers = [0, 2, 3, 0]
            if arch == "resnet50":
                channels = [256, 512, 1024, 2048]
                layers = [3, 4, 6, 3]
            elif arch == 'resnet34':
                channels = [64, 128, 256, 512]
                layers = [3, 4, 6, 3]
            elif arch == 'resnet18':
                channels = [64, 128, 256, 512]
                layers = [2, 2, 2, 2]
            self.NL_1 = nn.ModuleList(
                [Non_local(channels[0]) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList(
                [Non_local(channels[1]) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [Non_local(channels[2]) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [Non_local(channels[3]) for i in range(non_layers[3])])
            self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

        pool_dim = 2048
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.l2norm = Normalize(2)

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        # shared block
        if self.non_local == 'on':
            NL1_counter = 0
            if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
            for i in range(len(self.base_resnet.base.layer1)):
                x = self.base_resnet.base.layer1[i](x)
                if i == self.NL_1_idx[NL1_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_1[NL1_counter](x)
                    NL1_counter += 1
            # Layer 2
            NL2_counter = 0
            if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
            for i in range(len(self.base_resnet.base.layer2)):
                x = self.base_resnet.base.layer2[i](x)
                if i == self.NL_2_idx[NL2_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_2[NL2_counter](x)
                    NL2_counter += 1
            # Layer 3
            NL3_counter = 0
            if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
            for i in range(len(self.base_resnet.base.layer3)):
                x = self.base_resnet.base.layer3[i](x)
                if i == self.NL_3_idx[NL3_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_3[NL3_counter](x)
                    NL3_counter += 1
            # Layer 4
            NL4_counter = 0
            if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
            for i in range(len(self.base_resnet.base.layer4)):
                x = self.base_resnet.base.layer4[i](x)
                if i == self.NL_4_idx[NL4_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_4[NL4_counter](x)
                    NL4_counter += 1
        else:
            x = self.base_resnet(x)
        return x


class MyModel(nn.Module):
    def __init__(self,
                 class_num, embed_dim=768,
                 local_up_layers=10,
                 num_of_tabs=1,

                 # '补充的prompt参数'
                 prompt_length=5,  # 提示的长度
                 pool_size = 20,
                 top_k = 4,

                 Separation=False,
                 withPrompt=False,
                 withAdpt = False,
                 FreezeBackbone=False,

                 idOldModel=False,

                 isShareBlock=False,
                 KeyToken=False,
                 instance_wise=False,
                 base=False,
                 use_backbone=False,
                 GeneralToken=False):
        super().__init__()
        self.class_num = class_num
        self.embed_dim = embed_dim
        self.class_per_task = [class_num]

        self.Separation = Separation
        self.FreezeBackbone = FreezeBackbone
        self.withPrompt = withPrompt
        self.withAdpt = withAdpt

        self.idOldModel = idOldModel

        self.isShareBlock = isShareBlock
        self.KeyToken = KeyToken
        self.instance_wise = instance_wise
        self.GeneralToken = GeneralToken

        self.base = base
        self.use_backbone = use_backbone

        self.prompt_length = prompt_length
        self.top_k = top_k

        PRETRAIN_PATH = "./jx_vit_base_p16_224-80ecf9dd.pth"

        if self.withPrompt:
            'x_embed就是图像的token, cls_feature是经过一个完整ViT模型后，进入最终分类器之前的embedding，论文中是q(x)'
            '这个模块的作用，传入图像的embed数据，然后根据这个数据从pool中选择出最适合的prompt，最后将其拼接起来并返回，这里同时计算了key的损失函数'
            self.prompt = Prompt(length=prompt_length, embed_dim=embed_dim, embedding_key='cls',
                                 prompt_pool=True, prompt_key=True, pool_size=pool_size, top_k=top_k,
                                 batchwise_prompt=True, proj = True )
        #
        if self.withAdpt:
            # self.adpt = ADPT_Block()
            self.adptList = nn.ModuleList([ADPT_Block() for _ in range(10)])
            self.dap_emb = torch.nn.Embedding(4, 16)  # 第一个是任务  第二个是t的embedding维度

        # backbone
        transformer = ViT(img_size=[256, 128] if use_backbone == False else [288, 144],
                          embed_dim=embed_dim,
                          stride_size=[12, 12],  # [12,12]
                          drop_path_rate=0.1,
                          drop_rate=0.03,
                          attn_drop_rate=0.0)
        transformer.load_param(PRETRAIN_PATH)

        self.trans = copy.deepcopy(transformer)
        self.trans.load_param('./model_data/pretrain29-12layer-256-128.pth')  # 完整的12层作为特征提取器

        print('Loading pretrained ImageNet model......from {}'.format(PRETRAIN_PATH))

        del transformer.blocks[10:12]

        if use_backbone:
            self.backbone = embed_net(class_num, non_local="off")
            self.patchembed = PatchEmbed(2048, embed_dim)
            self.num_patches = self.patchembed.num_patches
        else:  # 此分支
            self.patchembed = transformer.patch_embed
            self.num_patches = self.patchembed.num_patches
            self.sabs = transformer.blocks[:local_up_layers]

        self.pos_embed = transformer.pos_embed
        self.pos_drop = transformer.pos_drop
        self.l2norm = Normalize(2)
        self.norm = nn.LayerNorm(embed_dim)
        self.cls_token = transformer.cls_token

        if self.base == False:  # 此分支
            if self.KeyToken:  # 此分支
                self.key_tokens = nn.Parameter(torch.randn(4, embed_dim))
                self.query_token = transformer.cls_token
                self.queryBlock = Block(embed_dim, 12, attention_type=SpecificAttention)

            if self.GeneralToken:
                self.head_div = nn.Linear(embed_dim, class_num, bias=False)
                self.head_div.apply(weights_init_classifier)
            else:
                # 散度分类器
                self.head_div = None

            # 使用generalToken（这里可以替换为prompt）
            if self.GeneralToken:
                print()
                general_token = nn.Parameter(torch.randn(1, 1, self.embed_dim).cuda())
                trunc_normal_(general_token, std=.02)
                self.general_token = general_token
                self.general_classifier = nn.Linear(embed_dim, class_num, bias=False)
                self.general_classifier.apply(weights_init_classifier)
                self.generalBlock = Block(embed_dim, 12, attention_type=SpecificAttention)

            if self.KeyToken:  # 此分支 用于创建BN层
                if self.GeneralToken:
                    bottleneck = nn.BatchNorm1d(embed_dim)
                    bottleneck.bias.requires_grad_(False)
                    bottleneck.apply(weights_init_kaiming)
                    self.bottlenecks = nn.ModuleList([bottleneck])  # 这里会对每一个key使用归一化层

                    self.bottleneck_general = nn.BatchNorm1d(embed_dim)
                    self.bottleneck_general.bias.requires_grad_(False)
                    self.bottleneck_general.apply(weights_init_kaiming)

                    self.bottleneck2 = nn.BatchNorm1d(embed_dim)
                    self.bottleneck2.bias.requires_grad_(False)
                    self.bottleneck2.apply(weights_init_kaiming)  # 在div_logic中使用
                else:
                    bottleneck = nn.BatchNorm1d(embed_dim)
                    bottleneck.bias.requires_grad_(False)
                    bottleneck.apply(weights_init_kaiming)
                    self.bottlenecks = nn.ModuleList([bottleneck])  # 这里会对每一个key使用归一化层
            else:
                if self.GeneralToken:
                    self.bottleneck = nn.BatchNorm1d(2 * embed_dim)
                    self.bottleneck.bias.requires_grad_(False)
                    self.bottleneck.apply(weights_init_kaiming)  # 这里没有多个key了

                    self.bottleneck_general = nn.BatchNorm1d(embed_dim)
                    self.bottleneck_general.bias.requires_grad_(False)
                    self.bottleneck_general.apply(weights_init_kaiming)

                    self.bottleneck2 = nn.BatchNorm1d(embed_dim)
                    self.bottleneck2.bias.requires_grad_(False)
                    self.bottleneck2.apply(weights_init_kaiming)
                else:
                    self.bottleneck = nn.BatchNorm1d(embed_dim)
                    self.bottleneck.bias.requires_grad_(False)
                    self.bottleneck.apply(weights_init_kaiming)

                    self.bottleneck2 = nn.BatchNorm1d(embed_dim)
                    self.bottleneck2.apply(weights_init_kaiming)

            if self.KeyToken:  # 此分支 用于创建fusion层 和 分类层
                if self.GeneralToken:
                    fusion_block = nn.Linear(2 * embed_dim, embed_dim, bias=False)
                    self.fusion_blocks = nn.ModuleList([fusion_block])

                    classifier = nn.Linear(embed_dim, class_num, bias=False)
                    classifier.apply(weights_init_classifier)
                    self.indiv_classifiers = nn.ModuleList([classifier])  # 每一个任务的分类器是独立的
                else:
                    classifier = nn.Linear(embed_dim, class_num, bias=False)
                    classifier.apply(weights_init_classifier)
                    self.indiv_classifiers = nn.ModuleList([classifier])
            else:
                if self.GeneralToken:
                    self.classifier = nn.Linear(2 * embed_dim, class_num, bias=False)  # 这里实现的还是fusion的功能
                    self.classifier.apply(weights_init_classifier)
                else:
                    self.classifier = nn.Linear(embed_dim, class_num, bias=False)  # 最普通的分类器
                    self.classifier.apply(weights_init_classifier)

        # 此分支使用基本的模块  这里删去special block，只使用ViT模块
        else:
            self.class_token = transformer.cls_token
            self.norm = nn.LayerNorm(embed_dim)

            self.bottleneck = nn.BatchNorm1d(embed_dim)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)

            self.classifier = nn.Linear(embed_dim, class_num, bias=False)
            self.classifier.apply(weights_init_classifier)

    def prompt_forward(self, x, training_phase, train, selection=None, task_id_estimated_emb=None, prompt_length = 40):
        def inner_loop(x):
            "class token   实例prompt   原始图像token"
            x = self.pos_drop(x)
            for i, blk in enumerate(self.sabs):
                x = blk(x)  # ([2*batch, 221, 768])  # 因为这里加入了prompt和cls token
                if i != 9:
                    x = self.adptList[i + 1](x, task_id_estimated_emb=task_id_estimated_emb, isFirst=False)

            return x
        return inner_loop(x)


    def forward(self, x1, x2, modal=0, training_phase=1):
        out = dict()

        if self.use_backbone:
            x = self.backbone(x1, x2, modal)
        else:
            if modal == 0:
                x = torch.cat((x1, x2), 0)
            elif modal == 1:
                x = x1
            elif modal == 2:
                x = x2
            elif modal == 3:
                x = torch.cat((x1, x2), 0)

        x_copy = x.clone()
        # 图像打成块
        x = self.patchembed(x)
        cls_features, x_feat = self.trans(x_copy)
        del x_copy

        B, L, N = x.shape
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.pos_embed.to(x.dtype)
        # 这里需要假如在cls之前
        'l2p模型'
        res = self.prompt(x, prompt_mask=None, cls_features=cls_features)
        del cls_features
        x = res['prompted_embedding']

        if self.base == False:
            query_token_norm = None
            feat = None


            # query-key mechanism
            if self.KeyToken:  # 原教旨主义，使用预训练的blk来处理query
                # x_query = self.prompt_forward(x, training_phase, True, None)
                x_feat = x_feat.detach()
                # x_detach = cls_features.detach()
                query_token = self.query_token.expand(B, -1, -1)  # querytoken是transformer的cls_token
                query_token = self.queryBlock(torch.cat([query_token, x_feat], dim=1))  # 论文中的拼接
                query_token_norm = self.l2norm(query_token[:, 0])  # B*2,embed_dim 把cls token取出来

                del x_feat

                key_token_norm = self.l2norm(self.key_tokens[training_phase - 1].unsqueeze(0))  # 1,embed_dim

                # query_token_norm = self.l2norm(x_detach)
                out["query"] = query_token_norm
                out["current_key"] = key_token_norm
                # out['reduce_sim'] = res['reduce_sim']
                if training_phase > 1:
                    out["old_keys"] = self.l2norm(self.key_tokens[:training_phase - 1])

            # 这里已经完成prompt的选择了
            if self.training:
                if self.KeyToken:
                    if self.GeneralToken:
                        print()
                    else:
                        task_id_estimated_emb = self.dap_emb(torch.tensor(training_phase-1).to("cuda").unsqueeze(0).expand(x.shape[0], -1))

                        x = self.adptList[0](x, task_id_estimated_emb=task_id_estimated_emb, isFirst = True)
                        '''只使用图片token进行实例级别图像生成'''
                        '''嵌入完成后开始传播'''
                        if self.withPrompt:  # 在前向过程的block中就需要加入prompt指导训练
                            x = self.prompt_forward(x, training_phase=None, train=False, selection=None,
                                                    task_id_estimated_emb=task_id_estimated_emb,prompt_length=self.prompt_length*self.top_k)
                        else:
                            x = self.pos_drop(x)
                            if self.use_backbone == False:
                                for blk in self.sabs:
                                    x = blk(x)  # [2*batch, 211, 768]
                                    if i != 9:
                                        x = self.adpt(x, task_id_estimated_emb=task_id_estimated_emb, isFirst=False)

                        # 这里需要把过往任务的输出。需要经过task次ViT？？？
                        old_logits = []
                        old_feats = []
                        # x = self.norm(self.prompt_forward(x, 0, True, None)[:, 0, :])
                        # x = self.norm(x)
                        # 此任务之前的所有特征。 这里修改为使用prompt的方式，
                        for i in range(training_phase):
                            old_feats.append(self.norm(x[:, 0, :]))
                            old_logits.append(self.indiv_classifiers[i](self.bottlenecks[i](old_feats[i])))
                        # 所有历史任务的特征信息
                        out["old_logits"] = old_logits[:-1]
                        out["old_feats"] = old_feats[:-1]

                        out["feat"] = old_feats[-1]  # 经过msa输出的特征
                        out["logits"] = old_logits[-1]
                        # print(old_logits[:-1], old_logits[-1])
                        if self.head_div != None:
                            out["div_logits"] = self.head_div(self.bottleneck2(old_feats[-1]))

            # 推理
            else:
                with torch.no_grad():
                    if self.KeyToken:
                        # compute the similarities  先计算出所有的key
                        key_token_norm = self.l2norm(self.key_tokens[:training_phase])
                        sim = torch.matmul(query_token_norm, key_token_norm.t())  # B,training_phase
                        idx_cf_bn = torch.max(sim, dim=1)[1]
                        # for debug
                        out["idx"] = idx_cf_bn
                        out["sim"] = sim

                        # idx = res['prompt_idx']
                        # sim = res['similarity']
                        # print("id", idx)
                        # for debug
                        # out["idx"] = idx  # [B*2]  # 每一个batch拥有一个idx
                        # out["sim"] = sim  # [B*2, 1]

                        token_id, id_counts = torch.unique(idx_cf_bn, return_counts=True)
                        idx = token_id[id_counts.argmax()]
                        task_id_estimated_emb = self.dap_emb(idx.clone().detach().to("cuda").unsqueeze(0).expand(x.shape[0], -1))

                        x = self.adptList[0](x, task_id_estimated_emb=task_id_estimated_emb, isFirst=True)
                        '''嵌入完成后开始传播'''
                        if self.withPrompt:  # 在前向过程的block中就需要加入prompt指导训练
                            x = self.prompt_forward(x, training_phase=None, train=False, selection=None,
                                                    task_id_estimated_emb = task_id_estimated_emb, prompt_length=self.prompt_length*self.top_k)
                        else:
                            x = self.pos_drop(x)
                            if self.use_backbone == False:
                                for i, blk in enumerate(self.sabs):
                                    x = blk(x)  # [2*batch, 211, 768]
                                    # x = self.adpt(x, task_id_estimated_emb=task_id_estimated_emb, isFirst=False)
                                    if i != 9:
                                        self.adptList[i+1](x, task_id_estimated_emb=task_id_estimated_emb, isFirst=False)
                        x_pool = x[:, 0, :]  # [b, 768]  # 取出cls_token作为最终的特征
                        x_pool = self.norm(x_pool)
                        '''待改进：修改成实例级别的推理'''
                        # instance-wise
                        if self.instance_wise:  # 这里的意思是单独一个实例选择一个prompt还是一个batch选择一个prompt
                            if self.GeneralToken:
                                print()

                            else:
                                # 这里应该是加了prompt后的不同输出
                                '''这里有个关键点：原模型中tokens会随着任务量的增加而增加'''
                                # 这里的feat要将所有的任务都放进去，因为需要这个来作推理工作

                                feat = torch.zeros((x_pool.shape[0], self.embed_dim),
                                                   device="cuda")  # [B*2, 768]
                                for i, id in enumerate(idx_cf_bn):  # 把batch和id对应起来
                                    feat[i] = self.bottlenecks[id](x_pool[i, :].unsqueeze(0))
                        # batch-wise
                        else:
                            token_id, id_counts = torch.unique(idx_cf_bn, return_counts=True)
                            idx = token_id[id_counts.argmax()]
                            if self.GeneralToken:
                                # x_pool = torch.cat((tokens[int(idx)],general_token),dim = 1)
                                print()
                            else:
                                feat = self.bottlenecks[idx](x_pool)  # 原来是idx来选择BN层
                    else:
                        print()

                    out["test"] = self.l2norm(feat)

        else:  # base是True 就是啥也没有
            x += self.pos_embed[:, :L]
            x = self.pos_drop(x)

            self.head_div = None

            for blk in self.sabs:
                x = blk(x)

            class_token = self.class_token.expand(B, -1, -1)
            # for blk in self.tabs:
            #     class_token = blk(torch.cat([class_token,x],dim = 1),token_nums = 1)
            x_pool = class_token[:, 0]
            feat = self.bottleneck(x_pool)

            if self.training:
                out["feat"] = x_pool
                out["logits"] = self.classifier(feat)
            else:
                out["test"] = self.l2norm(feat)
        return out

    @property
    def feat_dim(self):
        if self.base:
            return self.embed_dim
        if (self.GeneralToken and self.KeyToken) or (self.KeyToken == True, self.GeneralToken == False):
            return self.embed_dim
        elif self.GeneralToken:
            return (len(self.class_per_task) + 1) * self.embed_dim
        else:
            return (len(self.class_per_task)) * self.embed_dim

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def freeze_ViT(self, net):
        for name, param in net.named_parameters():
            param.requires_grad_(False)
        # Double check
        enabled = set()
        for name, param in net.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

    def freeze_backbone(self, sinet, training_phase):
        for name, param in sinet.named_parameters():
            param.requires_grad_(False)
            if "prompt_pool" + "." + str(training_phase) in name:
                param.requires_grad_(True)

        # Double check
        enabled = set()
        for name, param in sinet.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

    def update_id(self):  # 当前任务id
        self.sinet.update_fc()  # 当前任务id

    def add_model(self, new_classes, init_loader):
        self.class_per_task.append(new_classes)
        old_classes = sum(self.class_per_task[:-1])

        if self.base == False:
            if self.KeyToken:
                if self.GeneralToken:
                    # add a new independent classifier for new task
                    classifier = nn.Linear(self.embed_dim, new_classes, bias=False)
                    classifier.apply(weights_init_classifier)
                    self.indiv_classifiers.append(classifier)

                    # add a new independent batchnorm for new task
                    bottleneck = nn.BatchNorm1d(self.embed_dim)
                    bottleneck.bias.requires_grad_(False)
                    bottleneck.apply(weights_init_kaiming)
                    self.bottlenecks.append(bottleneck)

                    # add a new fusion module
                    fusion_block = nn.Linear(2 * self.embed_dim, self.embed_dim, bias=False)
                    self.fusion_blocks.append(fusion_block)

                    org_general_classifier_params = self.general_classifier.weight.data
                    self.general_classifier = torch.nn.Linear(self.embed_dim, sum(self.class_per_task), bias=False)
                    self.general_classifier.apply(weights_init_classifier)
                    self.general_classifier.weight.data[:old_classes].copy_(org_general_classifier_params)

                else:
                    classifier = nn.Linear(self.embed_dim, new_classes, bias=False)
                    classifier.apply(weights_init_classifier)
                    self.indiv_classifiers.append(classifier)

                    bottleneck = nn.BatchNorm1d(self.embed_dim)
                    bottleneck.bias.requires_grad_(False)
                    bottleneck.apply(weights_init_kaiming)
                    self.bottlenecks.append(bottleneck)
            else:
                if self.GeneralToken:
                    self.classifier = torch.nn.Linear((len(self.class_per_task) + 1) * self.embed_dim,
                                                      sum(self.class_per_task), bias=False)
                    self.classifier.apply(weights_init_classifier)

                    org_general_classifier_params = self.general_classifier.weight.data
                    self.general_classifier = torch.nn.Linear(self.embed_dim, sum(self.class_per_task), bias=False)
                    self.general_classifier.apply(weights_init_classifier)
                    self.general_classifier.weight.data[:old_classes].copy_(org_general_classifier_params)

                    org_num_features = self.bottleneck.num_features
                    org_bottleneck_running_mean = self.bottleneck.running_mean.data
                    org_bottleneck_running_var = self.bottleneck.running_var.data
                    org_bottleneck_weight = self.bottleneck.weight.data
                    self.bottleneck = nn.BatchNorm1d((len(self.class_per_task) + 1) * self.embed_dim)
                    self.bottleneck.bias.requires_grad_ = False
                    self.bottleneck.apply(weights_init_kaiming)
                    self.bottleneck.running_mean.data[:org_num_features] = org_bottleneck_running_mean
                    self.bottleneck.running_var.data[:org_num_features] = org_bottleneck_running_var
                    self.bottleneck.weight.data[:org_num_features] = org_bottleneck_weight
                else:
                    self.classifier = torch.nn.Linear(len(self.class_per_task) * self.embed_dim,
                                                      sum(self.class_per_task), bias=False)
                    self.classifier.apply(weights_init_classifier)

                    org_num_features = self.bottleneck.num_features
                    org_bottleneck_running_mean = self.bottleneck.running_mean.data
                    org_bottleneck_running_var = self.bottleneck.running_var.data
                    org_bottleneck_weight = self.bottleneck.weight.data
                    self.bottleneck = nn.BatchNorm1d((len(self.class_per_task)) * self.embed_dim)
                    self.bottleneck.bias.requires_grad_ = False
                    self.bottleneck.apply(weights_init_kaiming)
                    self.bottleneck.running_mean.data[:org_num_features] = org_bottleneck_running_mean
                    self.bottleneck.running_var.data[:org_num_features] = org_bottleneck_running_var
                    self.bottleneck.weight.data[:org_num_features] = org_bottleneck_weight

            # re-initialize the divergence classifier  # 这里修改了，原始的是有这个损失函数的
            # self.head_div = nn.Linear(self.embed_dim, new_classes + 1, bias=False)
            # self.head_div.apply(weights_init_classifier)
            #
            # self.bottleneck2 = nn.BatchNorm1d(self.embed_dim)
            # self.bottleneck2.apply(weights_init_kaiming)

            # add new specific tokens and specific blocks
            # task_token = copy.deepcopy(self.task_tokens[-1])
            # trunc_normal_(task_token, std=.02)
            # self.task_tokens.append(task_token)

            # if self.isShareBlock == False:
            #     specific_block = copy.deepcopy(self.tabs[-1])
            #     self.tabs.append(specific_block)
        else:
            # baseline
            org_classifier_params = self.classifier.weight.data
            self.classifier = nn.Linear(self.embed_dim, old_classes + new_classes, bias=False)
            self.classifier.weight.data[:old_classes].copy_(org_classifier_params)
            class_centers = initial_classifier(self, init_loader)
            self.classifier.weight.data[old_classes:].copy_(class_centers)

            self.bottleneck = nn.BatchNorm1d(self.embed_dim)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)

        self.cuda()

# 最基础的版本，加了prompt，加了query机制。
# net = MyModel(class_num=10, embed_dim=768, KeyToken=True, GeneralToken=False,
#                isShareBlock=False, base=False,instance_wise=True, withPrompt=True).to('cpu')
# net.eval()
# net(torch.randn(2,3,256,128).to('cpu'), torch.randn(2,3,256,128).to('cpu'))
