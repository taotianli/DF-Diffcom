import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
import logging
import torch.nn.functional as F

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
             DoubleConv(in_channels, out_channels),
        )
        self.emb_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:,:, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    

class Down_img(nn.Module):
    '''
    Down-sampling for images
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
             DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels//2)
        )

        self.emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        # print('scale up', x.shape)
        x = torch.cat([skip_x, x], dim=1)

        x = self.conv(x)
        emb = self.emb(t)[:,:, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        return x + emb
    
class SelfAttention(nn.Module):
    def __init__(self, in_channels, size):
        super().__init__()
        self.channels = in_channels
        self.size = size
        self.mha = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm(in_channels)
        self.ff_self = nn.Sequential(
            nn.LayerNorm([in_channels]),
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value

        return attention_value.swapaxes(1, 2).view(-1, self.channels, self.size, self.size)


class CrossAttention(nn.Module):##需要做修改
    def __init__(self, in_channels, size):
        super().__init__()
        self.channels = in_channels
        self.size = size
        self.mha = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm(in_channels)
        self.ff_self = nn.Sequential(
            nn.LayerNorm([in_channels]),
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
        )

    def forward(self, x, y):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        y = y.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        y_ln = self.ln(y)
        attention_value, _ = self.mha(x_ln, y_ln, y_ln)
        x = x + attention_value
        x = self.ff_self(x) + x
        
        return x.swapaxes(1, 2).view(-1, self.channels, self.size, self.size)
    
    
class GAM_Attention(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out

class ImageEncoder(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256):
        super(ImageEncoder, self).__init__()
        self.conv1 = DoubleConv(c_in, 64)
        self.conv2 = DoubleConv(64, c_out)
        self.pool = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(c_out * 60 * 60, 1024)
        self.fc1 = nn.Linear(c_out * 24 * 24, 1024)
        self.fc2 = nn.Linear(1024, time_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 1,1,240,240
        x = self.pool(x)# 1,1,120,120
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = self.pool(x)# 1,1,60,60
        # print(x.shape)
        # x = x.view(-1, 60 *60)
        x = x.view(-1, 24*24)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # print('final emb shape',x.shape)
        return x
    
class ImageEncoder_new(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256):
        super(ImageEncoder_new, self).__init__()
        self.conv1 = DoubleConv(c_in, 64)
        self.conv2 = DoubleConv(64, c_out)
        self.pool = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(c_out * 60 * 60, 1024)
        self.fc1 = nn.Linear(60*60, 1024)
        self.fc2 = nn.Linear(1024, time_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 1,1,240,240
        x = self.pool(x)# 1,1,120,120
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = self.pool(x)# 1,1,60,60
        print(x.shape)
        # x = x.view(-1, 60 *60)
        x = x.view(-1, 60*60)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # print('final emb shape',x.shape)
        return x



class UNet_conditional(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        # self.sa1 = SelfAttention(128, 120)
        # self.down2 = Down(128, 256)
        # self.sa2 = SelfAttention(256, 60)
        # self.down3 = Down(256, 256)
        # self.sa3 = SelfAttention(256, 30)
        self.sa1 = SelfAttention(128, 48)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 24)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 12)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        # self.sa4 = SelfAttention(128, 60)
        # self.up2 = Up(256, 64)
        # self.sa5 = SelfAttention(64, 120)
        # self.up3 = Up(128, 64)
        # self.sa6 = SelfAttention(64, 240)
        self.sa4 = SelfAttention(128, 24)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 48)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 96)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.image_encoder = ImageEncoder(c_in, c_out, self.time_dim)
        # if num_classes is not None:
        #     self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        # print('time embedding finished')
        if y is not None:
            t += self.image_encoder(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        # print('x2 shape', x2.shape)
        # x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        # print('x3 shape', x3.shape)
        # x3 = self.sa2(x3)
        
        x4 = self.down3(x3, t)
        # x4 = self.sa3(x4)
        # print('x4 shape', x4.shape)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        # print('x4 shape', x4.shape)

        x = self.up1(x4, x3, t)
        # print('x shape up 1', x.shape)
        # x = self.sa4(x)
        # print('sa4 shape', x.shape)
        x = self.up2(x, x2, t)
        # print('x shape up 2', x.shape)
        # x = self.sa5(x)
        # print('sa5 shape', x.shape)
        x = self.up3(x, x1, t)
        # x = self.sa6(x)
        output = self.outc(x)
        return output

class UNet_conditional_with_attention(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 48)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 24)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 12)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 24)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 48)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 96)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.image_encoder = ImageEncoder(c_in, c_out, self.time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        if y is not None:
            t += self.image_encoder(y)
        
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        # print('x2 shape', x2.shape)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        # print('x3 shape', x3.shape)
        x3 = self.sa2(x3)
        
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        # print('x4 shape', x4.shape)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        # print('x4 shape', x4.shape)

        x = self.up1(x4, x3, t)
        # print('x shape up 1', x.shape)
        x = self.sa4(x)
        # print('sa4 shape', x.shape)
        x = self.up2(x, x2, t)
        # print('x shape up 2', x.shape)
        x = self.sa5(x)
        # print('sa5 shape', x.shape)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    
class UNet_conditional_concat(nn.Module):
    '''
    在输入上concat一次
    '''
    def __init__(self, c_in=2, c_out=1, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 48)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 24)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 12)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 24)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 48)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 96)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.image_encoder = ImageEncoder(1, c_out, self.time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        # print(y.shape)
        if y is not None:
            t += self.image_encoder(y)
        x = torch.concat([x, y], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        # print('x2 shape', x2.shape)
        # x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        # print('x3 shape', x3.shape)
        x3 = self.sa2(x3)
        
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        # print('x4 shape', x4.shape)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        # print('x4 shape', x4.shape)

        x = self.up1(x4, x3, t)
        # print('x shape up 1', x.shape)
        x = self.sa4(x)
        # print('sa4 shape', x.shape)
        x = self.up2(x, x2, t)
        # print('x shape up 2', x.shape)
        x = self.sa5(x)
        # print('sa5 shape', x.shape)
        x = self.up3(x, x1, t)
        # x = self.sa6(x)
        output = self.outc(x)
        return output
    
class UNet_conditional_fully_concat(nn.Module):
    '''
    在每次downsampling时都concat下采样的原图
    '''
    def __init__(self, c_in=1, c_out=1, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in+1, 64)
        self.down1 = Down(64+4, 128)
        self.sa1 = SelfAttention(128, 48)
        self.down2 = Down(128+8, 256)
        self.sa2 = SelfAttention(256, 24)
        self.down3 = Down(256+16, 256)
        self.sa3 = SelfAttention(256, 12)

        self.bot1 = DoubleConv(256+16, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 24)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 48)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 96)

        self.inc_y = DoubleConv(1, 4)
        self.down1_y = Down_img(4, 8)
        self.down2_y = Down_img(8, 16)
        self.down3_y = Down_img(16, 16)

        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.image_encoder = ImageEncoder(1, c_out, self.time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        # print(y.shape)
        if y is not None:
            t += self.image_encoder(y)
        
        x = torch.concat([x, y], dim=1)
        x1 = self.inc(x)
        y1 = self.inc_y(y)
        x1_1 = torch.concat([x1, y1], dim=1)

        x2 = self.down1(x1_1, t)
        # print('x2 shape', x2.shape)
        x2 = self.sa1(x2)
        y2 = self.down1_y(y1)
        x2_1 = torch.concat([x2, y2], dim=1)


        x3 = self.down2(x2_1, t)
        # print('x3 shape', x3.shape)
        x3 = self.sa2(x3)
        y3 = self.down2_y(y2)
        x3_1 = torch.concat([x3, y3], dim=1)
        
        x4 = self.down3(x3_1, t)
        x4 = self.sa3(x4)
        y4 = self.down3_y(y3)
        x4_1 = torch.concat([x4, y4], dim=1)
        # print('x4 shape', x4.shape)

        x4 = self.bot1(x4_1)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        # print('x4 shape', x4.shape)

        x = self.up1(x4, x3, t)
        # print('x shape up 1', x.shape)
        x = self.sa4(x)
        # print('sa4 shape', x.shape)
        x = self.up2(x, x2, t)
        # print('x shape up 2', x.shape)
        x = self.sa5(x)
        # print('sa5 shape', x.shape)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    
class UNet_conditional_fully_add(nn.Module):
    '''
    在每次downsampling时都concat下采样的原图
    '''
    def __init__(self, c_in=1, c_out=1, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 48)
        self.ca1 = CrossAttention(128, 48)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 24)
        self.ca2 = CrossAttention(256, 24)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 12)
        self.ca3 = CrossAttention(256, 12)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 24)
        self.ca4 = CrossAttention(128, 24)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 48)
        self.ca5 = CrossAttention(64, 48)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 96)
        self.ca6 = CrossAttention(64, 96)

        self.inc_y = DoubleConv(c_in, 64)
        self.down1_y = Down_img(64, 128)
        self.down2_y = Down_img(128, 256)
        self.down3_y = Down_img(256, 256)

        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        # self.image_encoder = ImageEncoder(1, c_out, self.time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        # print(y.shape)
        # if y is not None:
        #     t += self.image_encoder(y)
        
        x = x + y
        x1 = self.inc(x)
        y1 = self.inc_y(y)
        x1_1 = x1 + y1

        x2 = self.down1(x1_1, t)
        # print('x2 shape', x2.shape)
        # x2 = self.sa1(x2)
        
        y2 = self.down1_y(y1)
        x2 = self.ca1(x2, y2)
        # print('x2 shape', x2.shape)
        x2_1 = x2 + y2

        x3 = self.down2(x2_1, t)
        # print('x3 shape', x3.shape)
        
        y3 = self.down2_y(y2)
        x3 = self.ca2(x3, y3)
        x3_1 = x3 + y3
        
        x4 = self.down3(x3_1, t)
        
        y4 = self.down3_y(y3)
        x4 = self.ca3(x4, y4)
        x4_1 = x4 + y4
        # print('x4 shape', x4.shape)

        x4 = self.bot1(x4_1)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        # print('x4 shape', x4.shape)

        x = self.up1(x4, x3, t)
        # print('x shape up 1', x.shape)
        # x = self.ca4(x)
        # print('sa4 shape', x.shape)
        x = self.up2(x, x2, t)
        # print('x shape up 2', x.shape)
        # x = self.sa5(x)
        # print('sa5 shape', x.shape)
        x = self.up3(x, x1, t)
        # x = self.sa6(x)
        output = self.outc(x)
        return output

class UNet_conditional_fully_concat_cross_attention(nn.Module):
    '''
    在每次downsampling时都concat下采样的原图
    '''
    def __init__(self, c_in=1, c_out=1, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in+1, 64)
        self.down1 = Down(64+4, 128)
        self.sa1 = SelfAttention(128, 48)
        self.down2 = Down(128+8, 256)
        self.sa2 = SelfAttention(256, 24)
        self.down3 = Down(256+16, 256)
        self.sa3 = SelfAttention(256, 12)

        self.bot1 = DoubleConv(256+16, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 24)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 48)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 96)

        self.inc_y = DoubleConv(1, 4)
        self.down1_y = Down_img(4, 8)
        self.down2_y = Down_img(8, 16)
        self.down3_y = Down_img(16, 16)

        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.image_encoder = ImageEncoder(1, c_out, self.time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        # print(y.shape)
        if y is not None:
            t += self.image_encoder(y)
        
        x = torch.concat([x, y], dim=1)
        x1 = self.inc(x)
        y1 = self.inc_y(y)
        x1_1 = torch.concat([x1, y1], dim=1)

        x2 = self.down1(x1_1, t)
        # print('x2 shape', x2.shape)
        # x2 = self.sa1(x2)
        y2 = self.down1_y(y1)
        x2_1 = torch.concat([x2, y2], dim=1)


        x3 = self.down2(x2_1, t)
        # print('x3 shape', x3.shape)
        # x3 = self.sa2(x3)
        y3 = self.down2_y(y2)
        x3_1 = torch.concat([x3, y3], dim=1)
        
        x4 = self.down3(x3_1, t)
        # x4 = self.sa3(x4)
        y4 = self.down3_y(y3)
        x4_1 = torch.concat([x4, y4], dim=1)
        # print('x4 shape', x4.shape)

        x4 = self.bot1(x4_1)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        # print('x4 shape', x4.shape)

        x = self.up1(x4, x3, t)
        # print('x shape up 1', x.shape)
        # x = self.sa4(x)
        # print('sa4 shape', x.shape)
        x = self.up2(x, x2, t)
        # print('x shape up 2', x.shape)
        # x = self.sa5(x)
        # print('sa5 shape', x.shape)
        x = self.up3(x, x1, t)
        # x = self.sa6(x)
        output = self.outc(x)
        return output
    
class UNet_conditional_concat_pseudo_3D(nn.Module):
    '''
    在输入上concat一次
    '''
    def __init__(self, c_in=4, c_out=1, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 48)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 24)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 12)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 24)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 48)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 96)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.image_encoder = ImageEncoder(1, c_out, self.time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        # print(y.shape)
        if y is not None:
            t += self.image_encoder(y)
        x = torch.concat([x, y], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        # print('x2 shape', x2.shape)
        # x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        # print('x3 shape', x3.shape)
        x3 = self.sa2(x3)
        
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        # print('x4 shape', x4.shape)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        # print('x4 shape', x4.shape)

        x = self.up1(x4, x3, t)
        # print('x shape up 1', x.shape)
        x = self.sa4(x)
        # print('sa4 shape', x.shape)
        x = self.up2(x, x2, t)
        # print('x shape up 2', x.shape)
        x = self.sa5(x)
        # print('sa5 shape', x.shape)
        x = self.up3(x, x1, t)
        # x = self.sa6(x)
        output = self.outc(x)
        return output
    
class UNet_conditional_concat_with_mask(nn.Module):
    '''
    在输入上concat一次
    '''
    def __init__(self, c_in=3, c_out=1, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 48)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 24)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 12)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 24)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 48)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 96)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.image_encoder = ImageEncoder(1, c_out, self.time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    

    def forward(self, x, t, y, m):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        # print(y.shape)
        if y is not None:
            t += self.image_encoder(y)
        x = torch.concat([x, y, m], dim=1)
        # print(x.shape)
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        # print('x2 shape', x2.shape)
        # x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        # print('x3 shape', x3.shape)
        x3 = self.sa2(x3)
        
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        # print('x4 shape', x4.shape)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        # print('x4 shape', x4.shape)

        x = self.up1(x4, x3, t)
        # print('x shape up 1', x.shape)
        x = self.sa4(x)
        # print('sa4 shape', x.shape)
        x = self.up2(x, x2, t)
        # print('x shape up 2', x.shape)
        x = self.sa5(x)
        # print('sa5 shape', x.shape)
        x = self.up3(x, x1, t)
        # x = self.sa6(x)
        output = self.outc(x)
        return output
    
class UNet_conditional_concat_with_mask_v2(nn.Module):
    '''
    在输入上concat一次
    mask通过class free的方式加进去
    '''
    def __init__(self, c_in=2, c_out=1, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 48)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 24)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 12)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 24)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 48)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 96)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.image_encoder = ImageEncoder(1, c_out, self.time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    

    def forward(self, x, t, y, m):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        # print(y.shape)
        # if y is not None:
        #     t += self.image_encoder(m)
        x = torch.concat([x, y], dim=1)
        # print(x.shape,y.shape)
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        # print('x2 shape', x2.shape)
        # x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        # print('x3 shape', x3.shape)
        x3 = self.sa2(x3)
        
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        # print('x4 shape', x4.shape)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        # print('x4 shape', x4.shape)

        x = self.up1(x4, x3, t)
        # print('x shape up 1', x.shape)
        x = self.sa4(x)
        # print('sa4 shape', x.shape)
        x = self.up2(x, x2, t)
        # print('x shape up 2', x.shape)
        x = self.sa5(x)
        # print('sa5 shape', x.shape)
        x = self.up3(x, x1, t)
        # x = self.sa6(x)
        output = self.outc(x)
        return output
    
class UNet_conditional_concat_Large(nn.Module):
    '''
    在输入上concat一次
    大杯：96*96-->128*128
    '''
    def __init__(self, c_in=2, c_out=1, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 64)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 32)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 16)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 32)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 128)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.image_encoder = ImageEncoder_new(1, c_out, self.time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    

    def forward(self, x, t, y, m):
        
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        # print(y.shape)
        print(t.shape)
        if y is not None:
            t += self.image_encoder(m)
        x = torch.concat([x, y], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        # print('x2 shape', x2.shape)
        # x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        # print('x3 shape', x3.shape)
        x3 = self.sa2(x3)
        
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        # print('x4 shape', x4.shape)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        # print('x4 shape', x4.shape)

        x = self.up1(x4, x3, t)
        # print('x shape up 1', x.shape)
        x = self.sa4(x)
        # print('sa4 shape', x.shape)
        x = self.up2(x, x2, t)
        # print('x shape up 2', x.shape)
        x = self.sa5(x)
        # print('sa5 shape', x.shape)
        x = self.up3(x, x1, t)
        # x = self.sa6(x)
        output = self.outc(x)
        return output
    
class UNet_conditional_concat_XLarge(nn.Module):
    '''
    在输入上concat一次
    XL 256*256
    '''
    def __init__(self, c_in=3, c_out=1, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 120)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 60)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 30)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 60)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 120)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 240)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.image_encoder = ImageEncoder_new(1, c_out, self.time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    

    def forward(self, x, t, y, m):
        
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        # print(y.shape)
        print(t.shape)
        # if y is not None:
        #     t += self.image_encoder(m)
        x = torch.concat([x, y], dim=1)
        x = torch.concat([x, m], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        # print('x2 shape', x2.shape)
        # x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        # print('x3 shape', x3.shape)
        x3 = self.sa2(x3)
        
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        # print('x4 shape', x4.shape)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        # print('x4 shape', x4.shape)

        x = self.up1(x4, x3, t)
        # print('x shape up 1', x.shape)
        x = self.sa4(x)
        # print('sa4 shape', x.shape)
        x = self.up2(x, x2, t)
        # print('x shape up 2', x.shape)
        # x = self.sa5(x)
        # print('sa5 shape', x.shape)
        x = self.up3(x, x1, t)
        # x = self.sa6(x)
        output = self.outc(x)
        return output