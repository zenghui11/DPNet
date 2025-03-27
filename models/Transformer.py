
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mobilenetv2 import mobilenet_v2 
from torch import nn



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        att = self.sigmoid(out)
        out = torch.mul(x, att)
        return out
    
class DAM(nn.Module):
    def __init__(self, in_dim, kernel_size=7):
        super(DAM, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tp = nn.Conv2d(in_dim, 64, kernel_size=1)
        self.ca = ChannelAttention(64)

    def forward(self, x1, x2):
        max_out, _ = torch.max(x2, dim=1, keepdim=True)
        x2 = max_out
        x2 = self.conv1(x2)
        att2 = self.sigmoid(x2+x1)
        out = torch.mul(x1, att2) + x2
        tp = self.tp(out)
        fuseout = self.ca(tp)

        return fuseout
    
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class endecoder(nn.Module):
    '''
    提取全局特征的transformer
    '''
    def __init__(self, d_model, heads, dropout, activation, flag):
        super(endecoder, self).__init__()
        self.activition = _get_activation_fn(activation)
        self.flag = flag

        self.linear_q1 = nn.Linear(d_model, d_model)
        self.linear_k1 = nn.Linear(d_model, d_model)
        self.linear_v1 = nn.Linear(d_model, d_model)
        self.linear1_1 = nn.Linear(d_model, 2 * d_model)
        self.linear1_2 = nn.Linear(2 * d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout1_1 = nn.Dropout(dropout)
        self.dropout1_2 = nn.Dropout(dropout)
        self.multihead_attn1 = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm1_1 = nn.LayerNorm(d_model)
        self.norm1_2 = nn.LayerNorm(d_model)
        self.norm1_3 = nn.LayerNorm(d_model)

        self.linear_q2 = nn.Linear(d_model, d_model)
        self.linear_k2 = nn.Linear(d_model, d_model)
        self.linear_v2 = nn.Linear(d_model, d_model)
        self.linear2_1 = nn.Linear(d_model, 2 * d_model)
        self.linear2_2 = nn.Linear(2 * d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout2_1 = nn.Dropout(dropout)
        self.dropout2_2 = nn.Dropout(dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm2_1 = nn.LayerNorm(d_model)
        self.norm2_2 = nn.LayerNorm(d_model)
        self.norm2_3 = nn.LayerNorm(d_model)

    def forward(self, x):
        rgb1, depth1 = x
        rediual = rgb1
        rediual_d = depth1
        if self.flag == 1:
            rgb1 = self.norm1_1(rgb1)
            depth1 = self.norm2_1(depth1)
        q = self.linear_q1(rgb1)
        k = self.linear_k1(depth1)
        v = self.linear_v1(depth1)

        q2 = self.linear_q2(depth1)
        k2 = self.linear_k2(rgb1)
        v2 = self.linear_v2(rgb1)

        k = torch.cat((k, k2), dim=1)
        v = torch.cat((v, v2), dim=1)

        src1, src1_1 = self.multihead_attn1(q, k, v)
        res = rediual + self.dropout1_1(src1)
        res1 = self.norm1_2(res)
        res1 = self.linear1_2(self.dropout1(self.activition(self.linear1_1(res1))))
        res2 = res + self.dropout1_2(res1)

        src2, src2_2 = self.multihead_attn2(q2, k, v)
        res3 = rediual_d + self.dropout2_1(src2)
        res4 = self.norm2_2(res3)
        res4 = self.linear2_2(self.dropout2(self.activition(self.linear2_1(res4))))
        res5 = res3 + self.dropout2_2(res4)

        return res2, res5
    
class interactive(nn.Module):
    def __init__(self, n, d_model, heads, dropout, activation, pos_feats, num_pos_feats, ratio):
        super(interactive, self).__init__()
        self.trans = []
        self.conv1 = nn.Conv2d(d_model, d_model//ratio, kernel_size=(1, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(d_model, d_model//ratio, kernel_size=(1, 1), stride=(1, 1))
        self.conv3 = nn.Conv2d(d_model//ratio, d_model, kernel_size=(1, 1), stride=(1, 1))
        self.conv4 = nn.Conv2d(d_model//ratio, d_model, kernel_size=(1, 1), stride=(1, 1))
        flag1 = 0
        for i in range(n):
            if flag1 == 0:
                self.trans.append(endecoder(d_model//ratio, heads, dropout, activation, 0).to(device=1))
                flag1 += 1
            elif flag1 > 0:
                self.trans.append(endecoder(d_model//ratio, heads, dropout, activation, 0).to(device=1))

        self.transall = nn.Sequential(*self.trans)
        total_params1 = sum(p.numel() for p in self.transall.parameters())
        print('总参数量：{}'.format(total_params1))

    def forward(self, rgb, depth):
        n, c, h, w = rgb.size()
        rgb = self.conv1(rgb)
        depth = self.conv2(depth)
        rgb1 = torch.flatten(rgb, start_dim=2, end_dim=3).permute(0, 2, 1)
        depth1 = torch.flatten(depth, start_dim=2, end_dim=3).permute(0, 2, 1)

        x = self.transall((rgb1, depth1))
        rgb1, depth1 = x
        res = rgb1.permute(0, 2, 1)
        res1 = depth1.permute(0, 2, 1)
        output = res.reshape(n, c//2, h, w)
        output1 = res1.reshape(n, c//2, h, w)
        output = self.conv3(output)
        output1 = self.conv4(output1)

        return output, output1