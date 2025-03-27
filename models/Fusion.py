import os
import torch
import torch.nn as nn

from models.CMDAF import CMDAF, CMDAF1, CMDAF2, CMDAF3
from transformer.transformer import interactive
from transformer.cainet import DAM
from models.Common import reflect_conv
from models.Net import TransformerBlock, TransformerBlock2

os.environ['CUDA_VISIBLE_DEVICES'] ='0,1'


def Fusion(vi_out, ir_out):
    return torch.cat([vi_out, ir_out], dim=1)

class LowFreqEncoder(nn.Module):
    def __init__(self):
        super(LowFreqEncoder, self).__init__()

        self.vi_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=8, stride=1, padding=0)
        self.ir_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=8, stride=1, padding=0)

        self.vi_conv2 = reflect_conv(in_channels=8, kernel_size=3, out_channels=16, stride=1, pad=1)
        self.ir_conv2 = reflect_conv(in_channels=8, kernel_size=3, out_channels=16, stride=1, pad=1)

        self.vi_conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.ir_conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)

        self.vi_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.ir_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)

        self.transformer_block1 = TransformerBlock(dim=64, num_heads=8, ffn_expansion_factor=4,
                                                  bias=False, LayerNorm_type='WithBias')
        self.trans1 = interactive(n=2, d_model=16, heads=4, dropout=0.1, activation="relu", pos_feats=32, num_pos_feats=128, ratio=1)
        self.trans2 = interactive(n=2, d_model=32, heads=4, dropout=0.1, activation="relu", pos_feats=16, num_pos_feats=256, ratio=1)
        self.trans3 = interactive(n=2, d_model=64, heads=4, dropout=0.1, activation="relu", pos_feats=8, num_pos_feats=256, ratio=1)
        self.trans4 = interactive(n=2, d_model=64, heads=4, dropout=0.1, activation="relu", pos_feats=8, num_pos_feats=256, ratio=1)

    def forward(self, y_vi_image, ir_image):
        activate = nn.GELU()
        activate = activate.cuda()
        vi_out = activate(self.vi_conv1(y_vi_image))
        ir_out = activate(self.ir_conv1(ir_image)) # [16, 8, 64, 64]

        vi_out, ir_out = self.trans1(activate(self.vi_conv2(vi_out)), activate(self.ir_conv2(ir_out)))
        vi_out, ir_out = self.trans2(activate(self.vi_conv3(vi_out)), activate(self.ir_conv3(ir_out)))
        vi_out, ir_out = self.trans3(activate(self.vi_conv4(vi_out)), activate(self.ir_conv4(ir_out)))

        vi_out, ir_out = self.transformer_block1(vi_out, ir_out)#1次修改
        vi_out, ir_out = self.trans4(vi_out, ir_out)

        return vi_out, ir_out

class HighFreqEncoder(nn.Module):
    def __init__(self):
        super(HighFreqEncoder, self).__init__()

        self.vi_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=64, stride=1, padding=0)
        self.ir_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=64, stride=1, padding=0)

        self.vi_conv2 = nn.Conv2d(in_channels=64, kernel_size=1, out_channels=64, stride=1, padding=0)
        self.ir_conv2 = nn.Conv2d(in_channels=64, kernel_size=1, out_channels=64, stride=1, padding=0)

        self.vi_conv3 = nn.Conv2d(in_channels=64, kernel_size=1, out_channels=64, stride=1, padding=0)
        self.ir_conv3 = nn.Conv2d(in_channels=64, kernel_size=1, out_channels=64, stride=1, padding=0)

        self.transformer_block1 = TransformerBlock(dim=64, num_heads=8, ffn_expansion_factor=4,
                                                   bias=False, LayerNorm_type='WithBias')

        self.lfe1 = DAM(64)
        self.lfe2 = DAM(64)
        self.lfe3 = DAM(64)

    def forward(self, y_vi_image, ir_image):
        activate = nn.GELU()
        activate = activate.cuda()
        vi_out = activate(self.vi_conv1(y_vi_image))
        ir_out = activate(self.ir_conv1(ir_image))

        vi_out, ir_out = self.transformer_block1(vi_out, ir_out)

        f1 = self.lfe1(vi_out, ir_out)

        vi_out = activate(self.vi_conv2(f1 + vi_out))
        ir_out = activate(self.ir_conv2(f1 + ir_out))

        f2 = self.lfe2(vi_out, ir_out)

        vi_out = activate(self.vi_conv3(f2 + vi_out))
        ir_out = activate(self.ir_conv3(f2+ ir_out))

        f3 = self.lfe3(vi_out, ir_out)

        return f3 + vi_out, f3 + ir_out

# # 无GCA消融
class wo_LowEncoder(nn.Module):
    def __init__(self):
        super(wo_LowEncoder, self).__init__()

        self.vi_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=8, stride=1, padding=0)
        self.ir_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=8, stride=1, padding=0)

        self.vi_conv2 = reflect_conv(in_channels=8, kernel_size=3, out_channels=16, stride=1, pad=1)
        self.ir_conv2 = reflect_conv(in_channels=8, kernel_size=3, out_channels=16, stride=1, pad=1)

        self.vi_conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.ir_conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)

        self.vi_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.ir_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)

        self.transformer_block1 = TransformerBlock(dim=64, num_heads=8, ffn_expansion_factor=4,
                                                  bias=False, LayerNorm_type='WithBias')

    def forward(self, y_vi_image, ir_image):
        activate = nn.GELU()
        activate = activate.cuda()
        vi_out = activate(self.vi_conv1(y_vi_image))
        ir_out = activate(self.ir_conv1(ir_image)) # [16, 8, 64, 64]

        vi_out, ir_out = (activate(self.vi_conv2(vi_out)), activate(self.ir_conv2(ir_out)))
        vi_out, ir_out = (activate(self.vi_conv3(vi_out)), activate(self.ir_conv3(ir_out)))
        vi_out, ir_out = (activate(self.vi_conv4(vi_out)), activate(self.ir_conv4(ir_out)))

        vi_out, ir_out = self.transformer_block1(vi_out, ir_out)#1次修改
        vi_out, ir_out = (vi_out, ir_out)

        return vi_out, ir_out

# #无LFA加入消融
class wo_HighEncoder(nn.Module):
    def __init__(self):
        super(wo_HighEncoder, self).__init__()

        self.vi_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=64, stride=1, padding=0)
        self.ir_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=64, stride=1, padding=0)

        self.vi_conv2 = nn.Conv2d(in_channels=64, kernel_size=1, out_channels=64, stride=1, padding=0)
        self.ir_conv2 = nn.Conv2d(in_channels=64, kernel_size=1, out_channels=64, stride=1, padding=0)

        self.vi_conv3 = nn.Conv2d(in_channels=64, kernel_size=1, out_channels=64, stride=1, padding=0)
        self.ir_conv3 = nn.Conv2d(in_channels=64, kernel_size=1, out_channels=64, stride=1, padding=0)

        self.transformer_block1 = TransformerBlock(dim=64, num_heads=8, ffn_expansion_factor=4,
                                                   bias=False, LayerNorm_type='WithBias')

    def forward(self, y_vi_image, ir_image):
        activate = nn.GELU()
        activate = activate.cuda()
        vi_out = activate(self.vi_conv1(y_vi_image))
        ir_out = activate(self.ir_conv1(ir_image))

        vi_out, ir_out = self.transformer_block1(vi_out, ir_out)

        vi_out = activate(self.vi_conv2(vi_out))
        ir_out = activate(self.ir_conv2(ir_out))

        vi_out = activate(self.vi_conv3(vi_out))
        ir_out = activate(self.ir_conv3(ir_out))


        return vi_out, ir_out
    

class Decoder_Transformer(nn.Module):
    def __init__(self):
        super(Decoder_Transformer, self).__init__()
        self.transformer_block = TransformerBlock2(dim=64, num_heads=8, ffn_expansion_factor=4, bias=False,
                                                  LayerNorm_type='WithBias')
        self.transformer_block = self.transformer_block.cuda()
        self.conv1 = nn.Conv2d(in_channels=64, kernel_size=1, out_channels=32, stride=1, padding=0)

    def forward(self, x):
        activate = nn.GELU()
        activate = activate.cuda()

        x = activate(self.transformer_block(x))


        x = nn.Tanh()(self.conv1(x)) / 2 + 0.5  # 将范围从[-1,1]转换为[0,1]
        return x

class Decoder_CNN(nn.Module):
    def __init__(self):
        super(Decoder_CNN, self).__init__()

        self.conv1 = reflect_conv(in_channels=64, kernel_size=1, out_channels=32, stride=1, pad=0)
        self.conv2 = nn.Conv2d(in_channels=32, kernel_size=1, out_channels=32, stride=1, padding=0)

    def forward(self, x):
        activate = nn.GELU()
        activate = activate.cuda()
        x = activate(self.conv1(x))
        x = nn.Tanh()(self.conv2(x)) / 2 + 0.5
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv1 = reflect_conv(in_channels=32, kernel_size=3, out_channels=16, stride=1, pad=1)  # 减半输出通道数
        self.conv2 = reflect_conv(in_channels=16, kernel_size=3, out_channels=16, stride=1, pad=1)  # 减半输出通道数
        self.conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=8, stride=1, pad=1)  # 减半输出通道数
        self.transformer_block3 = TransformerBlock2(dim=8, num_heads=8, ffn_expansion_factor=4, bias=False,
                                                  LayerNorm_type='WithBias')  # 减半num_heads
        self.transformer_block3 = self.transformer_block3.cuda()
        self.conv4 = nn.Conv2d(in_channels=8, kernel_size=1, out_channels=1, stride=1, padding=0)

    def forward(self, x):
        activate = nn.GELU()
        activate = activate.cuda()
        x = activate(self.conv1(x))
        x = activate(self.conv2(x))
        x = activate(self.conv3(x))
        x = activate(self.transformer_block3(x))
        x = nn.Tanh()(self.conv4(x)) / 2 + 0.5
        return x

class MBHFuse(nn.Module):
    def __init__(self,):
        super(MBHFuse, self).__init__()
        self.low_freq_encoder = LowFreqEncoder()
        self.high_freq_encoder = HighFreqEncoder()
        self.low_freq_fusion = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.high_freq_fusion = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.fusion = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
        self.decoder = Decoder()
        self.decoder_cnn = Decoder_CNN()
        self.decoder_transformer = Decoder_Transformer()

    def forward(self, y_vi_image, ir_image):

        vi_low_freq_encoded, ir_low_freq_encoded = self.low_freq_encoder(y_vi_image, ir_image)
        vi_high_freq_encoded, ir_high_freq_encoded = self.high_freq_encoder(y_vi_image, ir_image)
        
        low_freq_fused = self.low_freq_fusion(Fusion(vi_low_freq_encoded, ir_low_freq_encoded))
        high_freq_fused = self.high_freq_fusion(Fusion(vi_high_freq_encoded, ir_high_freq_encoded))

        low_freq_fused1 = self.decoder_transformer(low_freq_fused)
        high_freq_fused1 = self.decoder_cnn(high_freq_fused)
        fused_output = self.fusion(Fusion(low_freq_fused1, high_freq_fused1))

        fused_image = self.decoder(fused_output)

        return fused_image
    
#    # #左右均是low消融 
# class MBHFuse(nn.Module):
#      def __init__(self,):
#         super(MBHFuse, self).__init__()
#         self.low_freq_encoder = LowFreqEncoder()

#         self.high_freq_encoder = HighFreqEncoder()
#         self.low_freq_fusion = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
#         self.high_freq_fusion = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
#         self.fusion = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
#         self.decoder = Decoder()
#         self.decoder_cnn = Decoder_CNN()
#         self.decoder_transformer = Decoder_Transformer()

#      def forward(self, y_vi_image, ir_image):

#         vi_low_freq_encoded, ir_low_freq_encoded = self.low_freq_encoder(y_vi_image, ir_image)
#         vi_high_freq_encoded, ir_high_freq_encoded = self.low_freq_encoder(y_vi_image, ir_image)
        
#         low_freq_fused = self.low_freq_fusion(Fusion(vi_low_freq_encoded, ir_low_freq_encoded))
#         high_freq_fused = self.low_freq_fusion(Fusion(vi_high_freq_encoded, ir_high_freq_encoded))

#         low_freq_fused1 = self.decoder_transformer(low_freq_fused)
#         high_freq_fused1 = self.decoder_cnn(high_freq_fused)
#         fused_output = self.fusion(Fusion(low_freq_fused1, high_freq_fused1))

#         fused_image = self.decoder(fused_output)

#         return fused_image

#   #  #左右均是high消融 
# class MBHFuse(nn.Module):
#      def __init__(self,):
#         super(MBHFuse, self).__init__()
#         self.low_freq_encoder = LowFreqEncoder()
#         self.high_freq_encoder = HighFreqEncoder()
#         self.low_freq_fusion = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
#         self.high_freq_fusion = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
#         self.fusion = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
#         self.decoder = Decoder()
#         self.decoder_cnn = Decoder_CNN()
#         self.decoder_transformer = Decoder_Transformer()

#      def forward(self, y_vi_image, ir_image):

#         vi_low_freq_encoded, ir_low_freq_encoded = self.high_freq_encoder(y_vi_image, ir_image)
#         vi_high_freq_encoded, ir_high_freq_encoded = self.high_freq_encoder(y_vi_image, ir_image)
        
#         low_freq_fused = self.high_freq_fusion(Fusion(vi_low_freq_encoded, ir_low_freq_encoded))
#         high_freq_fused = self.high_freq_fusion(Fusion(vi_high_freq_encoded, ir_high_freq_encoded))

#         low_freq_fused1 = self.decoder_transformer(low_freq_fused)
#         high_freq_fused1 = self.decoder_cnn(high_freq_fused)
#         fused_output = self.fusion(Fusion(low_freq_fused1, high_freq_fused1))

#         fused_image = self.decoder(fused_output)

#         return fused_image


#   #  #无GCA消融 
# class MBHFuse(nn.Module):
#      def __init__(self,):
#         super(MBHFuse, self).__init__()
#         self.low_freq_encoder = wo_LowEncoder()

#         self.high_freq_encoder = HighFreqEncoder()
#         self.low_freq_fusion = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
#         self.high_freq_fusion = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
#         self.fusion = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
#         self.decoder = Decoder()
#         self.decoder_cnn = Decoder_CNN()
#         self.decoder_transformer = Decoder_Transformer()

#      def forward(self, y_vi_image, ir_image):

#         vi_low_freq_encoded, ir_low_freq_encoded = self.low_freq_encoder(y_vi_image, ir_image)
#         vi_high_freq_encoded, ir_high_freq_encoded = self.high_freq_encoder(y_vi_image, ir_image)
        
#         low_freq_fused = self.low_freq_fusion(Fusion(vi_low_freq_encoded, ir_low_freq_encoded))
#         high_freq_fused = self.high_freq_fusion(Fusion(vi_high_freq_encoded, ir_high_freq_encoded))

#         low_freq_fused1 = self.decoder_transformer(low_freq_fused)
#         high_freq_fused1 = self.decoder_cnn(high_freq_fused)
#         fused_output = self.fusion(Fusion(low_freq_fused1, high_freq_fused1))

#         fused_image = self.decoder(fused_output)

#         return fused_image


#  #  #无LFA消融 
# class MBHFuse(nn.Module):
#      def __init__(self,):
#         super(MBHFuse, self).__init__()
#         self.low_freq_encoder = LowFreqEncoder()
#         self.high_freq_encoder = wo_HighEncoder()
#         self.low_freq_fusion = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
#         self.high_freq_fusion = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
#         self.fusion = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
#         self.decoder = Decoder()
#         self.decoder_cnn = Decoder_CNN()
#         self.decoder_transformer = Decoder_Transformer()

#      def forward(self, y_vi_image, ir_image):

#         vi_low_freq_encoded, ir_low_freq_encoded = self.low_freq_encoder(y_vi_image, ir_image)
#         vi_high_freq_encoded, ir_high_freq_encoded = self.high_freq_encoder(y_vi_image, ir_image)
        
#         low_freq_fused = self.low_freq_fusion(Fusion(vi_low_freq_encoded, ir_low_freq_encoded))
#         high_freq_fused = self.high_freq_fusion(Fusion(vi_high_freq_encoded, ir_high_freq_encoded))

#         low_freq_fused1 = self.decoder_transformer(low_freq_fused)
#         high_freq_fused1 = self.decoder_cnn(high_freq_fused)
#         fused_output = self.fusion(Fusion(low_freq_fused1, high_freq_fused1))

#         fused_image = self.decoder(fused_output)

#         return fused_image


#  #  #LFA,GCA全无消融 
# class MBHFuse(nn.Module):
#      def __init__(self,):
#         super(MBHFuse, self).__init__()
#         self.low_freq_encoder = wo_LowEncoder()

#         self.high_freq_encoder = wo_HighEncoder()
#         self.low_freq_fusion = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
#         self.high_freq_fusion = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
#         self.fusion = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
#         self.decoder = Decoder()
#         self.decoder_cnn = Decoder_CNN()
#         self.decoder_transformer = Decoder_Transformer()

#      def forward(self, y_vi_image, ir_image):

#         vi_low_freq_encoded, ir_low_freq_encoded = self.low_freq_encoder(y_vi_image, ir_image)
#         vi_high_freq_encoded, ir_high_freq_encoded = self.high_freq_encoder(y_vi_image, ir_image)
        
#         low_freq_fused = self.low_freq_fusion(Fusion(vi_low_freq_encoded, ir_low_freq_encoded))
#         high_freq_fused = self.high_freq_fusion(Fusion(vi_high_freq_encoded, ir_high_freq_encoded))

#         low_freq_fused1 = self.decoder_transformer(low_freq_fused)
#         high_freq_fused1 = self.decoder_cnn(high_freq_fused)
#         fused_output = self.fusion(Fusion(low_freq_fused1, high_freq_fused1))

#         fused_image = self.decoder(fused_output)

#         return fused_image