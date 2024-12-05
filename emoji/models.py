from modules import FPN101, MultiHeadAttention
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable



class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fpn_net = FPN101()
        self.compress_1 = self.compress_layer()
        self.compress_2 = self.compress_layer()
        self.compress_3 = self.compress_layer()
        self.compress_4 = self.compress_layer()
        
        # self.mul_attn_1 = MultiHeadAttention(n_head=2, d_model=1176, d_k=1176, d_v=1176)
        # self.add_norm_1 = AddNorm(1176, 0.5)
        # self.mul_attn_2 = MultiHeadAttention(n_head=2, d_model=1176, d_k=1176, d_v=1176)
        # self.add_norm_2 = AddNorm(1176, 0.5)

        self.attn_1 = MultiHeadAttention(n_head=4, d_q=900, d_k=900, d_v=900, d_hid=900)
        self.attn_2 = MultiHeadAttention(n_head=5, d_q=225, d_k=225, d_v=225, d_hid=225)
        self.attn_3 = MultiHeadAttention(n_head=2, d_q=64, d_k=64, d_v=64, d_hid=64)
        self.attn_4 = MultiHeadAttention(n_head=2, d_q=16, d_k=16, d_v=16, d_hid=16)

        self.prediect_layer = self.gen_prediect_layer()
    
    def compress_layer(self):
        compress_1 = nn.Sequential()
        compress_1.append(nn.Conv2d(256, 128, kernel_size=1))
        compress_1.append(nn.Conv2d(128, 32, kernel_size=1))
        compress_1.append(nn.Conv2d(32, 4, kernel_size=1))
        compress_1.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return compress_1
    
    def gen_prediect_layer(self):
        layers = nn.Sequential()
        layers.append(nn.Linear(1176, 512, bias=False))
        layers.append(nn.Dropout(p=0.5))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(512, 128, bias=False))
        layers.append(nn.Dropout(p=0.5))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 50, bias=False))
        return layers

    def forward(self, x):
        t1, t2, t3, t4 = self.fpn_net(x)
        # print(t1.shape, t2.shape, t3.shape, t4.shape)
        
  
        t1 = t1.view(t1.shape[0], 256, 900)   
        t2 = t2.view(t2.shape[0], 256, 225)
        t3 = t3.view(t3.shape[0], 256, 64)
        t4 = t4.view(t4.shape[0], 256, 16)

        t1,_ = self.attn_1(t1, t1, t1)
        t2,_ = self.attn_2(t2, t2, t2)
        t3,_ = self.attn_3(t3, t3, t3)
        t4,_ = self.attn_4(t4, t4, t4)
        # print(t1.shape, t2.shape, t3.shape, t4.shape)
        
        t1 = t1.view(t1.shape[0], 256, 30, 30)   
        t2 = t2.view(t2.shape[0], 256, 15, 15)
        t3 = t3.view(t3.shape[0], 256, 8, 8)
        t4 = t4.view(t4.shape[0], 256, 4, 4)

        t1 = self.compress_1(t1)
        t2 = self.compress_2(t2)
        t3 = self.compress_3(t3)
        t4 = self.compress_4(t4)

        t1 = t1.view(t1.shape[0], t1.shape[1], -1)   
        t2 = t2.view(t2.shape[0], t1.shape[1], -1)
        t3 = t3.view(t3.shape[0], t1.shape[1], -1)
        t4 = t4.view(t4.shape[0], t1.shape[1], -1)
        tot_tensor = torch.cat((t1,t2,t3,t4), dim=-1)
        tot_tensor = tot_tensor.view(tot_tensor.shape[0], tot_tensor.shape[1] * tot_tensor.shape[2])        

        out = self.prediect_layer(tot_tensor)
        
        # print(out.shape)

        return out

    def forward_du(self, x):
        t1, t2, t3, t4 = self.fpn_net(x)
        print(t1.shape, t2.shape, t3.shape, t4.shape)
        t1 = self.compress_1(t1)
        t2 = self.compress_2(t2)
        t3 = self.compress_3(t3)
        t4 = self.compress_4(t4)
        print(t1.shape, t2.shape, t3.shape, t4.shape)
  
        t1 = t1.view(t1.shape[0], 4, 15*15)   
        t2 = t2.view(t2.shape[0], 4, 7*7)
        t3 = t3.view(t3.shape[0], 4, 4*4)
        t4 = t4.view(t4.shape[0], 4, 2*2)
        tot_tensor = torch.cat((t1,t2,t3,t4), dim=-1)
        tot_tensor = tot_tensor.view(tot_tensor.shape[0], tot_tensor.shape[1] * tot_tensor.shape[2])
        # print(tot_tensor.shape)

        # out,_ = self.mul_attn_1(tot_tensor,tot_tensor,tot_tensor)
        # # print(out.shape)
        # out = self.add_norm_1(out, out)
        # # print(out.shape)
        # out,_ = self.mul_attn_2(out,out,out)
        # out = self.add_norm_2(out, out)

        out = self.prediect_layer(tot_tensor)
        
        # print(out)

        return out

if __name__ == '__main__':
    mo = MyModel()
    mo(Variable(torch.randn(7,3,120,120)))
