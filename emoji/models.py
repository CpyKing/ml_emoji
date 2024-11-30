from modules import FPN101
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        
        q = self.w_qs(q).view(sz_b, n_head, d_k)
        k = self.w_ks(k).view(sz_b, n_head, d_k)
        v = self.w_vs(v).view(sz_b, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

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
        
        self.mul_attn_1 = MultiHeadAttention(n_head=2, d_model=1176, d_k=1176, d_v=1176)
        self.add_norm_1 = AddNorm(1176, 0.5)
        self.mul_attn_2 = MultiHeadAttention(n_head=2, d_model=1176, d_k=1176, d_v=1176)
        self.add_norm_2 = AddNorm(1176, 0.5)

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

        t1 = self.compress_1(t1)
        t2 = self.compress_2(t2)
        t3 = self.compress_3(t3)
        t4 = self.compress_4(t4)
        # print(t1.shape, t2.shape, t3.shape, t4.shape)
  
        t1 = t1.view(t1.shape[0], 4, 15*15)   
        t2 = t2.view(t2.shape[0], 4, 7*7)
        t3 = t3.view(t3.shape[0], 4, 4*4)
        t4 = t4.view(t4.shape[0], 4, 2*2)
        tot_tensor = torch.cat((t1,t2,t3,t4), dim=-1)
        tot_tensor = tot_tensor.view(tot_tensor.shape[0], tot_tensor.shape[1] * tot_tensor.shape[2])
        # print(tot_tensor.shape)

        out,_ = self.mul_attn_1(tot_tensor,tot_tensor,tot_tensor)
        # print(out.shape)
        out = self.add_norm_1(out, out)
        # print(out.shape)
        out,_ = self.mul_attn_2(out,out,out)
        out = self.add_norm_2(out, out)

        out = self.prediect_layer(out)
        
        # print(out)

        return out

if __name__ == '__main__':
    mo = MyModel()
    mo(Variable(torch.randn(7,3,120,120)))
