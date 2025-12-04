import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
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

    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, attn = self.attention(q, k, v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers=2, d_model=256, d_inner=512, n_head=4, d_k=64, d_v=64, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout) for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
        return x


class Enhance(nn.Module):
    def __init__(self, in_dim=256):
        super(Enhance, self).__init__()
        self.net = TransformerEncoder(n_layers=2, d_model=256, d_inner=512, n_head=4, d_k=64, d_v=64, dropout=0.1)

        self.se_mask = nn.Sequential(
            nn.Linear(1, in_dim),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x, expos, time):
        index = 0 if time else 2
        expos_mask = self.se_mask(expos[:, index].unsqueeze(1))
        token1 = x.unsqueeze(1)  # -> (B, 1, 256)
        token2 = expos_mask.unsqueeze(1)  # -> (B, 1, 256)
        x = torch.cat([token1, token2], dim=1)  # -> (B, 2, 256)
        return self.net(x)[:, 0, :]

## transformer_no time
class Enhance1(nn.Module):
    def __init__(self, in_dim=256):
        super(Enhance1, self).__init__()
        self.net = TransformerEncoder(n_layers=2, d_model=256, d_inner=512, n_head=4, d_k=64, d_v=64, dropout=0.1)

    def forward(self, x, expos, time):
        token1 = x.unsqueeze(1)  # -> (B, 1, 256)
        return self.net(token1)[:, 0, :]


class Enhance3(nn.Module):
    """
    tf_time:先验向量与时间向量融合之后，形成1token的tf输入
            融合方式：向量加法融合
    """
    def __init__(self, in_dim=256):
        super(Enhance3, self).__init__()
        self.net = TransformerEncoder(n_layers=2, d_model=256, d_inner=512, n_head=4, d_k=64, d_v=64, dropout=0.1)

        self.se_mask = nn.Sequential(
            nn.Linear(1, in_dim),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x, expos, time):
        index = 0 if time else 2
        expos_mask = self.se_mask(expos[:, index].unsqueeze(1))
        x = x + expos_mask  # -> (B, 256) + (B, 256)
        x = x.unsqueeze(1)  # -> (B, 1, 256)
        return self.net(x)[:, 0, :]


class Enhance4(nn.Module):
    """
    tf_time:先验向量与时间向量融合之后，形成1token的tf输入
            融合方式：向量乘法融合
    """
    def __init__(self, in_dim=256):
        super(Enhance4, self).__init__()
        self.net = TransformerEncoder(n_layers=2, d_model=256, d_inner=512, n_head=4, d_k=64, d_v=64, dropout=0.1)

        self.se_mask = nn.Sequential(
            nn.Linear(1, in_dim),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x, expos, time):
        index = 0 if time else 2
        expos_mask = self.se_mask(expos[:, index].unsqueeze(1))
        x = x * expos_mask  # -> (B, 256) * (B, 256)
        x = x.unsqueeze(1)  # -> (B, 1, 256)
        return self.net(x)[:, 0, :]

class Enhance5(nn.Module):
    """
    tf_time:先验向量与时间向量融合之后，形成1token的tf输入
            融合方式：通道维度相加后经过一个线性层恢复到256维
    """
    def __init__(self, in_dim=256):
        super(Enhance5, self).__init__()
        self.net = TransformerEncoder(n_layers=2, d_model=256, d_inner=512, n_head=4, d_k=64, d_v=64, dropout=0.1)

        self.se_mask = nn.Sequential(
            nn.Linear(1, in_dim),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
        self.fuse_proj = nn.Linear(in_dim*2, in_dim)

    def forward(self, x, expos, time):
        index = 0 if time else 2
        expos_mask = self.se_mask(expos[:, index].unsqueeze(1))
        x = self.fuse_proj(torch.cat([x, expos_mask], dim=1))
        x = x.unsqueeze(1)  # -> (B, 1, 256)
        return self.net(x)[:, 0, :]


class Enhance6(nn.Module):
    """
    tf_time:先验向量与时间向量融合之后，形成1token的tf输入
            融合方式：通道维度相加后经过一个mlp恢复到256维
    """
    def __init__(self, in_dim=256):
        super(Enhance6, self).__init__()
        self.net = TransformerEncoder(n_layers=2, d_model=256, d_inner=512, n_head=4, d_k=64, d_v=64, dropout=0.1)

        self.se_mask = nn.Sequential(
            nn.Linear(1, in_dim),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
        self.fuse_proj = nn.Sequential(nn.Linear(in_dim*2, in_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(in_dim, in_dim)
                                       )

    def forward(self, x, expos, time):
        index = 0 if time else 2
        expos_mask = self.se_mask(expos[:, index].unsqueeze(1))
        x = self.fuse_proj(torch.cat([x, expos_mask], dim=1))
        x = x.unsqueeze(1)  # -> (B, 1, 256)
        return self.net(x)[:, 0, :]


class Enhance7(nn.Module):
    """
    tf_time:先验向量与时间向量融合之后，形成1token的tf输入
            融合方式：通道维度相加后经过一个mlp恢复到256维,之后使用门控添加权重
    """
    def __init__(self, in_dim=256):
        super(Enhance7, self).__init__()
        self.net = TransformerEncoder(n_layers=2, d_model=256, d_inner=512, n_head=4, d_k=64, d_v=64, dropout=0.1)

        self.se_mask = nn.Sequential(
            nn.Linear(1, in_dim),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
        self.fuse_proj = nn.Sequential(nn.Linear(in_dim*2, in_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(in_dim, in_dim)
                                       )

    def forward(self, x, expos, time):
        index = 0 if time else 2
        expos_mask = self.se_mask(expos[:, index].unsqueeze(1))
        gate = torch.sigmoid(self.fuse_proj(torch.cat([x, expos_mask], dim=1)))
        x = gate * x + (1 - gate) * expos_mask
        x = x.unsqueeze(1)  # -> (B, 1, 256)
        return self.net(x)[:, 0, :]


if __name__ == "__main__":
    batch_size = 4
    d_model = 256

    x = torch.randn(batch_size,d_model)  # Dummy input
    expose = torch.randn(batch_size, 3)

    model = Enhance(in_dim=256)
    out = model(x,expose,True)

    print("Transformer output shape:", out.shape)  # should be [4, 2, 256]
    print("Output tokens:\n", out)
