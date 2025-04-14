import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, dropout, activation):
        super(FeedForward, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=ff_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        x = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        x = self.dropout(self.conv2(x).transpose(-1, 1))
        return x

class MultiheadFeedForward(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout, activation):
        super(MultiheadFeedForward, self).__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.mhfw = nn.ModuleList([
            FeedForward(self.head_dim, ff_dim, dropout, activation)
            for _ in range(n_heads)
        ])
    
    def forward(self, x): # (bs, seq_len, d_model)
        bs = x.shape[0]
        input = x.reshape((bs, -1, self.n_heads, self.head_dim)) # (bs, seq_len, n_heads, head_dim)
        outputs = []
        
        for i in range(self.n_heads):
            output = self.mhfw[i](input[:, :, i, :])
            outputs.append(output) # (bs, seq_len, head_dim)
        
        outputs = torch.stack(outputs, dim=-2).reshape((bs, -1, self.d_model)) # (bs, seq_len, d_model)
        
        return outputs
    
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, ff_dim, n_heads=8, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()
        ff_dim = ff_dim or 4 * d_model
        self.attention = attention
        self.mhfw = MultiheadFeedForward(d_model, n_heads, ff_dim, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        residual, attention = self.attention(x, x, x, attention_mask)
        x = x + self.dropout(residual)
        x = self.norm1(x)

        new_residual = self.mhfw(x)
        y = x + new_residual
        y = self.norm2(y)
        return y, attention

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, ff_dim, n_heads=8, dropout=0.1, activation='relu'):
        super(DecoderLayer, self).__init__()
        ff_dim = ff_dim or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.mhfw = MultiheadFeedForward(d_model, n_heads, ff_dim, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        residual, _ = self.self_attention(x, x, x, x_mask)
        x = x + self.dropout(residual)
        x = self.norm1(x)

        residual, _ = self.cross_attention(x, cross, cross, cross_mask)
        x = x + self.dropout(residual)
        x = self.norm2(x)

        new_residual = self.mhfw(x)
        y = x + new_residual
        return self.norm3(y)
    
class Encoder(nn.Module):
    def __init__(self, attention_layers: list, conv_layers: list = None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attention_layers = nn.ModuleList(attention_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm_layer = norm_layer
    
    def forward(self, x, attn_mask=None):
        attentions = []
        if self.conv_layers is not None:
            for attention_layer, conv_layer in zip(self.attention_layers, self.conv_layers):
                x, attention = attention_layer(x, attn_mask)
                x = conv_layer(x)
                attentions.append(attention)
        else:
            for attention_layer in self.attention_layers:
                x, attention = attention_layer(x, attn_mask)
                attentions.append(attention)
        
        if self.norm_layer is not None:
            x = self.norm_layer(x)

        return x, attentions

class Decoder(nn.Module):
    def __init__(self, layers: list, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm_layer = norm_layer
        self.projection = projection
        
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask, cross_mask)
        
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        
        if self.projection is not None:
            x = self.projection(x)
        
        return x