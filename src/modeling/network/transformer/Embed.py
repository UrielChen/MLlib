import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):

    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Note:
        缺少输入数据的编码？
    """

    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        # d_model: 词嵌入维度; vocab_size: 词汇表大小(在这里指day_len); dropout: dropout比例
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)

        # 建立一个0-vocab_size-1的索引，unsqueeze(1)可以将position在列方向扩展一个维度
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() # paper中的2i序列
            * (-math.log(10000.0) / d_model) # -ln(10000.0)/d_model
        ) # 1/(10000^(2i/d_model))

        pe[:, 0::2] = torch.sin(position * div_term) # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数位置
        
        pe = pe.unsqueeze(0) # 增加一个维度，方便在batch维度上扩展 (vocab_size, d_model) -> (1, vocab_size, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = self.pe[:, :x.size(1), :] # 取pe中对应位置的词嵌入
        x = self.dropout(x)
        return x

class TokenEncoding(nn.Module):
    def __init__(self, c_in, d_model):
        # c_in: 输入的维度; d_model: 词嵌入维度
        super(TokenEncoding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=padding, padding_mode='circular')
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    
    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class DataEncoding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEncoding, self).__init__()

        self.value_embedding = TokenEncoding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEncoding(d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        a = self.value_embedding(x)
        b = self.position_embedding(x)
        x = a + b
        return self.dropout(x)