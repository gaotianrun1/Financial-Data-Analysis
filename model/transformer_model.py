import torch
import torch.nn as nn
import torch.nn.init as init
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, num_layers=2, output_size=1, 
                 dropout=0.2, num_attention_heads=8):
        super(TransformerModel, self).__init__()
        
        self.hidden_layer_size = hidden_layer_size
        self.num_attention_heads = num_attention_heads
        
        # 输入
        self.input_projection = nn.Linear(input_size, self.hidden_layer_size)

        self.pos_encoder = PositionalEncoding(self.hidden_layer_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_layer_size,
            nhead=self.num_attention_heads,
            dim_feedforward=self.hidden_layer_size * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu' 
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 输出
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(self.hidden_layer_size, output_size)
        
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                init.constant_(module.weight, 1.0)
                init.constant_(module.bias, 0)
            elif isinstance(module, nn.MultiheadAttention):
                if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                    init.xavier_uniform_(module.in_proj_weight)
                if hasattr(module, 'out_proj') and module.out_proj.weight is not None:
                    init.xavier_uniform_(module.out_proj.weight)
                if hasattr(module, 'in_proj_bias') and module.in_proj_bias is not None:
                    init.constant_(module.in_proj_bias, 0)
                if hasattr(module, 'out_proj') and module.out_proj.bias is not None:
                    init.constant_(module.out_proj.bias, 0)

    def forward(self, x):
        x = self.input_projection(x) # [batch_size, seq_len, input_size] -> [batch_size, seq_len, hidden_size]

        x = x.transpose(0, 1) # [batch_size, seq_len, hidden_size] -> [seq_len, batch_size, hidden_size]
        x = self.pos_encoder(x) # 添加位置编码
        x = x.transpose(0, 1) # 转换回batch_first格式
        
        transformer_output = self.transformer_encoder(x)
        
        last_output = transformer_output[:, -1, :] # [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
        
        last_output = self.dropout(last_output)
        predictions = self.output_projection(last_output)

        return predictions.squeeze(-1) # [batch_size, 1] -> [batch_size] 返回单个预测值
