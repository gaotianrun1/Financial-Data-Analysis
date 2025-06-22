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

class TimeAwareAttention(nn.Module):
    """时间感知注意力机制"""
    def __init__(self, d_model, num_heads=8, time_decay=0.1):
        super(TimeAwareAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.time_decay = nn.Parameter(torch.tensor(time_decay))
        self.multihead_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # 创建时间感知的注意力偏置矩阵
        attn_bias = torch.zeros(seq_len, seq_len, device=x.device)
        for i in range(seq_len):
            for j in range(seq_len):
                # 对更近的时间点给予更高的权重
                distance = abs(i - j)
                attn_bias[i, j] = -torch.abs(self.time_decay) * distance
        
        attn_output, attn_weights = self.multihead_attn(
            x, x, x, 
            attn_mask=attn_bias,
            need_weights=True
        )
        
        output = self.layer_norm(x + attn_output)
        
        return output, attn_weights

class TransformerModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, num_layers=2, output_size=1, 
                 dropout=0.2, num_attention_heads=8, use_time_aware=False, time_decay=0.1):
        super(TransformerModel, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.num_attention_heads = num_attention_heads
        
        self.use_time_aware = use_time_aware
        self.time_decay = time_decay

        # 输入
        self.input_projection = nn.Linear(input_size, self.hidden_layer_size)

        self.pos_encoder = PositionalEncoding(self.hidden_layer_size)
        
        # 标准Transformer编码器层
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
        
        # 时间感知注意力层
        if self.use_time_aware:
            self.time_aware_attention = TimeAwareAttention(
                self.hidden_layer_size, 
                self.num_attention_heads,
                time_decay=time_decay
            )
            print(f"启用时间感知注意力机制，时间衰减参数: {time_decay}")
        
        # 输出层
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
        
        # 时间感知注意力处理
        if self.use_time_aware:
            transformer_output, attn_weights = self.time_aware_attention(transformer_output)

        last_output = transformer_output[:, -1, :] # [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
        
        last_output = self.dropout(last_output)
        predictions = self.output_projection(last_output)

        return predictions.squeeze(-1) # [batch_size, 1] -> [batch_size] 返回单个预测值
