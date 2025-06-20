from .lstm_model import LSTMModel
from .transformer_model import TransformerModel

def create_model(config):
    model_type = config["model"].get("model_type", "lstm").lower()
    
    if model_type == "lstm":
        return create_lstm_model(config)
    elif model_type == "transformer":
        return create_transformer_model(config)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def create_lstm_model(config):
    model_config = config["model"]
    
    model = LSTMModel(
        input_size=model_config["input_size"],
        hidden_layer_size=model_config["lstm_size"],
        num_layers=model_config["num_lstm_layers"],
        output_size=model_config.get("output_size", 1),
        dropout=model_config["dropout"]
    )
    
    print(f"创建LSTM模型: 输入维度={model_config['input_size']}, "
          f"隐藏层大小={model_config['lstm_size']}, "
          f"LSTM层数={model_config['num_lstm_layers']}, "
          f"dropout={model_config['dropout']}")
    
    return model

def create_transformer_model(config):
    model_config = config["model"]
    
    # 需要确保hidden_size能被attention_heads整除
    hidden_size = model_config["transformer_hidden_size"]
    attention_heads = model_config["num_attention_heads"]
    
    model = TransformerModel(
        input_size=model_config["input_size"],
        hidden_layer_size=hidden_size,
        num_layers=model_config["num_transformer_layers"],
        output_size=model_config.get("output_size", 1),
        dropout=model_config.get("transformer_dropout", model_config["dropout"]),
        num_attention_heads=attention_heads
    )
    
    print(f"创建Transformer模型: 输入维度={model_config['input_size']}, "
          f"隐藏层大小={hidden_size}, "
          f"Transformer层数={model_config['num_transformer_layers']}, "
          f"注意力头数={attention_heads}, "
          f"dropout={model_config.get('transformer_dropout', model_config['dropout'])}")
    
    return model
