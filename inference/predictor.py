import numpy as np
import torch
from torch.utils.data import DataLoader

def predict_on_dataset(model, dataset, config):
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    model.eval()

    predicted = np.array([])

    for idx, (x, y) in enumerate(dataloader):
        x = x.to(config["training"]["device"])
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted = np.concatenate((predicted, out))

    return predicted

def predict_next_day(model, data_x_unseen, config):
    model.eval()

    # 检查输入数据的维度
    if len(data_x_unseen.shape) == 2:
        # 如果是2D数据，需要添加batch维度: (seq_len, features) -> (1, seq_len, features)
        x = torch.tensor(data_x_unseen).float().to(config["training"]["device"]).unsqueeze(0)
    elif len(data_x_unseen.shape) == 1:
        # 如果是1D数据，添加batch和feature维度: (seq_len,) -> (1, seq_len, 1)
        x = torch.tensor(data_x_unseen).float().to(config["training"]["device"]).unsqueeze(0).unsqueeze(2)
    else:
        # 如果已经是3D数据，直接转换
        x = torch.tensor(data_x_unseen).float().to(config["training"]["device"])
        if x.shape[0] != 1:  # 如果不是batch=1，添加batch维度
            x = x.unsqueeze(0)
    
    prediction = model(x)
    prediction = prediction.cpu().detach().numpy()
    
    return prediction 