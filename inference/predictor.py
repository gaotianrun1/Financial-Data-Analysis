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
