import numpy as np

class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        # 避免除零错误
        self.sd = np.where(self.sd == 0, 1.0, self.sd)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x
    
    def transform(self, x):
        if self.mu is None or self.sd is None:
            raise ValueError("标准化器未训练，请先调用fit_transform方法")
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        if self.mu is None or self.sd is None:
            raise ValueError("标准化器未训练，请先调用fit_transform方法")
        return (x*self.sd) + self.mu 