class PretrainConfig(object):
    def __init__(self):
        # Hyperparameters
        self.lr = 1e-4
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epochs = 200
        self.batch_size = 64
        # Environment
        self.seed = 7


class FinetuneConfig(object):
    def __init__(self):

        self.lr = 1e-4
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epochs = 100
        self.batch_size = 64
        # Environment
        self.seed = 7