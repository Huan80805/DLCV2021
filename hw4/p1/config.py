class TrainConfig(object):
    def __init__(self):

        # Hyperparameters
        self.lr = 1e-4
        self.seed = 7
        self.beta1 = 0.9
        self.beta2 = 0.999

        # Training
        self.n_way = 30
        self.n_shot = 1
        self.n_query = 15
class TestConfig(object):
    def __init__(self):
    # Testing
        self.n_way = 5
        self.n_shot = 1
        self.n_query = 15