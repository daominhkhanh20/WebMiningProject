

class BaseTrainer(object):
    def __init__(self, **kwargs):
        raise NotImplemented()

    def evaluate(self, **kwargs):
        raise NotImplemented()

    def train_one_epoch(self, **kwargs):
        raise NotImplemented()

    def fit(self, **kwargs):
        raise NotImplemented()