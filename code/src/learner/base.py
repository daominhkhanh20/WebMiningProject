from abc import abstractmethod


class BaseLeaner(object):
    @abstractmethod
    def save(self, **kwargs):
        raise NotImplemented()

    @abstractmethod
    def evaluate(self, **kwargs):
        raise NotImplemented()

    @abstractmethod
    def train_one_epoch(self, **kwargs):
        raise NotImplemented()

    @abstractmethod
    def fit(self, **kwargs):
        raise NotImplemented()
