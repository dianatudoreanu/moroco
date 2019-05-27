from abc import ABC, abstractmethod


class AbstractNeuralNetwork(ABC):

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass