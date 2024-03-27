import abc


class ModelStrategy(abc.ABC):
    """Interface containing the methods used to declare a model, train it, and validate it"""
    @abc.abstractmethod
    def defineModel(self, device, data, nbands, windowSize, classes, train_y, pca, pls):
        pass

    @abc.abstractmethod
    def trainFoldStrategy(self, trainx, train_y, train, batch_size, classes, device,
                          epochs, valx, test, means, stds, filepath, printProcess):
        pass

    @abc.abstractmethod
    def loadModelStrategy(self, path):
        pass

    @abc.abstractmethod
    def evaluateFoldStrategy(self, valx, train_y, test, means, stds, batch_size, classes, device):
        pass
