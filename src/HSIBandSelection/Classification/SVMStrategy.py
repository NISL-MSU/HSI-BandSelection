from ..Classification.ModelStrategy import ModelStrategy
from sklearn import svm
import numpy as np
from ..utils import applynormalize
import pickle


class SVMStrategy(ModelStrategy):

    def __init__(self):
        self.model = None

    def defineModel(self, device, data, nbands, windowSize, classes, train_y, pca, pls):
        """Override model declaration method"""
        return svm.SVC(C=1000, gamma='auto', max_iter=5000000, random_state=7)

    def trainFoldStrategy(self, trainx, train_y, train, batch_size, classes, device,
                          epochs, valx, test, means, stds, filepath, printProcess=True):
        # Permute and reshape the data
        X = trainx.copy()[:, 0, :, :, :]
        X = X.transpose((1, 0, 2, 3))
        X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
        X = X.transpose((1, 0))

        # Get augmented target vector
        Y = []
        for i in range(len(trainx)):
            yt = train_y[train][i]
            for _ in range(int(len(X) / len(trainx))):
                Y.append(yt)
        Y = np.array(Y)

        # Shuffle
        np.random.seed(7)  # Initialize seed to get reproducible results
        ind = [i for i in range(X.shape[0])]
        np.random.shuffle(ind)
        X = X[ind][0:8000, :]
        Y = Y[ind][0:8000]

        # Train model
        self.model.fit(X, Y)

        # Save model
        with open(filepath, 'wb') as fi:
            pickle.dump(self.model, fi)

    def evaluateFoldStrategy(self, valx, train_y, test, means, stds, batch_size, classes, device):
        # Normalize the validation set based on the previous statistics
        valxn = applynormalize(valx, means, stds)
        # Permute and reshape the data
        X = valxn[:, 0, :, :, :]
        X = X.transpose((1, 0, 2, 3))
        X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
        X = X.transpose((1, 0))

        # Get augmented target vector
        Y = []
        for i in range(len(test)):
            yt = train_y[test][i]
            for _ in range(int(len(X) / len(test))):
                Y.append(yt)
        Y = np.array(Y)

        # Shuffle
        np.random.seed(seed=7)  # Initialize seed to get reproducible results
        ind = [i for i in range(X.shape[0])]
        np.random.shuffle(ind)
        X = X[ind]
        Y = Y[ind]

        ypred = self.model.predict(X)

        return Y, ypred

    def loadModelStrategy(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
