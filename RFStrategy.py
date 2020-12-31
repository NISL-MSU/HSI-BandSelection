from ModelStrategy import ModelStrategy
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import utils
import pickle


class RFStrategy(ModelStrategy):

    def defineModel(self, device, data, nbands, windowSize, classes, train_y):
        """Override model declaration method"""
        return RandomForestClassifier(n_jobs=-1, max_features=5, n_estimators=500, max_depth=200, random_state=7)

    def trainFoldStrategy(self, model, trainx, train_y, train, batch_size, classes, device,
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
        np.random.seed(seed=7)  # Initialize seed to get reproducible results
        ind = [i for i in range(X.shape[0])]
        np.random.shuffle(ind)
        X = X[ind][0:10000, :]
        Y = Y[ind][0:10000]

        # Train model
        model.fit(X, Y)

        # Save model
        with open(filepath, 'wb') as fi:
            pickle.dump(model, fi)

    def evaluateFoldStrategy(self, model, valx, train_y, test, means, stds, batch_size, classes, device):
        # Normalize the validation set based on the previous statistics
        valxn = utils.applynormalize(valx, means, stds)
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

        ypred = model.predict(X)

        return Y, ypred

    def loadModelStrategy(self, model, path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
