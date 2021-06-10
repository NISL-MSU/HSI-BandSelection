from ClassificationStrategy.ModelStrategy import ModelStrategy
import torch.optim as optim
from ClassificationStrategy.networks import *
import numpy as np
import utils
import torch


class ANNObject:
    """Helper class used to store the main information of a CNN for training"""
    def __init__(self, model, criterion, optimizer):
        self.network = model
        self.criterion = criterion
        self.optimizer = optimizer


class ANNStrategy(ModelStrategy):

    def __init__(self):
        self.model = None
        self.pca = None
        self.pls = None

    def defineModel(self, device, data, nbands, windowSize, classes, train_y, pca, pls):
        """Override model declaration method"""
        model = WeedANN(img_shape=(nbands,), classes=int(classes))
        model.to(device)
        # Training parameters
        if classes == 2:
            criterion = nn.BCEWithLogitsLoss()
        elif data == "Kochia":
            class_count = [i for i in utils.get_class_distributionKochia(train_y).values()]
            class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            class_count = [i for i in utils.get_class_distributionIP(train_y).values()]
            class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

        optimizer = optim.Adadelta(model.parameters(), lr=1.0)

        return ANNObject(model, criterion, optimizer)

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
        np.random.seed(seed=7)  # Initialize seed to get reproducible results
        ind = [i for i in range(X.shape[0])]
        np.random.shuffle(ind)
        X = X[ind][0:int(len(X)/4), :]
        Y = Y[ind][0:len(X)]

        indexes = np.arange(len(X))  # Prepare list of indexes for shuffling
        T = np.ceil(1.0 * len(X) / batch_size).astype(np.int32)  # Compute the number of steps in an epoch
        val_acc = 0
        loss = 1
        for epoch in range(epochs):  # Epoch loop
            # Shuffle indexes when epoch begins
            np.random.shuffle(indexes)

            self.model.network.train()  # Sets training mode
            running_loss = 0.0
            for step in range(T):  # Batch loop
                # Generate indexes of the batch
                inds = indexes[step * batch_size:(step + 1) * batch_size]

                # Get actual batches
                trainxb = torch.from_numpy(X[inds]).float().to(device)
                trainyb = torch.from_numpy(Y[inds]).long().to(device)

                # zero the parameter gradients
                self.model.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model.network(trainxb)
                if classes == 2:
                    trainyb = trainyb.unsqueeze(1)
                    trainyb = trainyb.float()
                loss = self.model.criterion(outputs, trainyb)
                loss.backward()
                self.model.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if step % 10 == 9 and printProcess:   # print every 200 mini-batches
                    print('[%d, %5d] loss: %.5f' %
                          (epoch + 1, step + 1, running_loss / 10))
                    running_loss = 0.0

            # Validation step
            ytest, ypred = self.evaluateFoldStrategy(valx, train_y, test,
                                                     means, stds, batch_size, classes, device)
            if classes == 2:
                ypred = np.array(ypred).reshape((len(ypred),))
            correct_pred = (np.array(ypred) == ytest).astype(float)
            oa = correct_pred.sum() / len(correct_pred) * 100  # Calculate accuracy

            # Save model if accuracy improves
            if oa >= val_acc:
                val_acc = oa
                torch.save(self.model.network.state_dict(), filepath)  # saves checkpoint

            if printProcess:
                print('VALIDATION: Epoch %d, loss: %.5f, acc: %.3f, best_acc: %.3f' %
                      (epoch + 1, loss.item(), oa.item(), val_acc))

    def evaluateFoldStrategy(self, valx, train_y, test, means, stds, batch_size, classes, device):
        # Normalize the validation set based on the previous statistics
        valxn = utils.applynormalize(valx, means, stds)
        # Permute and reshape the data
        X = valxn.copy()[:, 0, :, :, :]
        X = X.transpose((1, 0, 2, 3))
        X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
        X = X.transpose((1, 0))

        # Get augmented target vector
        ytest = []
        for i in range(len(valxn)):
            yt = train_y[test][i]
            for _ in range(int(len(X) / len(valxn))):
                ytest.append(yt)
        ytest = np.array(ytest)

        # Shuffle
        np.random.seed(seed=7)  # Initialize seed to get reproducible results
        ind = [i for i in range(X.shape[0])]
        np.random.shuffle(ind)
        X = X[ind]
        ytest = ytest[ind]

        ypred = []
        with torch.no_grad():
            self.model.network.eval()
            Teva = np.ceil(1.0 * len(ytest) / batch_size).astype(np.int32)
            indtest = np.arange(len(ytest))
            for b in range(Teva):
                inds = indtest[b * batch_size:(b + 1) * batch_size]
                ypred_batch = self.model.network(torch.from_numpy(X[inds]).float().to(device))
                if classes > 2:
                    y_pred_softmax = torch.log_softmax(ypred_batch, dim=1)
                    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
                else:
                    y_pred_tags = torch.round(torch.sigmoid(ypred_batch))
                ypred = ypred + (y_pred_tags.cpu().numpy()).tolist()
        ytest = torch.from_numpy(ytest).long().cpu().numpy()

        return ytest, ypred

    def loadModelStrategy(self, path):
        self.model.network.load_state_dict(torch.load(path))
