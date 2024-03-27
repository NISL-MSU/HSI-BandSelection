import torch
import random
import numpy as np
import torch.optim as optim
from torchsummary import summary
from ..Classification.networks import *
from src.HSIBandSelection import utils
from ..Classification.ModelStrategy import ModelStrategy

np.random.seed(7)  # Initialize seed to get reproducible results
random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class CNNObject:
    """Helper class used to store the main information of a CNN for training"""

    def __init__(self, model, criterion, optimizer):
        self.network = model
        self.criterion = criterion
        self.optimizer = optimizer


class CNNStrategy(ModelStrategy):

    def __init__(self):
        self.pca = None
        self.pls = None
        self.nbands = None
        self.data = None
        self.model = None

    def defineModel(self, device, data, nbands, windowSize, classes, train_y, pca, pls):
        """Override model declaration method"""
        model = Hyper3DNetLite(img_shape=(1, nbands, windowSize, windowSize), classes=int(classes), data=data)
        model.to(device)
        # Training parameters
        if classes == 2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            class_count = [i for i in utils.get_class_distribution(train_y).values()]
            class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

        optimizer = optim.Adadelta(model.parameters(), lr=1.0)

        self.pca = pca
        self.pls = pls
        self.nbands = nbands
        self.data = data

        self.model = CNNObject(model, criterion, optimizer)

    def trainFoldStrategy(self, trainx, train_y, train, batch_size, classes, device,
                          epochs, valx, test, means, stds, filepath, printProcess):
        np.random.seed(7)  # Initialize seed to get reproducible results
        random.seed(7)
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Prints summary of the modelif printProcess:      
        if printProcess:
            summary(self.model.network, (1, trainx.shape[2], trainx.shape[3], trainx.shape[4]))
            
        indexes = np.arange(len(train))  # Prepare list of indexes for shuffling
        T = np.ceil(1.0 * len(train) / batch_size).astype(np.int32)  # Compute the number of steps in an epoch
        val_acc = 0
        loss = 1
        for epoch in range(epochs):  # Epoch loop
            # Shuffle indexes when epoch begins

            self.model.network.train()  # Sets training mode
            running_loss = 0.0
            for step in range(T):  # Batch loop
                # Generate indexes of the batch
                inds = indexes[step * batch_size:(step + 1) * batch_size]
                trainb = train[inds]

                # Get actual batches
                trainxb = torch.from_numpy(trainx[inds]).float().to(device)
                trainyb = torch.from_numpy(train_y[trainb]).long().to(device)

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
                if step % 10 == 9 and printProcess:  # print every 10 mini-batches
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
        valxn = utils.applynormalize(valx, means, stds)
        if self.pca:
            valxn = utils.applyPCA(valxn, numComponents=self.nbands, dataset=self.data)
        elif self.pls:
            valxn = utils.applyPLS(valxn, numComponents=self.nbands, dataset=self.data)

        ypred = []
        with torch.no_grad():
            self.model.network.eval()
            Teva = np.ceil(1.0 * len(test) / batch_size).astype(np.int32)
            indtest = np.arange(len(test))
            for b in range(Teva):
                inds = indtest[b * batch_size:(b + 1) * batch_size]
                ypred_batch = self.model.network(torch.from_numpy(valxn[inds]).float().to(device))
                if classes > 2:
                    y_pred_softmax = torch.log_softmax(ypred_batch, dim=1)
                    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
                else:
                    y_pred_tags = torch.round(torch.sigmoid(ypred_batch))
                ypred = ypred + (y_pred_tags.cpu().numpy()).tolist()
        ytest = torch.from_numpy(train_y[test]).long().cpu().numpy()

        return ytest, ypred

    def loadModelStrategy(self, path):
        self.model.network.load_state_dict(torch.load(path))
