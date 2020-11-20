import utils
import os
import numpy as np
import statsmodels.api as sm
from torchsummary import summary
from networks import *
from sklearn.model_selection import StratifiedKFold
import torch
from torch import from_numpy
import torch.optim as optim
import pickle
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from LRP import LRP
from sklearn.model_selection import train_test_split

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Static Functions
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def weight_reset(m):
    """Reset model weights after one epoch"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def plot_confusion_matrix(cm, cms, classescf, cmap=plt.cm.Blues):
    """Print and plot the confusion matrix"""
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classescf))
    plt.xticks(tick_marks, classescf, rotation=45)
    plt.yticks(tick_marks, classescf)

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            if (cm[i, j] == 100 or cm[i, j] == 0) and cms[i, j] == 0:
                plt.text(j, i, '{0:.0f}'.format(cm[i, j]) + '\n$\pm$' + '{0:.0f}'.format(cms[i, j]),
                         horizontalalignment="center",
                         verticalalignment="center", fontsize=15,
                         color="white" if cm[i, j] > thresh else "black")

            else:
                plt.text(j, i, '{0:.2f}'.format(cm[i, j]) + '\n$\pm$' + '{0:.2f}'.format(cms[i, j]),
                         horizontalalignment="center",
                         verticalalignment="center", fontsize=15,
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Main Class Definition
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class TrainSelection:
    """Class used for training a dataset using the Hyper3DNetLite network"""

    def __init__(self, nbands=6, method='SSA', transform=False, average=True, batch_size=128, epochs=150,
                 data='Kochia', plot=False, selection=None, th='', median=False, vif=0):
        """
        @param: nbands - Desired number of bands.
        @param: method - Band selection method. Options: 'SSA', 'OCF', 'GA', 'PLS', 'FNGBS', 'FullSpec'.
        @param: transform - Flag used to simulate Gaussian bandwidths.
        @param: average - Flag used to reduce the number of bands averaging consecutive bands.
        @param: batch_size - Size of the batch used for training.
        @param: epochs - Number of epochs used for training.
        @param: data - Name of the dataset. Options: 'Kochia', 'Avocado'
        @param: selection - Load only the selected bands from the dataset
        @param: th - Optional index to add in the end of the generated files
        @param: median - If True, perform a median filtering on the spectral bands.
        """
        if selection is not None:
            self.nbands = len(selection)
        else:
            self.nbands = nbands
        self.method = method
        self.transform = transform
        self.average = average
        self.batch_size = batch_size
        self.epochs = epochs
        self.data = data
        self.th = th

        # Read the data using the specified parameters
        self.trainx, self.train_y, self.indexes = \
            utils.load_data(nbands=nbands, flag_average=average, method=method, transform=transform, data=self.data,
                            selection=selection, median=median, vifv=vif)
        # Reshape as a 4-D TENSOR
        self.trainx = np.reshape(self.trainx, (self.trainx.shape[0], self.trainx.shape[1], self.trainx.shape[2],
                                               self.trainx.shape[3], 1))
        # Find the minimum and maximum values per dimension
        self.minD = np.zeros((self.nbands,))
        self.maxD = np.zeros((self.nbands,))
        for i in range(self.nbands):
            self.minD[i] = np.min(self.trainx[:, :, :, i, 0])
            self.maxD[i] = np.max(self.trainx[:, :, :, i, 0])

        # Shuffle dataset
        np.random.seed(seed=7)  # Initialize seed to get reproducible results
        ind = [i for i in range(self.trainx.shape[0])]
        np.random.shuffle(ind)
        self.trainx = self.trainx[ind]
        self.train_y = self.train_y[ind]

        # Transpose dimensions to fit Pytorch order
        self.trainx = self.trainx.transpose((0, 4, 3, 1, 2))

        self.windowSize = self.trainx.shape[-1]  # Gets the size of the window
        self.classes = len(np.unique(self.train_y))  # Gets the number of classes in the target vector
        folds = 2
        if self.data == "Kochia" or self.data == "Avocado":
            folds = 10
        self.kfold = StratifiedKFold(n_splits=folds, shuffle=False)  # Initialize kfold object

        # Load model
        print("Loading model...")
        self.model = Hyper3DNetLite(img_shape=(1, self.nbands, self.windowSize, self.windowSize),
                                    classes=int(self.classes))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Prints summary of the model
        summary(self.model, (1, self.nbands, self.windowSize, self.windowSize))

        # Training parameters
        if self.classes == 2:
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.data == "Kochia":
            class_count = [i for i in self.get_class_distributionKochia().values()]
            class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            class_count = [i for i in self.get_class_distributionIP().values()]
            class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=1.0)

        self.plot = plot

    def evaluate(self, test):
        """Return the numpy target and predicted vectors as numpy vectors.
        @ param: test - List of test indexes
        """
        ypred = []
        with torch.no_grad():
            self.model.eval()
            Teva = np.ceil(1.0 * len(test) / self.batch_size).astype(np.int32)
            for b in range(Teva):
                inds = test[b * self.batch_size:(b + 1) * self.batch_size]
                ypred_batch = self.model(from_numpy(self.trainx[inds]).float().to(self.device))
                if self.classes > 2:
                    y_pred_softmax = torch.log_softmax(ypred_batch, dim=1)
                    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
                else:
                    y_pred_tags = torch.round(torch.sigmoid(ypred_batch))
                ypred = ypred + (y_pred_tags.cpu().numpy()).tolist()
        ytest = from_numpy(self.train_y[test]).long().cpu().numpy()

        return ytest, ypred

    def confusion_matrix(self, ypred, ytest):
        """Calculate confusion matrix"""
        con_mat = np.zeros((self.classes, self.classes))
        for p in zip(ypred, ytest):
            tl, pl = p
            con_mat[tl, pl] = con_mat[tl, pl] + 1

        con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=3)
        classes_list = list(range(0, int(self.classes)))
        con_mat_df = pd.DataFrame(con_mat_norm, index=classes_list, columns=classes_list)
        return con_mat_df

    def get_class_distributionKochia(self):
        """Get number of samples per class"""
        count_dict = {"0": 0, "1": 0, "2": 0}

        for i in self.train_y:
            if i == 0:
                count_dict['0'] += 1
            elif i == 1:
                count_dict['1'] += 1
            elif i == 2:
                count_dict['2'] += 1

        return count_dict

    def get_class_distributionIP(self):
        """Get number of samples per class"""
        count_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0, "10": 0, "11": 0,
                      "12": 0, "13": 0, "14": 0, "15": 0}

        for i in self.train_y:
            if i == 0:
                count_dict['0'] += 1
            elif i == 1:
                count_dict['1'] += 1
            elif i == 2:
                count_dict['2'] += 1
            if i == 3:
                count_dict['3'] += 1
            elif i == 4:
                count_dict['4'] += 1
            elif i == 5:
                count_dict['5'] += 1
            if i == 6:
                count_dict['6'] += 1
            elif i == 7:
                count_dict['7'] += 1
            elif i == 8:
                count_dict['8'] += 1
            if i == 9:
                count_dict['9'] += 1
            elif i == 10:
                count_dict['10'] += 1
            elif i == 11:
                count_dict['11'] += 1
            if i == 12:
                count_dict['12'] += 1
            elif i == 13:
                count_dict['13'] += 1
            elif i == 14:
                count_dict['14'] += 1
            elif i == 15:
                count_dict['15'] += 1

        return count_dict

    def train(self):
        if self.data == "Kochia" or self.data == "Avocado":
            self.train10x1()
        else:
            self.train5x2()

    def train10x1(self):

        np.random.seed(seed=7)  # Initialize seed to get reproducible results
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        """Train the network"""
        # Create lists to store metrics
        cvoa = []
        cvpre = []
        cvrec = []
        cvf1 = []

        # If the folder does not exist, create it
        folder = self.data + "//results//" + self.method + "//" + str(self.nbands) + " bands"
        if not os.path.exists(folder):
            os.mkdir(folder)

        ntrain = 1
        transform = ''
        if self.transform:
            transform = 'GAUSS'
        for train, test in self.kfold.split(self.trainx, self.train_y):
            print("\n******************************")
            print("Starting fold: " + str(ntrain))
            print("******************************")
            indexes = np.arange(len(train))  # Prepare list of indexes for shuffling
            T = np.ceil(1.0 * len(train) / self.batch_size).astype(np.int32)  # Compute the number of steps in an epoch
            val_acc = 0
            loss = 1

            filepath = ''
            for epoch in range(self.epochs):  # Epoch loop
                # Shuffle indexes when epoch begins
                np.random.shuffle(indexes)

                self.model.train()  # Sets training mode
                running_loss = 0.0
                for step in range(T):  # Batch loop
                    # Generate indexes of the batch
                    inds = indexes[step * self.batch_size:(step + 1) * self.batch_size]
                    trainb = train[inds]

                    # Get actual batches
                    trainxb = from_numpy(self.trainx[trainb]).float().to(self.device)
                    trainyb = from_numpy(self.train_y[trainb]).long().to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.model(trainxb)
                    if self.classes == 2:
                        trainyb = trainyb.unsqueeze(1)
                        trainyb = trainyb.float()
                    loss = self.criterion(outputs, trainyb)
                    loss.backward()
                    self.optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    if step % 10 == 9:  # print every 200 mini-batches
                        print('[%d, %5d] loss: %.5f' %
                              (epoch + 1, step + 1, running_loss / 10))
                        running_loss = 0.0

                # Validation step
                ytest, ypred = self.evaluate(test)
                if self.classes == 2:
                    ypred = np.array(ypred).reshape((len(ypred),))
                correct_pred = (np.array(ypred) == ytest).astype(float)
                oa = correct_pred.sum() / len(correct_pred) * 100  # Calculate accuracy

                # Save model if accuracy improves
                if oa >= val_acc:
                    val_acc = oa
                    filepath = folder + "//selected" + self.method + str(self.nbands) + "-weights-hyper3dnetLite-" \
                               + self.data + str(ntrain) + transform + self.th  # saves checkpoint
                    torch.save(self.model.state_dict(), filepath)  # saves checkpoint

                print('VALIDATION: Epoch %d, loss: %.5f, acc: %.3f, best_acc: %.3f' %
                      (epoch + 1, loss.item(), oa.item(), val_acc))

            # Calculate metrics for the ntrain-fold
            self.model.load_state_dict(torch.load(filepath))  # loads checkpoint
            ytest, ypred = self.evaluate(test)
            if self.classes == 2:
                ypred = np.array(ypred).reshape((len(ypred),))
            correct_pred = (np.array(ypred) == ytest).astype(float)
            oa = correct_pred.sum() / len(correct_pred) * 100
            prec, rec, f1, support = precision_recall_fscore_support(ytest, ypred, average='macro')

            # Add metrics to the list
            cvoa.append(oa)
            cvpre.append(prec * 100)
            cvrec.append(rec * 100)
            cvf1.append(f1 * 100)

            # Reset all weights
            self.model.apply(weight_reset)

            ntrain += 1

        # Save metrics in a txt file
        file_name = folder + "//classification_report_hyper3dnetLite_" + self.method + str(self.nbands) + self.data \
                    + transform + self.th + ".txt"
        with open(file_name, 'w') as x_file:
            x_file.write("Overall accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvoa)), float(np.std(cvoa))))
            x_file.write('\n')
            x_file.write("Precision accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvpre)), float(np.std(cvpre))))
            x_file.write('\n')
            x_file.write("Recall accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvrec)), float(np.std(cvrec))))
            x_file.write('\n')
            x_file.write("F1 accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvf1)), float(np.std(cvf1))))

    def train5x2(self):
        """Train using 5x2 validation"""
        np.random.seed(seed=7)  # Initialize seed to get reproducible results
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Create lists to store metrics
        cvoa = []
        cvpre = []
        cvrec = []
        cvf1 = []

        # If the folder does not exist, create it
        folder = self.data + "//results//" + self.method + "//" + str(self.nbands) + " bands"
        if not os.path.exists(folder):
            os.mkdir(folder)

        # Choose seeds for each iteration
        seeds = [13, 51, 137, 24659, 347, 436, 123, 64, 958, 234]
        ntrain = 1
        transform = ''
        if self.transform:
            transform = 'GAUSS'
        # Iterate through each partition
        for i_s, seed in enumerate(seeds):
            # Split the dataset in 2 parts with the current seed
            train, test = train_test_split(range(len(self.trainx)), test_size=0.70, random_state=seed,
                                           stratify=self.train_y)
            print("\n******************************")
            print("Starting fold: " + str(ntrain))
            print("******************************")
            indexes = np.arange(len(train))  # Prepare list of indexes for shuffling
            T = np.ceil(1.0 * len(train) / self.batch_size).astype(np.int32)  # Compute the number of steps in an epoch
            val_acc = 0
            loss = 1

            filepath = ''
            for epoch in range(self.epochs):  # Epoch loop
                # Shuffle indexes when epoch begins
                np.random.shuffle(indexes)

                self.model.train()  # Sets training mode
                running_loss = 0.0
                for step in range(T):  # Batch loop
                    # Generate indexes of the batch
                    inds = indexes[step * self.batch_size:(step + 1) * self.batch_size]
                    trainb = np.array(train)[inds]

                    # Get actual batches
                    trainxb = from_numpy(self.trainx[trainb]).float().to(self.device)
                    trainyb = from_numpy(self.train_y[trainb]).long().to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.model(trainxb)
                    if self.classes == 2:
                        trainyb = trainyb.unsqueeze(1)
                        trainyb = trainyb.float()
                    loss = self.criterion(outputs, trainyb)
                    loss.backward()
                    self.optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    if step % 10 == 9:  # print every 200 mini-batches
                        print('[%d, %5d] loss: %.5f' %
                              (epoch + 1, step + 1, running_loss / 10))
                        running_loss = 0.0

                # Validation step
                ytest, ypred = self.evaluate(test)
                if self.classes == 2:
                    ypred = np.array(ypred).reshape((len(ypred),))
                correct_pred = (np.array(ypred) == ytest).astype(float)
                oa = correct_pred.sum() / len(correct_pred) * 100  # Calculate accuracy

                # Save model if accuracy improves
                if oa >= val_acc:
                    val_acc = oa
                    filepath = folder + "//selected" + self.method + str(self.nbands) + "-weights-hyper3dnetLite-" \
                               + self.data + str(ntrain) + transform + self.th  # saves checkpoint
                    torch.save(self.model.state_dict(), filepath)  # saves checkpoint

                print('VALIDATION: Epoch %d, loss: %.5f, acc: %.3f, best_acc: %.3f' %
                      (epoch + 1, loss.item(), oa.item(), val_acc))

            # Calculate metrics for the ntrain-fold
            self.model.load_state_dict(torch.load(filepath))  # loads checkpoint
            ytest, ypred = self.evaluate(test)
            if self.classes == 2:
                ypred = np.array(ypred).reshape((len(ypred),))
            correct_pred = (np.array(ypred) == ytest).astype(float)
            oa = correct_pred.sum() / len(correct_pred) * 100
            prec, rec, f1, support = precision_recall_fscore_support(ytest, ypred, average='macro')

            # Add metrics to the list
            cvoa.append(oa)
            cvpre.append(prec * 100)
            cvrec.append(rec * 100)
            cvf1.append(f1 * 100)

            # Reset all weights
            self.model.apply(weight_reset)

            ntrain += 1

        # Save metrics in a txt file
        file_name = folder + "//classification_report_hyper3dnetLite_" + self.method + str(self.nbands) + self.data \
                    + transform + self.th + ".txt"
        with open(file_name, 'w') as x_file:
            x_file.write("Overall accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvoa)), float(np.std(cvoa))))
            x_file.write('\n')
            x_file.write("Precision accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvpre)), float(np.std(cvpre))))
            x_file.write('\n')
            x_file.write("Recall accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvrec)), float(np.std(cvrec))))
            x_file.write('\n')
            x_file.write("F1 accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvf1)), float(np.std(cvf1))))

    def validate(self):
        if self.data == "Kochia" or self.data == "Avocado":
            self.validate10x1()
        else:
            self.validate5x2()

    def validate10x1(self):
        """Calculate validation metrics and plot confusion matrix"""

        np.random.seed(seed=7)  # Initialize seed to get reproducible results
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Create lists to store metrics
        cvoa = []
        cvpre = []
        cvrec = []
        cvf1 = []

        folder = self.data + "//results//" + self.method + "//" + str(self.nbands) + " bands"

        ntrain = 1
        transform = ''
        if self.transform:
            transform = 'GAUSS'
        confmatrices = np.zeros((10, int(self.classes), int(self.classes)))
        for train, test in self.kfold.split(self.trainx, self.train_y):
            print("\n******************************")
            print("Validating fold: " + str(ntrain))
            print("******************************")
            filepath = self.data + "//results//" + self.method + "//" + str(self.nbands) \
                       + " bands//selected" + self.method + str(self.nbands) + "-weights-hyper3dnetLite-" + \
                       self.data + str(ntrain) + transform + self.th  # saves checkpoint
            # Calculate metrics for the ntrain-fold
            self.model.load_state_dict(torch.load(filepath))  # loads checkpoint
            ytest, ypred = self.evaluate(test)
            if self.classes == 2:
                ypred = np.array(ypred).reshape((len(ypred),)).astype(np.int8)
            correct_pred = (np.array(ypred) == ytest).astype(float)
            oa = correct_pred.sum() / len(correct_pred) * 100
            prec, rec, f1, support = precision_recall_fscore_support(ytest, ypred, average='macro')
            con_mat_df = self.confusion_matrix(ypred, ytest)
            confmatrices[ntrain - 1, :, :] = con_mat_df.values

            # Add metrics to the list
            cvoa.append(oa)
            cvpre.append(prec * 100)
            cvrec.append(rec * 100)
            cvf1.append(f1 * 100)

            ntrain += 1

        # Save metrics in a txt file
        file_name = folder + "//classification_report_hyper3dnetLite_" + self.method + str(self.nbands) + self.data \
                    + transform + self.th + ".txt"
        with open(file_name, 'w') as x_file:
            x_file.write("Overall accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvoa)), float(np.std(cvoa))))
            x_file.write('\n')
            x_file.write("Precision accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvpre)), float(np.std(cvpre))))
            x_file.write('\n')
            x_file.write("Recall accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvrec)), float(np.std(cvrec))))
            x_file.write('\n')
            x_file.write("F1 accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvf1)), float(np.std(cvf1))))

        # Calculate mean and std
        means = np.mean(confmatrices * 100, axis=0)
        stds = np.std(confmatrices * 100, axis=0)

        with open(folder + '//meanshyper3dnet' + self.method + str(self.nbands) + self.data +
                  transform + self.th, 'wb') as fi:
            pickle.dump(means, fi)
        with open(folder + '//stdshyper3dnet' + self.method + str(self.nbands) + self.data +
                  transform + self.th, 'wb') as fi:
            pickle.dump(stds, fi)
        with open(folder + '//cvf1hyper3dnet' + self.method + str(self.nbands) + self.data +
                  transform + self.th, 'wb') as fi:
            pickle.dump(cvf1, fi)

        # Plot confusion matrix
        if self.plot:
            classes_list = list(range(0, int(self.classes)))
            plot_confusion_matrix(means, stds, classescf=classes_list)
            plt.savefig(folder + '//MatrixConfusion_' + self.method +
                        str(self.nbands) + self.data + transform + '.png', dpi=600)

    def validate5x2(self):
        """Validate using 5x2 validation"""
        # Create lists to store metrics
        cvoa = []
        cvpre = []
        cvrec = []
        cvf1 = []

        # If the folder does not exist, create it
        folder = self.data + "//results//" + self.method + "//" + str(self.nbands) + " bands"
        if not os.path.exists(folder):
            os.mkdir(folder)

        # Choose seeds for each iteration
        seeds = [13, 51, 137, 24659, 347, 436, 123, 64, 958, 234]
        ntrain = 1
        transform = ''
        if self.transform:
            transform = 'GAUSS'
        confmatrices = np.zeros((10, int(self.classes), int(self.classes)))
        # Iterate through each partition
        for i_s, seed in enumerate(seeds):
            # Split the dataset in 2 parts with the current seed
            train, test = train_test_split(range(len(self.trainx)), test_size=0.70, random_state=seed,
                                           stratify=self.train_y)
            print("\n******************************")
            print("Validating fold: " + str(ntrain))
            print("******************************")
            filepath = self.data + "//results//" + self.method + "//" + str(self.nbands) \
                       + " bands//selected" + self.method + str(self.nbands) + "-weights-hyper3dnetLite-" + \
                       self.data + str(ntrain) + transform + self.th  # saves checkpoint
            # Calculate metrics for the ntrain-fold
            self.model.load_state_dict(torch.load(filepath))  # loads checkpoint
            ytest, ypred = self.evaluate(test)
            if self.classes == 2:
                ypred = np.array(ypred).reshape((len(ypred),)).astype(np.int8)
            correct_pred = (np.array(ypred) == ytest).astype(float)
            oa = correct_pred.sum() / len(correct_pred) * 100
            prec, rec, f1, support = precision_recall_fscore_support(ytest, ypred, average='macro')
            con_mat_df = self.confusion_matrix(ypred, ytest)
            confmatrices[ntrain - 1, :, :] = con_mat_df.values

            # Add metrics to the list
            cvoa.append(oa)
            cvpre.append(prec * 100)
            cvrec.append(rec * 100)
            cvf1.append(f1 * 100)

            ntrain += 1

            # Save metrics in a txt file
        file_name = folder + "//classification_report_hyper3dnetLite_" + self.method + str(self.nbands) + self.data \
                    + transform + self.th + ".txt"
        with open(file_name, 'w') as x_file:
            x_file.write("Overall accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvoa)), float(np.std(cvoa))))
            x_file.write('\n')
            x_file.write("Precision accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvpre)), float(np.std(cvpre))))
            x_file.write('\n')
            x_file.write("Recall accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvrec)), float(np.std(cvrec))))
            x_file.write('\n')
            x_file.write("F1 accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvf1)), float(np.std(cvf1))))

        # Calculate mean and std
        means = np.mean(confmatrices * 100, axis=0)
        stds = np.std(confmatrices * 100, axis=0)

        with open(folder + '//meanshyper3dnet' + self.method + str(self.nbands) + self.data +
                  transform + self.th, 'wb') as fi:
            pickle.dump(means, fi)
        with open(folder + '//stdshyper3dnet' + self.method + str(self.nbands) + self.data +
                  transform + self.th, 'wb') as fi:
            pickle.dump(stds, fi)
        with open(folder + '//cvf1hyper3dnet' + self.method + str(self.nbands) + self.data +
                  transform + self.th, 'wb') as fi:
            pickle.dump(cvf1, fi)

        # Plot confusion matrix
        if self.plot:
            classes_list = list(range(0, int(self.classes)))
            plot_confusion_matrix(means, stds, classescf=classes_list)
            plt.savefig(folder + '//MatrixConfusion_' + self.method +
                        str(self.nbands) + self.data + transform + '.png', dpi=600)

    def saliency(self):
        """Estimate the saliency of each of the pre-selected spectral bands"""

        np.random.seed(seed=7)  # Initialize seed to get reproducible results
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        SA = np.zeros((10, self.nbands,))  # Create array to store saliency metrics
        ntrain = 1
        for train, test in self.kfold.split(self.trainx, self.train_y):
            print("\n******************************")
            print("Analyzing fold: " + str(ntrain))
            print("******************************")

            # Load weights of the network trained with the pre-selected bands.
            filepath = self.data + "//results//" + self.method + "//" + str(self.nbands) \
                       + " bands//selected" + self.method + str(self.nbands) + "-weights-hyper3dnetLite-" + \
                       self.data + str(ntrain) + self.th
            self.model.load_state_dict(torch.load(filepath))  # load checkpoint

            # Calculate losses
            with torch.no_grad():
                self.model.eval()
                # Calculate loss using original data
                ypred = self.model(from_numpy(self.trainx[test]).float().to(self.device))
                trainyb = from_numpy(self.train_y[test]).long().to(self.device)
                if self.classes == 2:
                    trainyb = trainyb.unsqueeze(1)
                    trainyb = trainyb.float()
                loss1 = self.criterion(ypred, trainyb)
                loss1 = loss1.item()
                # Calculate loss removing one band
                for nchannel in range(0, self.trainx.shape[2]):
                    xtest = self.trainx[test].copy()
                    # Zero-out selected band
                    xtest[:, :, nchannel, :, :] = np.zeros(
                        (self.trainx[test].shape[0], self.trainx.shape[1], self.trainx.shape[3], self.trainx.shape[4]))
                    # Calculate new loss
                    ypred = self.model(from_numpy(xtest).float().to(self.device))
                    loss2 = self.criterion(ypred, trainyb)
                    loss2 = loss2.item()
                    # Calculate the saliency as the change in loss
                    SA[ntrain - 1][nchannel] = np.sum(loss2 - loss1)

            ntrain += 1

        # Stores saliencies
        with open(self.data + "//results//" + 'SA_' + self.data, 'wb') as fi:
            pickle.dump(SA, fi)

    def relevance(self):
        """Apply LRP (not used in the paper) """
        ntrain = 1
        for train, test in self.kfold.split(self.trainx, self.train_y):
            print("\n******************************")
            print("Analyzing fold: " + str(ntrain))
            print("******************************")

            # Load weights of the network trained with the pre-selected bands.
            filepath = self.data + "\\results\\" + self.method + "\\" + str(self.nbands) \
                       + " bands\\selected" + self.method + str(self.nbands) + "-weights-hyper3dnetLite-" + \
                       self.data + str(ntrain) + self.th
            self.model.load_state_dict(torch.load(filepath))  # load checkpoint

            # Get the relevances of each images in the training set
            self.model.eval()
            LRP(self.model, from_numpy(self.trainx[train]).float().to(self.device), device=self.device)

    def selection(self, select=6):

        # Calculate the entropy of each pre-selected band
        entropies = [utils.entropy(self.trainx[:, :, i, :, :]) for i in range(len(self.indexes))]

        # Sort the pre-selected bands according to their entropies (in decreasing order)
        preselected = self.indexes.copy()
        pairs = list(tuple(zip(preselected, entropies)))
        pairs.sort(key=lambda x: x[1], reverse=True)
        preselected, _ = zip(*pairs)

        # Select the first "select" bands
        preselected = list(preselected)
        selection = preselected[:select]
        selection.sort()
        preselected = preselected[select:]

        # Train using the bands in "selection"
        f1base = self.tune(selection=selection)
        bestselection = selection.copy()

        # Try new bands until there is no more elements in the list
        while len(preselected) > 0:
            # Calculate the maximum VIF of all the band in "selection"
            VIF = self.checkMulticollinearity(s=selection)
            # Remove the band with the highest VIF of "selection"
            selection.remove(selection[VIF.index(max(VIF))])
            # Pop the next available band from "preselected"
            selection.append(preselected[0])
            selection.sort()
            preselected = preselected[1:]
            # Train using the bands in "selection"
            f1 = self.tune(selection=selection)
            # Check if the new selection has better performance than the previous one. If not, break
            if f1 > f1base:
                bestselection = selection.copy()
                f1base = f1
            elif f1 <= f1base - 0.05:
                break
            print("Best selection so far: " + str(bestselection) + "with an F1 score of " + str(f1base))

        return bestselection, entropies

    def tune(self, selection):
        """Get the F1 validation score using a set of selected bands"""
        np.random.seed(seed=7)  # Initialize seed to get reproducible results
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set model
        models = Hyper3DNetLite(img_shape=(1, len(selection), self.windowSize, self.windowSize),
                                classes=int(self.classes))
        models.to(self.device)
        if self.classes == 2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            class_count = [i for i in self.get_class_distributionKochia().values()]
            class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        optimizers = optim.Adadelta(models.parameters(), lr=1.0)

        # Select bands
        trainx = np.zeros((self.trainx.shape[0], 1, len(selection), self.windowSize, self.windowSize))
        c = 0
        for ib, band in enumerate(self.indexes):
            if band in selection:
                trainx[:, :, c, :, :] = self.trainx[:, :, ib, :, :]
                c += 1
        print("Training using bands: " + str(selection))

        # Train a simple classifier using individual bands
        for train, test in self.kfold.split(self.trainx, self.train_y):
            indexes = np.arange(len(train))  # Prepare list of indexes for shuffling
            T = np.ceil(1.0 * len(train) / self.batch_size).astype(np.int32)  # Compute the number of steps in an epoch
            for epoch in range(150):  # Epoch loop
                models.train()  # Sets training mode
                for step in range(T):  # Batch loop
                    # Generate indexes of the batch
                    inds = indexes[step * self.batch_size:(step + 1) * self.batch_size]
                    trainb = train[inds]

                    # Get actual batches
                    trainxb = from_numpy(trainx[trainb]).float().to(self.device)
                    trainyb = from_numpy(self.train_y[trainb]).long().to(self.device)

                    # zero the parameter gradients
                    optimizers.zero_grad()

                    # forward + backward + optimize
                    outputs = models(trainxb)
                    if self.classes == 2:
                        trainyb = trainyb.unsqueeze(1)
                        trainyb = trainyb.float()
                    loss = criterion(outputs, trainyb)
                    loss.backward()
                    optimizers.step()

            # Validation step
            ypred = []
            with torch.no_grad():
                models.eval()
                Teva = np.ceil(1.0 * len(test) / self.batch_size).astype(np.int32)
                for b in range(Teva):
                    inds = test[b * self.batch_size:(b + 1) * self.batch_size]
                    ypred_batch = models(from_numpy(trainx[inds]).float().to(self.device))
                    if self.classes > 2:
                        y_pred_softmax = torch.log_softmax(ypred_batch, dim=1)
                        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
                    else:
                        y_pred_tags = torch.round(torch.sigmoid(ypred_batch))
                    ypred = ypred + (y_pred_tags.cpu().numpy()).tolist()
            ytest = from_numpy(self.train_y[test]).long().cpu().numpy()
            if self.classes == 2:
                ypred = np.array(ypred).reshape((len(ypred),))
            # Calculate F1 score
            prec, rec, f1, support = precision_recall_fscore_support(ytest, ypred, average='macro')
            print("F1 score: " + str(f1))
            models.apply(weight_reset)
            return f1

    def checkMulticollinearity(self, s=None):
        vifV = []
        nbands = len(s)
        for n, i in enumerate(s):
            y = self.trainx[:, 0, np.where(np.array(self.indexes) == i)[0][0], :, :]
            x = np.zeros((self.trainx.shape[0], self.trainx.shape[3], self.trainx.shape[4], nbands - 1))
            c = 0
            for nb in s:
                if nb != i:
                    x[:, :, :, c] = self.trainx[:, 0, np.where(np.array(self.indexes) == nb)[0][0], :, :]
                    c += 1
            x = x.reshape((x.shape[0] * x.shape[1] * x.shape[2], nbands - 1))
            y = y.reshape((y.shape[0] * y.shape[1] * y.shape[2], 1))
            model = sm.OLS(y, x)
            results = model.fit()
            rsq = results.rsquared
            vifV.append(round(1 / (1 - rsq), 2))
            print("R Square value of {} band is {} keeping all other bands as features".format(s[n],
                                                                                               (round(rsq, 4))))
            print("Variance Inflation Factor of {} band is {} \n".format(s[n], vifV[n]))

        return vifV


if __name__ == '__main__':

    # Train using the bands selected by the other methods: Kochia
    net = TrainSelection(nbands=6, method='GA', transform=False, average=True, batch_size=128,
                         epochs=130, plot=False, data='Kochia')
    net.train()
    net = TrainSelection(nbands=6, method='GA', transform=True, average=True, batch_size=128,
                         epochs=130, plot=False, data='Kochia')
    net.train()
    net = TrainSelection(nbands=6, method='FNGBS', transform=False, average=True, batch_size=128,
                         epochs=130, plot=False, data='Kochia')
    net.train()
    net = TrainSelection(nbands=6, method='FNGBS', transform=True, average=True, batch_size=128,
                         epochs=130, plot=False, data='Kochia')
    net.train()
    net = TrainSelection(nbands=6, method='OCF', transform=False, average=True, batch_size=128,
                         epochs=130, plot=False, data='Kochia')
    net.train()
    net = TrainSelection(nbands=6, method='OCF', transform=True, average=True, batch_size=128,
                         epochs=130, plot=False, data='Kochia')
    net.train()
    net = TrainSelection(nbands=6, method='PLS', transform=False, average=True, batch_size=128,
                         epochs=130, plot=False, data='Kochia')
    net.train()
    net = TrainSelection(nbands=6, method='PLS', transform=True, average=True, batch_size=128,
                         epochs=130, plot=False, data='Kochia')
    net.train()
    net = TrainSelection(nbands=10, method='GA', transform=False, average=True, batch_size=128,
                         epochs=130, plot=False, data='Kochia')
    net.train()
    net = TrainSelection(nbands=10, method='GA', transform=True, average=True, batch_size=128,
                         epochs=130, plot=False, data='Kochia')
    net.train()
    net = TrainSelection(nbands=10, method='FNGBS', transform=False, average=True, batch_size=128,
                         epochs=130, plot=False, data='Kochia')
    net.train()
    net = TrainSelection(nbands=10, method='FNGBS', transform=True, average=True, batch_size=128,
                         epochs=130, plot=False, data='Kochia')
    net.train()
    net = TrainSelection(nbands=10, method='OCF', transform=False, average=True, batch_size=128,
                         epochs=130, plot=False, data='Kochia')
    net.train()
    net = TrainSelection(nbands=10, method='OCF', transform=True, average=True, batch_size=128,
                         epochs=130, plot=False, data='Kochia')
    net.train()
    net = TrainSelection(nbands=10, method='PLS', transform=False, average=True, batch_size=128,
                         epochs=130, plot=False, data='Kochia')
    net.train()
    net = TrainSelection(nbands=10, method='PLS', transform=True, average=True, batch_size=128,
                         epochs=130, plot=False, data='Kochia')
    net.train()

    # Train using the bands selected by the other methods: Avocado dataset

    net = TrainSelection(nbands=5, method='GA', transform=False, average=True, batch_size=8, median=True,
                         epochs=100, plot=False, data='Avocado')
    net.train()
    net = TrainSelection(nbands=5, method='GA', transform=True, average=True, batch_size=8, median=True,
                         epochs=100, plot=False, data='Avocado')
    net.train()
    net = TrainSelection(nbands=5, method='FNGBS', transform=False, average=True, batch_size=8, median=True,
                         epochs=100, plot=False, data='Avocado')
    net.train()
    net = TrainSelection(nbands=5, method='FNGBS', transform=True, average=True, batch_size=8, median=True,
                         epochs=100, plot=False, data='Avocado')
    net.validate()
    net = TrainSelection(nbands=5, method='OCF', transform=False, average=True, batch_size=8, median=True,
                         epochs=100, plot=False, data='Avocado')
    net.train()
    net = TrainSelection(nbands=5, method='OCF', transform=True, average=True, batch_size=8, median=True,
                         epochs=100, plot=False, data='Avocado')
    net.train()
    net = TrainSelection(nbands=5, method='PLS', transform=False, average=True, batch_size=8, median=True,
                         epochs=100, plot=False, data='Avocado')
    net.train()
    net = TrainSelection(nbands=5, method='PLS', transform=True, average=True, batch_size=8, median=True,
                         epochs=100, plot=False, data='Avocado')
    net.train()

    # Train using the selected bands by our method and the transformed data: Kochia and Avocado

    net = TrainSelection(nbands=6, method='SSA', transform=True, average=True, batch_size=128,
                         epochs=130, plot=False, data='Kochia')
    net.train()
    net = TrainSelection(nbands=10, method='SSA', transform=True, average=True, batch_size=128,
                         epochs=130, plot=False, data='Kochia')
    net.train()
    net = TrainSelection(nbands=5, method='SSA', transform=False, average=True, batch_size=8, median=True,
                         epochs=150, plot=False, data='Avocado')
    net.train5x2()

    # Train the bands selected by out inter-redundancy band method (VIF:12-5): Kochia and Avocado

    net = TrainSelection(nbands=19, method='SSA', transform=False, average=True, batch_size=128,
                         epochs=100, plot=False, data='Kochia', vif=12, th='12')
    net.train()
    net = TrainSelection(nbands=21, method='SSA', transform=False, average=True, batch_size=128,
                         epochs=100, plot=False, data='Kochia', vif=11, th='11')
    net.train()
    net = TrainSelection(nbands=17, method='SSA', transform=False, average=True, batch_size=128,
                         epochs=100, plot=False, data='Kochia', vif=10, th='10')
    net.train()
    net = TrainSelection(nbands=15, method='SSA', transform=False, average=True, batch_size=128,
                         epochs=120, plot=False, data='Kochia', vif=9, th='9')
    net.train()
    net = TrainSelection(nbands=16, method='SSA', transform=False, average=True, batch_size=128,
                         epochs=120, plot=False, data='Kochia', vif=8, th='8')
    net.train()
    net = TrainSelection(nbands=16, method='SSA', transform=False, average=True, batch_size=128,
                         epochs=120, plot=False, data='Kochia', vif=7, th='7')
    net.train()
    net = TrainSelection(nbands=15, method='SSA', transform=False, average=True, batch_size=128,
                         epochs=120, plot=False, data='Kochia', vif=6, th='6')
    net.train()
    net = TrainSelection(nbands=10, method='SSA', transform=False, average=True, batch_size=128,
                         epochs=150, plot=False, data='Kochia', vif=5, th='5')
    net.train()

    net = TrainSelection(nbands=10, method='SSA', transform=False, average=True, batch_size=8, median=True,
                         epochs=100, plot=False, data='Avocado', vif=12, th='12')
    net.train()
    net = TrainSelection(nbands=8, method='SSA', transform=False, average=True, batch_size=8, median=True,
                         epochs=100, plot=False, data='Avocado', vif=11, th='11')
    net.train()
    net = TrainSelection(nbands=9, method='SSA', transform=False, average=True, batch_size=8, median=True,
                         epochs=100, plot=False, data='Avocado', vif=10, th='10')
    net.train()
    net = TrainSelection(nbands=8, method='SSA', transform=False, average=True, batch_size=8, median=True,
                         epochs=100, plot=False, data='Avocado', vif=9, th='9')
    net.train()
    net = TrainSelection(nbands=9, method='SSA', transform=False, average=True, batch_size=8, median=True,
                         epochs=100, plot=False, data='Avocado', vif=8, th='8')
    net.train()
    net = TrainSelection(nbands=7, method='SSA', transform=False, average=True, batch_size=8, median=True,
                         epochs=100, plot=False, data='Avocado', vif=7, th='7')
    net.train()
    net = TrainSelection(nbands=7, method='SSA', transform=False, average=True, batch_size=8, median=True,
                         epochs=100, plot=False, data='Avocado', vif=6, th='6')
    net.train()
    net = TrainSelection(nbands=5, method='SSA', transform=False, average=True, batch_size=8, median=True,
                         epochs=100, plot=False, data='Avocado', vif=5, th='5')
    net.train()

    # Train using all the bands: Kochia and Avocado

    net = TrainSelection(nbands=150, method='FullSpec', transform=False, average=True, batch_size=128,
                         epochs=100, plot=False, data='Kochia')
    net.train()
    net = TrainSelection(nbands=150, method='FullSpec', transform=False, average=True, batch_size=8, median=True,
                         epochs=100, plot=False, data='Avocado')
    net.train()
