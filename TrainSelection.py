import utils
import os
import numpy as np
import statsmodels.api as sm
from ClassificationStrategy.networks import *
from sklearn.model_selection import StratifiedKFold
import torch
import pickle
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ClassificationStrategy.Model import Model

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

    def __init__(self, nbands=6, method='SSA', classifier='CNN', transform=False, average=True, batch_size=128,
                 epochs=150, data='Kochia', size=100, plot=False, selection=None, th='', median=False, vif=0,
                 pca=False, pls=False):
        """
        @param nbands: Desired number of bands.
        @param method: Band selection method. Options: 'SSA', 'OCF', 'GA', 'PLS', 'FNGBS', 'FullSpec', 'Compressed.
        @param classifier: Type of model used to train the classifiers. Options: 'CNN', 'SVM', 'RF'.
        @param transform: Flag used to simulate Gaussian bandwidths.
        @param average: Flag used to reduce the number of bands averaging consecutive bands.
        @param batch_size: Size of the batch used for training.
        @param epochs: Number of epochs used for training.
        @param data: Name of the dataset. Options: 'Kochia', 'Avocado'
        @param size: Percentage of the dataset used for the experiments.
        @param selection: Load only the selected bands from the dataset
        @param th: Optional index to add in the end of the generated files
        @param median: If True, perform a median filtering on the spectral bands.
        @param pca: If True, we use the IBRA method to form a set of candidate bands and then we reduce the number of \
        bands using PCA.
        @param pls: If True, we use the IBRA method to form a set of candidate bands and then we reduce the number of \
        bands using LDA.
        """
        if selection is not None:
            self.nbands = len(selection)
        else:
            self.nbands = nbands
        self.method = method
        self.transform = transform
        self.average = average
        self.batch_size = batch_size
        self.classifier = classifier
        self.epochs = epochs
        self.data = data
        self.size = size
        self.plot = plot
        self.pca = pca
        self.pls = pls
        self.th = th

        # Read the data using the specified parameters
        self.trainx, self.train_y, self.indexes = \
            utils.load_data(nbands=nbands, flag_average=average, method=method, transform=transform, data=self.data,
                            selection=selection, median=median, vifv=vif, pca=pca, pls=pls)
        # Reshape as a 4-D TENSOR
        self.trainx = np.reshape(self.trainx, (self.trainx.shape[0], self.trainx.shape[1], self.trainx.shape[2],
                                               self.trainx.shape[3], 1))
        # In case of using Kyle Webster's compression method, the number of bands is decided by the imported data
        if self.method == 'Compressed':
            self.nbands = self.trainx.shape[3]
        # Find the minimum and maximum values per dimension
        self.minD = np.zeros((self.nbands,))
        self.maxD = np.zeros((self.nbands,))
        for i in range(self.nbands):
            self.minD[i] = np.min(self.trainx[:, :, :, i, 0])
            self.maxD[i] = np.max(self.trainx[:, :, :, i, 0])

        # Shuffle dataset and reduce dataset size
        np.random.seed(seed=7)  # Initialize seed to get reproducible results
        ind = [i for i in range(self.trainx.shape[0])]
        np.random.shuffle(ind)
        self.trainx = self.trainx[ind][0:int(len(self.trainx) * self.size / 100), :, :, :, :]
        self.train_y = self.train_y[ind][0:int(len(self.trainx))]

        # Transpose dimensions to fit Pytorch order
        self.trainx = self.trainx.transpose((0, 4, 3, 1, 2))

        self.windowSize = self.trainx.shape[-1]  # Gets the size of the window
        self.classes = len(np.unique(self.train_y))  # Gets the number of classes in the target vector
        folds = 10
        self.kfold = StratifiedKFold(n_splits=folds, shuffle=False)  # Initialize kfold object

        # Load model
        print("Loading model...")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Model(self.classifier, self.data, self.device, self.nbands, self.windowSize,
                           self.train_y, self.classes, self.pca, self.pls)

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

    def train(self):
        """Train using 10x1 or 5x2 cross-validation"""
        if self.data == "Avocado":
            crossval = '10x1'
            iterator = self.kfold.split(self.trainx, self.train_y)
        else:
            # Choose seeds for each iteration is using 5x2 cross-validation
            seeds = [13, 51, 137, 24659, 347, 436, 123, 64, 958, 234]
            crossval = '5x2'
            iterator = enumerate(seeds)

        # Create lists to store metrics
        cvoa = []
        cvpre = []
        cvrec = []
        cvf1 = []

        # If the folder does not exist, create it
        folder = self.data
        if not os.path.exists(folder):
            os.mkdir(folder)
            os.mkdir(folder + "//results//")
            os.mkdir(folder + "//results//" + self.method)
        folder = self.data + "//results//" + self.method + "//" + str(self.nbands) + " bands"
        if not os.path.exists(folder):
            os.mkdir(folder)

        # Set string values to define the file names
        ntrain = 1
        size = ''
        if self.size != 100:
            size = str(self.size)
        transform = ''
        if self.transform:
            transform = 'GAUSS'
        pca = ''
        if self.pca:
            pca = 'PCA'
        elif self.pls:
            pca = 'PLS'

        # Iterate through each partition
        for first, second in iterator:
            if crossval == '10x1':
                # Gets the list of training and test images using kfold.split
                train = np.array(first)
                test = np.array(second)
                print("Using 10x1 cross-validation for this dataset")
            else:
                # Split the dataset in 2 parts with the current seed
                train, test = train_test_split(range(len(self.trainx)), test_size=0.50, random_state=second)
                train = np.array(train)
                test = np.array(test)
                print("Using 5x2 cross-validation for this dataset")

            print("\n******************************")
            print("Starting fold: " + str(ntrain))
            print("******************************")
            # Normalize using the training set
            trainx, means, stds = utils.normalize(self.trainx[train])
            if self.pca:
                trainx = utils.getPCA(trainx, numComponents=self.nbands, dataset=self.data)
            elif self.pls:
                trainx = utils.getPLS(trainx, self.train_y[train], numComponents=self.nbands, dataset=self.data)
            valx = self.trainx[test]

            # Define path where the model will be saved
            filepath = folder + "//selected" + crossval + size + self.method + pca + str(self.nbands) + \
                       "-weights-" + self.classifier + "-" + self.data + str(ntrain) + transform + self.th

            # Train the model using the current training-validation split
            self.model.trainFold(trainx, self.train_y, train, self.batch_size, self.epochs, valx, test, means, stds,
                                 filepath)

            # Calculate metrics for the ntrain-fold
            self.model.loadModel(filepath)  # load checkpoint
            ytest, ypred = self.model.evaluateFold(valx, self.train_y, test, means, stds, self.batch_size)
            if self.classes == 2:
                ypred = np.array(ypred).reshape((len(ypred),))
            correct_pred = (np.array(ypred) == ytest).astype(float)
            oa = correct_pred.sum() / len(correct_pred) * 100
            prec, rec, f1, support = precision_recall_fscore_support(ytest, ypred, average='macro')
            print("Validation accuracy: " + str(oa))

            # Add metrics to the list
            cvoa.append(oa)
            cvpre.append(prec * 100)
            cvrec.append(rec * 100)
            cvf1.append(f1 * 100)

            # Reset all weights if training a CNN
            if self.classifier == 'CNN' or self.classifier == 'ANN':
                self.model.strategy.model.network.apply(weight_reset)

            ntrain += 1

        # Save metrics in a txt file
        file_name = folder + "//classification_report" + crossval + "_" + size + self.classifier + "_" + self.method + \
                    str(self.nbands) + self.data + transform + pca + self.th + ".txt"
        with open(file_name, 'w') as x_file:
            x_file.write("Overall accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvoa)), float(np.std(cvoa))))
            x_file.write('\n')
            x_file.write("Precision accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvpre)), float(np.std(cvpre))))
            x_file.write('\n')
            x_file.write("Recall accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvrec)), float(np.std(cvrec))))
            x_file.write('\n')
            x_file.write("F1 accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvf1)), float(np.std(cvf1))))

    def validate(self):
        """Calculate validation metrics and plot confusion matrix"""
        if self.data == "Avocado":
            crossval = '10x1'
            iterator = self.kfold.split(self.trainx, self.train_y)
        else:
            # Choose seeds for each iteration is using 5x2 cross-validation
            seeds = [13, 51, 137, 24659, 347, 436, 123, 64, 958, 234]
            crossval = '5x2'
            iterator = enumerate(seeds)

        # Create lists to store metrics
        cvoa = []
        cvpre = []
        cvrec = []
        cvf1 = []

        # If the folder does not exist, create it
        folder = self.data + "//results//" + self.method + "//" + str(self.nbands) + " bands"
        if not os.path.exists(folder):
            os.mkdir(folder)

        # Set string values to define the file names
        ntrain = 1
        size = ''
        if self.size != 100:
            size = str(self.size)
        transform = ''
        if self.transform:
            transform = 'GAUSS'
        pca = ''
        if self.pca:
            pca = 'PCA'
        elif self.pls:
            pca = 'PLS'

        confmatrices = np.zeros((10, int(self.classes), int(self.classes)))
        # Iterate through each partition
        for first, second in iterator:
            if crossval == '10x1':
                # Gets the list of training and test images using kfold.split
                train = np.array(first)
                test = np.array(second)
                print("Using 10x1 cross-validation for this dataset")
            else:
                # Split the dataset in 2 parts with the current seed
                train, test = train_test_split(range(len(self.trainx)), test_size=0.50, random_state=second)
                train = np.array(train)
                test = np.array(test)
                print("Using 5x2 cross-validation for this dataset")

            print("\n******************************")
            print("Validating fold: " + str(ntrain))
            print("******************************")
            trainx, means, stds = utils.normalize(self.trainx[train])
            if self.pca:
                _ = utils.getPCA(trainx, numComponents=self.nbands, dataset=self.data)
            elif self.pls:
                _ = utils.getPLS(trainx, self.train_y[train], numComponents=self.nbands, dataset=self.data)
            valx = self.trainx[test]

            # Define path where the model is saved
            filepath = folder + "//selected" + crossval + size + self.method + pca + str(self.nbands) + \
                       "-weights-" + self.classifier + "-" + self.data + str(ntrain) + transform + self.th
            self.model.loadModel(filepath)

            # Calculate metrics for the ntrain-fold
            ytest, ypred = self.model.evaluateFold(valx, self.train_y, test, means, stds, self.batch_size)
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
        file_name = folder + "//classification_report5x2_" + size + self.classifier + "_" + self.method + \
                    str(self.nbands) + self.data + transform + pca + self.th + ".txt"
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

        with open(folder + '//means' + size + self.classifier + self.method + str(self.nbands) + self.data +
                  transform + pca + self.th, 'wb') as fi:
            pickle.dump(means, fi)
        with open(folder + '//stds' + size + self.classifier + self.method + str(self.nbands) + self.data +
                  transform + pca + self.th, 'wb') as fi:
            pickle.dump(stds, fi)
        with open(folder + '//cvf1' + size + self.classifier + self.method + str(self.nbands) + self.data +
                  transform + pca + self.th, 'wb') as fi:
            pickle.dump(cvf1, fi)

        # Plot confusion matrix
        if self.plot:
            classes_list = list(range(0, int(self.classes)))
            plot_confusion_matrix(means, stds, classescf=classes_list)
            plt.savefig(folder + '//MatrixConfusion_' + self.method + self.classifier + size +
                        str(self.nbands) + self.data + transform + pca + '.png', dpi=600)

    def selection(self, select=6):
        """Select the top k bands using the Greedy Spectral Selection method"""
        # Calculate the entropy of each pre-selected band
        trainx, _, _ = utils.normalize(self.trainx)
        entropies = [utils.entropy(trainx[:, :, i, :, :]) for i in range(len(self.indexes))]

        # Sort the pre-selected bands according to their entropy (in decreasing order)
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
            print("Mean F1: " + str(f1))
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
        models = Model(self.classifier, self.data, self.device, len(selection), self.windowSize,
                       self.train_y, self.classes, self.pca, self.pls)

        # Select bands
        trainx = np.zeros((self.trainx.shape[0], 1, len(selection), self.windowSize, self.windowSize))
        c = 0
        for ib, band in enumerate(self.indexes):
            if band in selection:
                trainx[:, :, c, :, :] = self.trainx[:, :, ib, :, :]
                c += 1
        print("Training using bands: " + str(selection))

        f1t = 0
        # Choose seeds for each iteration
        seeds = [13, 51, 137, 24659, 347, 436, 123, 64, 958, 234]
        # Train a simple classifier using individual bands
        #        for train, test in self.kfold.split(trainx, self.train_y):
        for i_s, seed in enumerate(seeds):
            # Split the dataset in 2 parts with the current seed
            train, test = train_test_split(range(len(self.trainx)), test_size=0.50, random_state=seed,
                                           stratify=self.train_y)
            train = np.array(train)
            test = np.array(test)
            # Normalize using the training set
            trainxn, means, stds = utils.normalize(trainx[train])
            valx = trainx[test]

            filepath = self.data + "//results//" + self.method + "//temp2"
            # Train the model using the current training-validation split
            models.trainFold(trainxn, self.train_y, train, self.batch_size, 100, valx, test, means, stds,
                             filepath, False)
            # Validation step
            models.loadModel(filepath)
            ytest, ypred = models.evaluateFold(valx, self.train_y, test, means, stds, self.batch_size)
            os.remove(filepath)  # Remove the file as it is no longer needed
            # Calculate F1 score
            _, _, f1, _ = precision_recall_fscore_support(ytest, ypred, average='macro')
            print("F1 score: " + str(f1))

            f1t += f1
            # Reset all weights if training a CNN or ANN
            if self.classifier == 'CNN' or self.classifier == 'ANN':
                models.strategy.model.network.apply(weight_reset)

        return f1t / 10

    def checkMulticollinearity(self, s=None):
        """Calculate the VIF value of each selected band in s"""
        vifV = []
        nbands = len(s)
        trainx, _, _ = utils.normalize(self.trainx)
        for n, i in enumerate(s):
            y = trainx[:, 0, np.where(np.array(self.indexes) == i)[0][0], :, :]
            x = np.zeros((trainx.shape[0], trainx.shape[3], trainx.shape[4], nbands - 1))
            c = 0
            for nb in s:
                if nb != i:
                    x[:, :, :, c] = trainx[:, 0, np.where(np.array(self.indexes) == nb)[0][0], :, :]
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
    methods = ['GA', 'FNGBS', 'OCF', 'PLS']
    transforms = [False, True]
    bands = [6, 10]
    # for ba in bands:
    #     for me in methods:
    #         for tr in transforms:
    #             net = TrainSelection(nbands=ba, method=me, transform=tr, average=True, batch_size=128,
    #                                  epochs=130, plot=False, data='Kochia')
    #             net.train()

    # Kyle's experiment
    net = TrainSelection(method='Compressed', transform=False, average=False, batch_size=128,
                         epochs=130, plot=False, data='IP')
    net.train()
