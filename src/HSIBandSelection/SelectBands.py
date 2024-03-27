import os
import torch
import pickle
import numpy as np
from scipy.signal import find_peaks
from src.HSIBandSelection.utils import Dataset, process_data
from src.HSIBandSelection.TrainSelection import TrainSelection
from src.HSIBandSelection.InterBandRedundancy import InterBandRedundancy


class SelectBands:
    def __init__(self, dataset: Dataset, method: str = 'GSS', classifier: str = 'CNN', nbands: int = 5,
                 transform: bool = False, average: bool = False, epochs: int = 150, batch_size: int = 128,
                 scratch: bool = True):
        """Class used for performing hyperspectral band selection
        :param dataset: utils.Dataset object
        :param method: Method name. Options: 'GSS', 'PCA' (IBRA+PCA), and 'PLS' (IBRA+PLS)
        :param classifier: Classifier type. Options: 'CNN' (if data is 2D), 'ANN', 'RF', 'SVM'
        :param nbands: How many spectral bands you want to select
        :param transform: If True, the final selected bands will suffer a Gaussian transformation to simulate being a multispectral band
        :param average: If True, average consecutive bands to reduce the initial total # of bands to half
        :param epochs: Number of iterations used to train the NN models
        :param batch_size: Batch size used to train the NN models
        :param scratch: If True, execute the IBRA process from scratch and replace previously saved results
        """
        self.method = method
        self.classifier = classifier
        self.nbands = nbands
        self.transform = transform
        self.average = average
        self.epochs = epochs
        self.batch_size = batch_size
        self.scratch = scratch

        # Pre-process data
        self.dataset = process_data(dataset, flag_average=average, transform=False)

        # Initialize IBRA
        self.interB = InterBandRedundancy(dataset=dataset, printProcess=False)

        if method == 'PCA':
            self.pca = True  # IBRA forms a set of candidate bands and then we reduce the number of bands using PCA
        elif method == 'PLS':
            self.pls = True  # IBRA forms a set of candidate bands and then we reduce the number of bands using LDA

    def run_selection(self, init_vf=12, final_vf=5):
        """Execute band selection algorithm
        :param init_vf: Initial Variance Inflation Factor threshold (used for IBRA)
        :param final_vf: Final Variance Inflation Factor threshold (used for IBRA)"""
        data = self.dataset.name

        for t in reversed(range(final_vf, init_vf + 1)):
            print("Testing VIF threshold: " + str(t))

            # Check if the analysis have been made before
            filepreselected = data + "//results//" + self.method + "//preselection_" + data + "_VIF" + str(t)
            filedistances = data + "//results//" + self.method + "//distances_" + data + "_VIF" + str(t)
            fileselected = data + "//results//" + self.method + "//" + str(self.nbands) + " bands//selection_" + data + '100' + \
                           self.classifier + str(self.nbands) + "bands_VIF" + str(t) + ".txt"

            # If the folder does not exist, create it
            folder = data
            if not os.path.exists(folder):
                os.mkdir(folder)
                os.mkdir(folder + "//results//")
                os.mkdir(folder + "//results//" + self.method)
            folder = data + "//results//" + self.method
            if not os.path.exists(folder):
                os.mkdir(folder)

            if os.path.exists(filepreselected) or (not self.scratch):
                with open(filepreselected, 'rb') as f:
                    indexes = pickle.load(f)
            else:
                self.interB.setT(t)
                # Get the distribution of distances
                dist = self.interB.clusters()
                # Calculate local minima
                dist.insert(0, 100)  # Add high values at the beginning and the end so that initial
                dist.append(100)  # and final bands can be considered as local minima
                indexes, _ = find_peaks(np.max(dist) - dist, height=0)
                dist = np.array(dist[1:-1])  # Remove the dummy points previously added
                indexes = indexes - 1
                # Remove points with a distance greater or equal than 5 (not suitable centers)
                indexes = [p for p in indexes if dist[p] < 5]
                # Save pre-selected bands for VIF value of t
                with open(filepreselected, 'wb') as fi:
                    pickle.dump(indexes, fi)
                # Save distribution of distances for VIF value of t
                with open(filedistances, 'wb') as fi:
                    pickle.dump(dist, fi)

            # Get the k-selected bands based on IE
            new_dataset = process_data(self.dataset, flag_average=False, transform=self.transform)
            net = TrainSelection(method=self.method, classifier=self.classifier, batch_size=self.batch_size,
                                 epochs=self.epochs, plot=False, selection=indexes, th=str(t), dataset=new_dataset,
                                 pca=self.pca, pls=self.pls, transform=self.transform)
            if self.method == 'GSS':
                index, entr = net.selection(select=self.nbands)
                # Save selected bands as txt file
                with open(fileselected, 'w') as x_file:
                    x_file.write(str(index))
                # Save scores of each of the bands
                with open(data + "//results//" + self.method + "//bandScores_" + data + '100' + self.classifier + "_VIF" + str(t), 'wb') as fi:
                    pickle.dump(entr, fi)
            else:
                if self.pca:
                    print("Applying PCA over the IBRA-preselected bands and training a classifier")
                else:
                    print("Applying PLS over the IBRA-preselected bands and training a classifier")

            # Train selected bands if the selected set of bands was not trained before
            if not os.path.exists(
                    data + "//results//" + self.method + "//" + str(self.nbands) + " bands//classification_report5x2_100" +
                    self.classifier + self.method + str(self.nbands) + data + str(t) + ".txt"):
                np.random.seed(7)  # Re-Initialize seed to get reproducible results
                torch.manual_seed(7)
                torch.cuda.manual_seed(7)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                net = TrainSelection(method=self.method, classifier=self.classifier, batch_size=self.batch_size,
                                     epochs=self.epochs, plot=False, selection=indexes, th=str(t), dataset=new_dataset,
                                     pca=self.pca, pls=self.pls, transform=self.transform)
                net.train()
                # net.validate()  # Store the evaluation metrics
