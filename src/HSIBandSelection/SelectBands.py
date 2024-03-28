import os
import torch
import pickle
import numpy as np
from scipy.signal import find_peaks
from .TrainSelection import TrainSelection
from .InterBandRedundancy import InterBandRedundancy
from .utils import Dataset, process_data, getPCA, getPLS


class SelectBands:
    def __init__(self, dataset: Dataset, method: str = 'GSS', classifier: str = 'CNN', nbands: int = 5,
                 transform: bool = False, average: bool = False, epochs: int = 150, batch_size: int = 128,
                 scratch: bool = True):
        """Class used for performing hyperspectral band selection
        :param dataset: utils.Dataset object
        :param method: Method name. Options: 'GSS', 'PCA' (IBRA+PCA), and 'PLS' (IBRA+PLS)
        :param classifier: Classifier type. Options: 'CNN' (if data is 2D), 'ANN', 'RF', 'SVM'
        :param nbands: How many spectral bands you want to select or reduce to
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
        self.dataset = dataset

        # Pre-process data for IBRA
        IBRAdataset = process_data(dataset, flag_average=average, transform=False, normalization=True)

        # Initialize IBRA
        self.interB = InterBandRedundancy(dataset=IBRAdataset, printProcess=False)

        self.pca, self.pls = False, False
        if method == 'PCA':
            self.pca = True  # IBRA forms a set of candidate bands and then we reduce the number of bands using PCA
        elif method == 'PLS':
            self.pls = True  # IBRA forms a set of candidate bands and then we reduce the number of bands using LDA

    def run_selection(self, init_vf=12, final_vf=5):
        """Execute band selection algorithm
        :param init_vf: Initial Variance Inflation Factor threshold (used for IBRA)
        :param final_vf: Final Variance Inflation Factor threshold (used for IBRA)
        """
        data = self.dataset.name
        f1_best = 0
        IBRA_best, stats_best, VIF_best, GSS_best = None, None, None, None

        for t in reversed(range(final_vf, init_vf + 1)):
            print("*************************************")
            print("Testing VIF threshold: " + str(t))
            print("*************************************")

            # Check if the analysis have been made before
            filepreselected = data + "//results//" + self.method + "//preselection_" + data + "_VIF" + str(t)
            filedistances = data + "//results//" + self.method + "//distances_" + data + "_VIF" + str(t)
            fileselected = data + "//results//" + self.method + "//" + str(
                self.nbands) + " bands//selection_" + data + '100' + \
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
                    IBRAindexes = pickle.load(f)
            else:
                self.interB.setT(t)
                # Get the distribution of distances
                dist = self.interB.clusters()
                # Calculate local minima
                dist.insert(0, 100)  # Add high values at the beginning and the end so that initial
                dist.append(100)  # and final bands can be considered as local minima
                IBRAindexes, _ = find_peaks(np.max(dist) - dist, height=0)
                dist = np.array(dist[1:-1])  # Remove the dummy points previously added
                IBRAindexes = IBRAindexes - 1
                # Remove points with a distance greater or equal than 5 (not suitable centers)
                IBRAindexes = [p for p in IBRAindexes if dist[p] < 5]
                # Save pre-selected bands for VIF value of t
                with open(filepreselected, 'wb') as fi:
                    pickle.dump(IBRAindexes, fi)
                # Save distribution of distances for VIF value of t
                with open(filedistances, 'wb') as fi:
                    pickle.dump(dist, fi)

            # Get the k-selected bands based on IE
            new_dataset = process_data(self.dataset, selection=IBRAindexes, flag_average=False,
                                       transform=self.transform)
            net = TrainSelection(method=self.method, classifier=self.classifier, batch_size=self.batch_size,
                                 epochs=self.epochs, plot=False, th=str(t), dataset=new_dataset,
                                 pca=self.pca, pls=self.pls, transform=self.transform)
            GSSindexes = None
            if self.method == 'GSS':
                GSSindexes, entr = net.selection(select=self.nbands)
            else:
                if self.pca:
                    print("Applying PCA over the IBRA-preselected bands and training a classifier")
                else:
                    print("Applying PLS over the IBRA-preselected bands and training a classifier")

            # Train selected bands if the selected set of bands was not trained before
            print("\n Training a model using 5x2 CV using the final selected or reduced bands...")
            np.random.seed(7)  # Re-Initialize seed to get reproducible results
            torch.manual_seed(7)
            torch.cuda.manual_seed(7)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            net = TrainSelection(method=self.method, classifier=self.classifier, batch_size=self.batch_size,
                                 epochs=self.epochs, plot=False, th=str(t), dataset=new_dataset,
                                 pca=self.pca, pls=self.pls, transform=self.transform)
            stats = net.train()

            if stats.f1 > f1_best:
                f1_best = stats.f1
                IBRA_best = IBRAindexes
                stats_best, VIF_best, GSS_best = stats, t, GSSindexes

        print("The best F1 performance was achieved using a VIF = {}".format(VIF_best))

        if self.method == 'GSS':
            print("The best band combination obtained using GSS was {}".format(GSS_best))
            return VIF_best, IBRA_best, GSS_best, stats_best
        else:
            new_dataset = process_data(self.dataset, selection=IBRA_best, flag_average=False, transform=self.transform)
            if self.pca:
                return VIF_best, IBRA_best, getPCA(new_dataset.train_x, numComponents=self.nbands), stats_best
            else:
                return VIF_best, IBRA_best, getPLS(new_dataset.train_x, numComponents=self.nbands), stats_best


if __name__ == '__main__':
    from HSIBandSelection.readSAT import loadata, createImageCubes

    X, Y = loadata(name='IP')
    X, Y = createImageCubes(X, Y, window=5)
    datast = Dataset(train_x=X, train_y=Y, name='IP')

    selector = SelectBands(dataset=datast, method='GSS', nbands=5)
    selector.run_selection()
