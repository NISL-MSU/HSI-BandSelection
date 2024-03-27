import pickle
import numpy as np
import seaborn as sns
from tqdm import trange
from .utils import Dataset
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class InterBandRedundancy:

    def __init__(self, dataset: Dataset, threshold=10, printProcess=True):
        """Initialize Inter-band redundancy (IBRA) method
        :param dataset: Dataset object containing [train_x (data), train_y (labels), dataset_name]
        :param threshold: VIF threshold used for analysis
        :param printProcess: If True, print the VIF calculation per band
        """
        self.train_x, self.train_y, self.name = dataset.train_x, dataset.train_y, dataset.name
        self.printProcess = printProcess
        self.threshold = threshold
        self.table = np.zeros((self.train_x.shape[3], self.train_x.shape[3]))

    def setT(self, threshold):
        """Set VIF threshold"""
        self.threshold = threshold

    def vifPair(self, i, i2):
        """Calculates VIF value between i-th and i2-th bands"""
        y = self.train_x[:, :, :, i]  # Sets dependant variable
        x = self.train_x[:, :, :, i2]  # Sets independent variable
        # Reshape images into a 1-D vector
        x = x.reshape((self.train_x.shape[0] * self.train_x.shape[1] * self.train_x.shape[2], 1))
        y = y.reshape((self.train_x.shape[0] * self.train_x.shape[1] * self.train_x.shape[2], 1))
        model = sm.OLS(y, x)  # OLS regression
        results = model.fit()
        rsq = results.rsquared  # Gets R^2 value
        VIFValue = round(1 / (1 - rsq), 2)  # Computes VIF value
        if self.printProcess:
            print("Comparing band: " + str(i) + " and band: " + str(i2) + ". VIF: " + str(VIFValue))
        return VIFValue

    def clusters(self):
        distances_left = np.zeros((self.train_x.shape[3]))
        distances_right = np.zeros((self.train_x.shape[3]))

        for band in trange(self.train_x.shape[3]):
            # Check left
            d = 1  # Set initial distance
            vifVal = np.infty
            while vifVal > self.threshold and (band - d) > 0:
                if self.printProcess:
                    print("Evaluating band ", band, " with a distance ", d)
                if self.table[band, band - d] == 0:
                    self.table[band, band - d] = self.vifPair(band, band - d)
                    self.table[band - d, band] = self.table[band, band - d]
                vifVal = self.table[band, band - d]
                d += 1
            distances_left[band] = d - 1

            # Check right
            d = 1  # Set initial distance
            vifVal = np.infty
            while vifVal > self.threshold and (band + d) < self.train_x.shape[3]:
                if self.printProcess:
                    print("Evaluating band ", band, " with a distance ", d)
                if self.table[band, band + d] == 0:
                    self.table[band, band + d] = self.vifPair(band, band + d)
                    self.table[band + d, band] = self.table[band, band + d]
                vifVal = self.table[band, band + d]
                d += 1
            distances_right[band] = d - 1

        return list(np.abs(distances_left - distances_right))

    def plotSample(self, distances):
        """Plot the reflectance response and the distance plot. Only works for the Kochia dataset.
        This is used to replicate the figure reported in the papers"""
        # Takes a sample of 1000 random pixels
        ind = np.where(self.train_y == 0)[0]
        np.random.shuffle(ind)
        sample = np.reshape(self.train_x.transpose(3, 0, 1, 2), (self.train_x.shape[3],
                                                                 self.train_x.shape[0] * self.train_x.shape[1] *
                                                                 self.train_x.shape[2]))[:, ind[0:1000]]
        # Mean and std of each band
        means0 = np.mean(sample, axis=1)
        stds0 = np.std(sample, axis=1)

        # Takes a sample of 1000 random pixels
        ind = np.where(self.train_y == 1)[0]
        np.random.shuffle(ind)
        sample = np.reshape(self.train_x.transpose(3, 0, 1, 2),
                            (self.train_x.shape[3], self.train_x.shape[0] * self.train_x.shape[1] *
                             self.train_x.shape[2]))[:, ind[0:1000]]
        # Mean and std of each band
        means1 = np.mean(sample, axis=1)
        stds1 = np.std(sample, axis=1)

        # Takes a sample of 1000 random pixels
        ind = np.where(self.train_y == 2)[0]
        np.random.shuffle(ind)
        sample = np.reshape(self.train_x.transpose(3, 0, 1, 2),
                            (self.train_x.shape[3], self.train_x.shape[0] * self.train_x.shape[1] *
                             self.train_x.shape[2]))[:, ind[0:1000]]
        # Mean and std of each band
        means2 = np.mean(sample, axis=1)
        stds2 = np.std(sample, axis=1)

        # Plot graph
        fig, ax = plt.subplots()
        ax.set_ylabel('Reflectance')
        ax.set_xlabel('n : Band Index')
        clrs = sns.color_palette()
        with sns.axes_style("darkgrid"):
            class1, class2, class3 = 'class1', 'class2', 'class3'
            if self.name == "Kochia":
                class1, class2, class3 = 'Susceptible Kochia', 'Glyphosate Resistant', 'Dicamba Resistant'

            epochss = list(range(self.train_x.shape[3]))
            ax.plot(epochss, means0, c=clrs[0], linestyle='-', linewidth=2.5, label=class1)
            ax.fill_between(epochss, means0 - stds0, means0 + stds0, alpha=0.3, facecolor=clrs[0])
            ax.legend()
            ax.plot(epochss, means1, c=clrs[1], linestyle=':', linewidth=2.5, label=class2)
            ax.fill_between(epochss, means1 - stds1, means1 + stds1, alpha=0.3, facecolor=clrs[1])
            ax.legend()
            if len(clrs) > 2:
                ax.plot(epochss, means2, c=clrs[2], linestyle='-.', linewidth=3, label=class3)
                ax.fill_between(epochss, means2 - stds2, means2 + stds2, alpha=0.3, facecolor=clrs[2])
                ax.legend()
        # with open("Kochia/plots/Kochia_distances_VIF12", 'rb') as fil:
        #     ds = list(pickle.load(fil))
        with open(distances, 'rb') as fil:
            ds = list(pickle.load(fil))
        ax2 = ax.twinx()  # position of the xticklabels in the old x-axis
        ax2.set_ylabel('d(x_n) = |d_left(x_n) - d_right(x_n)|')
        ax2.plot(epochss, ds, c='k')
        ds.insert(0, 100)  # Add high values at the beginning and the end so that initial
        ds.append(100)
        indx, _ = find_peaks(np.max(ds) - ds, height=0)
        ds = np.array(ds[1:-1])
        indx = indx - 1
        # Remove points with a distance greater or equal than 5 (not suitable centers)
        indx = [p for p in indx if ds[p] < 5]
        ax2.plot(np.array(indx), ds[np.array(indx)], "x", c='r')
        ax3 = ax.twiny()
        newlabel = list(range(387, 1023 + 50, 50))
        funct = lambda tx: (tx - 387) / 4.24 + 10
        newpos = [funct(tx) for tx in newlabel]
        ax3.set_xlim([0, 150])
        ax3.set_xticks(newpos)
        ax3.set_xlabel('Wavelength (nm)')
        ax3.set_xticklabels(newlabel)
        plt.savefig('VIF12plusReflectance.png', dpi=600)
