import h5py
from scipy import integrate
from scipy import ndimage
from scipy.stats import norm
import statsmodels.api as sm
import pickle
import cv2
from Data.readSAT import *
from scipy import stats
import torch
from sklearn.metrics import r2_score
from numpy.random import binomial as binom
from sklearn.cross_decomposition import PLSRegression

# import matplotlib.pyplot as plt
np.random.seed(seed=7)  # Initialize seed to get reproducible results
torch.manual_seed(7)
torch.cuda.manual_seed(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def generic_gaussians(indices, bandwidth, L):
    """
    Given the indices of the wavelength centers and the
    filter bandwidth (given in nanometers), return the
    gaussians defined by these inputs
    """

    index_bandwidth = bandwidth
    all_gaussians = []

    # Given the bandwidth (which is equivalent to the full width-
    # half maximum of a Gaussian), calculate the standard deviation
    # FWHM = 2*sqrt(2ln(2))*stdev => stdev = FWHM/(2*sqrt(2ln(2)))
    stdev = np.divide(index_bandwidth, np.multiply(2, np.sqrt(np.multiply(np.log(2), 2))))
    
    for ind in indices:
        curve = np.linspace(0, L, L)
        pdf = norm.pdf(curve, ind, stdev)
        # pdf = np.multiply(pdf, np.divide(highest_count, np.max(pdf)))
        gauss = np.divide(pdf, np.max(pdf))  # Normalize to 1
        all_gaussians.append(gauss)
        # plt.plot(curve, pdf)

    return all_gaussians


def transform_data(produce_spectras, indices, bandwidth, L):
    """
    Get the original produce data, then transofrm that data
    using the histogram used to fit the histogram of wavelengths
    """

    gaussians = generic_gaussians(indices, bandwidth, L)

    new_data = []
    for spectrum in produce_spectras:
        old_spectrum = spectrum
        new_point = []

        # For each gaussian, integrate under the curve
        # multiplied by the original data point to
        # simulate a filter with the bandwidth of the
        # gaussian
        for g in gaussians:
            curve = np.multiply(old_spectrum, g)
            integral = integrate.trapz(curve)
            new_point.append(integral)

        # new_spectrum = np.multiply(old_spectrum, gaussian)

        new_data.append(new_point)

    return new_data


def add_rotation_flip(x, y):
    x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], x.shape[3], 1))

    # Flip horizontally
    x_h = np.flip(x[:, :, :, :, :], 1)
    # Flip vertically
    x_v = np.flip(x[:, :, :, :, :], 2)
    # Flip horizontally and vertically
    x_hv = np.flip(x_h[:, :, :, :, :], 2)

    # Concatenate
    x = np.concatenate((x, x_hv, x_v))
    y = np.concatenate((y, y, y))

    return x, y


def load_data(flag_average=True, median=False, nbands=np.infty, method='SSA', selection=None,
              transform=False, data='', vifv=0, pca=False, pls=False, normalization=False):
    """Load one of the satellite HSI datasets"""
    if data == "IP" or data == "PU" or data == "SA":
        train_x, train_y = loadata(data)
        train_x, train_y = createImageCubes(train_x, train_y, window=5)
    else:
        """Load Kochia or Avocado dataset"""
        hdf5_file = ''
        if data == "Kochia":
            hdf5_file = h5py.File('weed_dataset_w25.hdf5', "r")
        elif data == "Avocado":
            hdf5_file = h5py.File('avocado_dataset_w64.hdf5', "r")
        train_x = np.array(hdf5_file["train_img"][...]).astype(np.float32)
        train_y = np.array(hdf5_file["train_labels"][...])
    print("Dataset shape: " + str(train_x.shape))
    print("Loading and transforming the data into the correct format...")
    # Average consecutive bands
    if flag_average:
        if data == 'Avocado':
            img2 = np.zeros(
                (train_x.shape[0], int(train_x.shape[1] / 2), int(train_x.shape[2] / 2), int(train_x.shape[3] / 2)))
        else:
            img2 = np.zeros((train_x.shape[0], train_x.shape[1], train_x.shape[2], int(train_x.shape[3] / 2)))
        for n in range(0, train_x.shape[0]):
            if data == 'Avocado':
                xt = cv2.resize(np.float32(train_x[n, :, :, :]), (32, 32), interpolation=cv2.INTER_CUBIC)
                for i in range(0, train_x.shape[3], 2):
                    if median:
                        img2[n, :, :, int(i / 2)] = (ndimage.median_filter(xt[:, :, i], size=5) +
                                                     ndimage.median_filter(xt[:, :, i + 1], size=5)) / 2.
                    else:
                        img2[n, :, :, int(i / 2)] = (xt[:, :, i] + xt[:, :, i + 1]) / 2.
            # Average consecutive bands
            else:
                for i in range(0, train_x.shape[3], 2):
                    if median and int(i / 2) > 120:
                        img2[n, :, :, int(i / 2)] = (ndimage.median_filter(train_x[n, :, :, i], size=7) +
                                                     ndimage.median_filter(train_x[n, :, :, i + 1], size=7)) / 2.
                    else:
                        img2[n, :, :, int(i / 2)] = (train_x[n, :, :, i] + train_x[n, :, :, i + 1]) / 2.

        train_x = img2

    # Select a subset of bands if the flag "selected is activated"
    indexes = []
    if nbands < int(train_x.shape[3]) or selection is not None:
        if data == "Kochia":  # Selects indexes for the Kochia dataset
            if method == 'SSA':
                if nbands == 6 and not pca and not pls:
                    indexes = [1, 18, 43, 68, 81, 143]
                elif nbands == 8 and not pca and not pls:
                    indexes = [0, 4, 18, 31, 45, 63, 68, 74]
                elif nbands == 10 and not pca and not pls:
                    indexes = [2, 5, 18, 31, 42, 54, 68, 74, 79, 143]
                elif vifv == 12:  # Selected by the Inter-band redundancy method. VIF: 12.
                    indexes = [2, 5, 18, 31, 42, 54, 65, 68, 74, 79, 85, 89, 103, 106, 128, 132, 137, 143, 147]
                elif vifv == 11:  # Selected by the Inter-band redundancy method. VIF: 11.
                    indexes = [2, 5, 18, 31, 42, 47, 54, 65, 68, 74, 77, 80, 84, 89, 105, 132, 136, 139, 141, 143, 147]
                elif vifv == 10:  # Selected by the Inter-band redundancy method. VIF: 10.
                    indexes = [1, 18, 31, 43, 54, 64, 68, 78, 81, 84, 89, 105, 132, 136, 140, 143, 146]
                elif vifv == 9:  # Selected by the Inter-band redundancy method. VIF: 9.
                    indexes = [1, 18, 31, 43, 46, 54, 67, 74, 78, 81, 105, 132, 136, 139, 143]
                elif vifv == 8:  # Selected by the Inter-band redundancy method. VIF: 8.
                    indexes = [0, 4, 18, 31, 45, 55, 63, 68, 74, 79, 106, 132, 136, 140, 144, 146]
                elif vifv == 7:  # Selected by the Inter-band redundancy method. VIF: 7.
                    indexes = [0, 4, 18, 31, 43, 46, 56, 63, 66, 68, 74, 79, 105, 132, 140, 146]
                elif vifv == 6:  # Selected by the Inter-band redundancy method. VIF: 6.
                    indexes = [0, 4, 18, 47, 57, 62, 69, 74, 78, 81, 105, 132, 136, 140, 145]
                elif vifv == 5:  # Selected by the Inter-band redundancy method. VIF: 5.
                    indexes = [0, 18, 47, 61, 74, 79, 105, 132, 140, 145]
            elif method == 'FNGBS':
                if nbands == 6:
                    indexes = [14, 51, 78, 106, 111, 148]
                elif nbands == 8:
                    indexes = [14, 35, 51, 67, 78, 94, 111, 148]
                elif nbands == 10:
                    indexes = [14, 32, 33, 51, 67, 85, 100, 111, 128, 144]
            elif method == 'OCF':
                if nbands == 6:
                    indexes = [1, 129, 77, 82, 24, 42]
                elif nbands == 8:
                    indexes = [35, 63, 72, 78, 95, 97, 118, 138]
                elif nbands == 10:
                    indexes = [1, 15, 34, 42, 48, 56, 77, 82, 129, 132]
            elif method == 'GA':
                if nbands == 6:
                    indexes = [4, 21, 40, 61, 124, 138]
                elif nbands == 8:
                    indexes = [5, 14, 26, 45, 65, 84, 95, 102]
                elif nbands == 10:
                    indexes = [12, 30, 50, 69, 82, 98, 118, 132, 140, 146]
            elif method == 'PLS':
                if nbands == 6:
                    indexes = [35, 73, 88, 129, 134, 140]
                elif nbands == 8:
                    indexes = [23, 31, 35, 62, 71, 88, 113, 118]
                elif nbands == 10:
                    indexes = [23, 31, 35, 69, 73, 120, 129, 134, 140, 147]
            elif method == 'SRSSIM':
                if nbands == 6:
                    indexes = [2, 16, 42, 48, 55, 76]
                elif nbands == 8:
                    indexes = [2, 4, 16, 42, 48, 55, 75, 76]
                elif nbands == 10:
                    indexes = [2, 4, 16, 30, 42, 48, 55, 71, 75, 76]

        elif data == "Avocado":  # Selects indexes for the Avocado dataset
            if method == 'SSA':
                if nbands == 5:
                    indexes = [20, 41, 74, 102, 123]
                elif nbands == 10 and vifv == 12:  # Selected by the Inter-band redundancy method. VIF: 12.
                    indexes = [0, 20, 30, 43, 55, 60, 74, 98, 125, 133]
                elif nbands == 8 and vifv == 11:  # Selected by the Inter-band redundancy method. VIF: 11.
                    indexes = [0, 20, 30, 43, 60, 74, 99, 125]
                elif nbands == 9:  # Selected by the Inter-band redundancy method. VIF: 10.
                    indexes = [0, 20, 29, 43, 54, 59, 74, 99, 124]
                elif nbands == 8 and vifv == 9:  # Selected by the Inter-band redundancy method. VIF: 9.
                    indexes = [0, 20, 34, 43, 59, 74, 100, 125]
                elif nbands == 8 and vifv == 8:  # Selected by the Inter-band redundancy method. VIF: 8.
                    indexes = [0, 21, 36, 43, 59, 74, 101, 120, 124]
                elif nbands == 8 and vifv == 7:  # Selected by the Inter-band redundancy method. VIF: 7.
                    indexes = [0, 20, 36, 41, 74, 102, 123]
                elif nbands == 7 and vifv == 6:  # Selected by the Inter-band redundancy method. VIF: 6.
                    indexes = [0, 35, 46, 62, 75, 103, 126]
                elif nbands == 6 and vifv == 5:  # Selected by the Inter-band redundancy method. VIF: 5.
                    indexes = [0, 36, 75, 104, 112]
            elif method == 'FNGBS':
                if nbands == 5:
                    indexes = [5, 25, 38, 90, 124]
            elif method == 'OCF':
                if nbands == 5:
                    indexes = [33, 73, 97, 117, 144]
            elif method == 'GA':
                if nbands == 5:
                    indexes = [7, 23, 35, 49, 100]
            elif method == 'PLS':
                if nbands == 5:
                    indexes = [0, 74, 95, 135, 140]

        elif data == "IP":  # Selects indexes for the Avocado dataset
            if method == 'SSA':
                if nbands == 5 and not pca and not pls:
                    indexes = [11, 25, 34, 39, 67]
                elif vifv == 12:  # Selected by the Inter-band redundancy method. VIF: 12.
                    indexes = [2, 7, 11, 17, 26, 34, 39, 47, 56, 58, 60, 67, 74, 80, 89, 99, 104, 109, 125, 142, 144,
                               146, 148, 150, 169, 187, 191, 198]
                elif vifv == 11:  # Selected by the Inter-band redundancy method. VIF: 11.
                    indexes = [0, 7, 11, 15, 17, 20, 26, 34, 37, 39, 47, 56, 58, 60, 67, 74, 78, 89, 99, 104, 109, 125,
                               142, 144, 146, 148, 150, 169, 191, 198]
                elif vifv == 10:  # Selected by the Inter-band redundancy method. VIF: 10.
                    indexes = [0, 7, 11, 15, 17, 25, 34, 37, 39, 44, 47, 56, 58, 60, 67, 74, 78, 86, 93, 99, 104, 109,
                               125, 142, 144, 146, 148, 150, 169, 191, 199]
                elif vifv == 9:  # Selected by the Inter-band redundancy method. VIF: 9.
                    indexes = [0, 7, 12, 15, 17, 23, 25, 34, 46, 56, 58, 60, 67, 74, 78, 86, 93, 99, 104, 109, 125, 142,
                               144, 146, 148, 151, 169, 173, 199]
                elif vifv == 8:  # Selected by the Inter-band redundancy method. VIF: 8.
                    indexes = [0, 6, 12, 22, 26, 34, 46, 56, 58, 60, 67, 74, 78, 87, 93, 99, 104, 106, 109, 123, 143,
                               146, 148, 151, 168, 171, 198]
                elif vifv == 7:  # Selected by the Inter-band redundancy method. VIF: 7.
                    indexes = [0, 5, 12, 18, 20, 34, 46, 56, 58, 60, 67, 74, 78, 87, 93, 99, 104, 107, 109, 124, 143,
                               146, 148, 151, 171]
                elif vifv == 6:  # Selected by the Inter-band redundancy method. VIF: 6.
                    indexes = [0, 5, 19, 34, 46, 56, 58, 60, 67, 74, 78, 87, 93, 99, 104, 124, 144, 148, 150, 171]
                elif vifv == 5:  # Selected by the Inter-band redundancy method. VIF: 5.
                    indexes = [0, 19, 35, 45, 56, 58, 60, 67, 74, 78, 87, 93, 99, 104, 124, 144, 146, 148, 150, 169]
            elif method == 'FNGBS':
                if nbands == 5:
                    indexes = [28, 70, 92, 107, 129]
            elif method == 'OCF':
                if nbands == 5:
                    indexes = [16, 28, 50, 67, 90]
            elif method == 'GA':
                if nbands == 5:
                    indexes = [17, 31, 55, 75, 119]
            elif method == 'PLS':
                if nbands == 5:
                    indexes = [4, 27, 83, 96, 148]
            elif method == 'SRSSIM':
                if nbands == 5:
                    indexes = [28, 52, 91, 104, 121]

        elif data == "SA":  # Selects indexes for the Avocado dataset
            if method == 'SSA':
                if nbands == 5 and not pca and not pls:
                    indexes = [37, 60, 82, 92, 175]
                elif vifv == 12:  # Selected by the Inter-band redundancy method. VIF: 12.
                    indexes = [2, 16, 19, 22, 27, 37, 60, 65, 91, 104, 106, 127, 147, 175, 202]
                elif vifv == 11:  # Selected by the Inter-band redundancy method. VIF: 11.
                    indexes = [2, 17, 22, 28, 37, 60, 63, 91, 104, 106, 127, 147, 175, 202]
                elif vifv == 10:  # Selected by the Inter-band redundancy method. VIF: 10.
                    indexes = [2, 17, 22, 29, 37, 60, 64, 92, 103, 106, 127, 147, 175, 202]
                elif vifv == 9:  # Selected by the Inter-band redundancy method. VIF: 9.
                    indexes = [2, 17, 21, 37, 60, 92, 103, 106, 127, 147, 175, 202]
                elif vifv == 8:  # Selected by the Inter-band redundancy method. VIF: 8.
                    indexes = [2, 21, 37, 60, 82, 92, 103, 106, 127, 147, 175, 202]
                elif vifv == 7:  # Selected by the Inter-band redundancy method. VIF: 7.
                    indexes = [2, 21, 37, 60, 82, 92, 103, 106, 127, 147, 174, 202]
                elif vifv == 6:  # Selected by the Inter-band redundancy method. VIF: 6.
                    indexes = [2, 20, 38, 60, 82, 91, 104, 106, 127, 147, 174, 202]
                elif vifv == 5:  # Selected by the Inter-band redundancy method. VIF: 5.
                    indexes = [0, 19, 38, 60, 91, 104, 106, 127, 147, 174, 202]
            elif method == 'FNGBS':
                if nbands == 3:
                    indexes = [31, 88, 193]
                elif nbands == 5:
                    indexes = [16, 31, 113, 132, 175]
            elif method == 'OCF':
                if nbands == 3:
                    indexes = [34, 45, 120]
                elif nbands == 5:
                    indexes = [34, 45, 58, 93, 120]
            elif method == 'GA':
                if nbands == 3:
                    indexes = [19, 51, 91]
                elif nbands == 5:
                    indexes = [13, 20, 31, 44, 84]
            elif method == 'PLS':
                if nbands == 3:
                    indexes = [13, 38, 80]
                elif nbands == 5:
                    indexes = [10, 13, 38, 80, 146]
            elif method == 'SRSSIM':
                if nbands == 5:
                    indexes = [5, 47, 61, 81, 201]

        if selection is not None:
            indexes = selection

        # Sort indexes
        indexes.sort()
        print("Selecting bands: ", indexes)

        if transform:
            print("Transforming data...")
            nu = train_x.shape[0]
            w = train_x.shape[1]
            sp = train_x.shape[3]
            train_x = np.array(transform_data(produce_spectras=train_x.reshape((nu * w * w, sp)), bandwidth=5,
                                              indices=indexes, L=int(train_x.shape[3])))
            train_x = train_x.reshape((nu, w, w, len(indexes)))
        else:
            # Select bands from original image
            temp = np.zeros((train_x.shape[0], train_x.shape[1], train_x.shape[2], len(indexes)))

            for nb in range(0, len(indexes)):
                temp[:, :, :, nb] = train_x[:, :, :, indexes[nb]]

            train_x = temp.astype(np.float32)

    # Apply normalization from the beginning (used for the IBRA method)
    if normalization:
        train_x, _, _ = normalize(train_x)

    return train_x, train_y, indexes


def normalize(trainx):
    """Normalize and returns the calculated means and stds for each band"""
    trainxn = trainx.copy()
    means = np.zeros((trainx.shape[2], 1))
    stds = np.zeros((trainx.shape[2], 1))
    for n in range(trainx.shape[2]):
        if trainx.ndim == 5:  # Apply normalization to the data that is already in Pytorch format
            means[n, ] = np.mean(trainxn[:, :, n, :, :])
            stds[n, ] = np.std(trainxn[:, :, n, :, :])
            trainxn[:, :, n, :, :] = (trainxn[:, :, n, :, :] - means[n, ]) / (stds[n, ])
        elif trainx.ndim == 4:
            means[n, ] = np.mean(trainxn[:, :, :, n])
            stds[n, ] = np.std(trainxn[:, :, :, n])
            trainxn[:, :, :, n] = (trainxn[:, :, :, n] - means[n, ]) / (stds[n, ])
    return trainxn, means, stds


def applynormalize(testx, means, stds):
    """Apply normalization based on previous calculated means and stds"""
    testxn = testx.copy()
    for n in range(testx.shape[2]):
        testxn[:, :, n, :, :] = (testxn[:, :, n, :, :] - means[n, ]) / (stds[n, ])
    return testxn


def vif(inputX, printO=False, indexes=None):
    VIF = np.zeros((inputX.shape[3]))

    for i in range(0, inputX.shape[3]):
        y = inputX[:, :, :, i]
        x = np.zeros((inputX.shape[0], inputX.shape[1], inputX.shape[2], inputX.shape[3] - 1))
        c = 0
        for nb in range(0, inputX.shape[3]):
            if nb != i:
                x[:, :, :, c] = inputX[:, :, :, nb]
                c += 1
        x = x.reshape((inputX.shape[0] * inputX.shape[1] * inputX.shape[2], inputX.shape[3] - 1))
        y = y.reshape((inputX.shape[0] * inputX.shape[1] * inputX.shape[2], 1))
        model = sm.OLS(y, x)
        results = model.fit()
        rsq = results.rsquared
        VIF[i] = round(1 / (1 - rsq), 2)
        if printO:
            print("R Square value of {} band is {} keeping all other bands as features".format(
                indexes[i], (round(rsq, 2))))
            print("Variance Inflation Factor of {} band is {} \n".format(indexes[i], VIF[i]))

    return VIF


def tTest(method1='', nbands1=6, method2='', nbands2=6, data='', transform=False, file1=None, file2=None):
    """Perform a t-test between the results of two different methods."""

    if file1 is None and file2 is None:
        if transform:
            transform = 'GAUSS'
        else:
            transform = ''
        file1 = data + "\\results\\" + method1 + "\\" + str(nbands1) + " bands\\cvf1hyper3dnet" + method1 + \
                str(nbands1) + data + transform
        file2 = data + "\\results\\" + method2 + "\\" + str(nbands2) + " bands\\cvf1hyper3dnet" + method2 + \
                str(nbands2) + data + transform

    # Load the vectors of F1-scores obtain after 10-fold cross-validation
    with open(file1, 'rb') as f:
        cvf1 = pickle.load(f)

    with open(file2, 'rb') as f:
        cvf2 = pickle.load(f)

    for i in cvf1:
        print(i)
    print()
    for i in cvf2:
        print(i)

    return stats.ttest_rel(cvf1, cvf2).pvalue  # return the pvalue


def entropy(labels, base=2):
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()


def get_class_distributionKochia(train_y):
    """Get number of samples per class"""
    count_dict = {"0": 0, "1": 0, "2": 0}

    for i in train_y:
        if i == 0:
            count_dict['0'] += 1
        elif i == 1:
            count_dict['1'] += 1
        elif i == 2:
            count_dict['2'] += 1

    return count_dict


def get_class_distributionIP(train_y):
    """Get number of samples per class"""
    count_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0, "10": 0, "11": 0,
                  "12": 0, "13": 0, "14": 0, "15": 0}

    for i in train_y:
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


def getPCA(Xc, numComponents=5, dataset='Kochia'):
    """Reduce the number of components or channels using PCA"""
    newX = Xc.transpose((0, 3, 4, 2, 1))
    newX = np.reshape(newX, (-1, newX.shape[3]))
    pcaC = PCA(n_components=numComponents, whiten=True)
    newX = pcaC.fit_transform(newX)
    newX = np.reshape(newX, (Xc.shape[0], Xc.shape[3], Xc.shape[3], numComponents, Xc.shape[1]))
    newX = newX.transpose((0, 4, 3, 1, 2))
    # Save pca transformation
    file = dataset + "//results//PCA_transformations//pca_" + str(numComponents)
    with open(file, 'wb') as f:
        pickle.dump(pcaC, f)
    return newX


def applyPCA(Xc, numComponents=5, dataset='Kochia'):
    """Apply previously calculated PCA transformation"""
    # Load pca transformation
    file = dataset + "//results//PCA_transformations//pca_" + str(numComponents)
    with open(file, 'rb') as f:
        pcaC = pickle.load(f)
    newX = Xc.transpose((0, 3, 4, 2, 1))
    newX = np.reshape(newX, (-1, newX.shape[3]))
    print("Explained variance in the training set:")
    print(np.sum(pcaC.explained_variance_ratio_))
    print("Explained variance in the test set:")
    print(r2_score(newX, pcaC.inverse_transform(pcaC.transform(newX)), multioutput='variance_weighted'))
    newX = pcaC.transform(newX)
    newX = np.reshape(newX, (Xc.shape[0], Xc.shape[3], Xc.shape[3], numComponents, Xc.shape[1]))
    newX = newX.transpose((0, 4, 3, 1, 2))
    return newX


def getPLS(Xc, yc, numComponents=5, dataset='Kochia'):
    """Reduce the number of components or channels using PLS"""
    newX = Xc.transpose((0, 3, 4, 2, 1))
    newX = np.reshape(newX, (-1, newX.shape[3]))
    # Inflate the labels so that len(newY) matches len(newX)
    newY = []
    for yi in yc:
        for rep in range(int(len(newX) / len(yc))):
            newY.append(yi)
    PLS_transform = PLSRegression(n_components=numComponents)
    PLS_transform.fit(newX, newY)
    newX = PLS_transform.transform(newX)
    # print("Explained variance in the training set:")
    # print(np.sum(PLS_transform.explained_variance_ratio_))
    newX = np.reshape(newX, (Xc.shape[0], Xc.shape[3], Xc.shape[3], numComponents, Xc.shape[1]))
    newX = newX.transpose((0, 4, 3, 1, 2))
    # Save pca transformation
    file = dataset + "//results//PLS_transformations//pls_" + str(numComponents)
    with open(file, 'wb') as f:
        pickle.dump(PLS_transform, f)
    return newX


def applyPLS(Xc, numComponents=5, dataset='Kochia'):
    """Apply previously calculated PCA transformation"""
    # Load pca transformation
    file = dataset + "//results//PLS_transformations//pls_" + str(numComponents)
    with open(file, 'rb') as f:
        PLSC = pickle.load(f)
    newX = Xc.transpose((0, 3, 4, 2, 1))
    newX = np.reshape(newX, (-1, newX.shape[3]))
    newX = PLSC.transform(newX)
    newX = np.reshape(newX, (Xc.shape[0], Xc.shape[3], Xc.shape[3], numComponents, Xc.shape[1]))
    newX = newX.transpose((0, 4, 3, 1, 2))
    return newX


def permutationTest(scores1, scores2):
    """Perform a permutation paired t-test"""
    # Calculate the real differences between two samples
    d = np.array(scores1) - np.array(scores2)
    n = len(scores1)
    # Sets the number of iterations
    reps = 1000
    # Create permutation matrix
    x = 1 - 2 * binom(1, .5, 10 * reps)
    x.shape = (reps, n)
    # Apply permutations
    sim = x * d
    # Get distribution of simulated t-values
    dist = sim.mean(axis=1) / (sim.std(axis=1, ddof=1) / np.sqrt(n))
    # Get real t-value
    observed_ts = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    # Return 2-sided p-value
    return np.mean(np.abs(dist) >= np.abs(observed_ts))
