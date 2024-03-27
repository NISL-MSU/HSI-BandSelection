import os
import torch
import pickle
import numpy as np
from scipy.signal import find_peaks
from src.HSIBandSelection.utils import load_predefined_data
from src.HSIBandSelection.TrainSelection import TrainSelection
from src.HSIBandSelection.InterBandRedundancy import InterBandRedundancy


if __name__ == '__main__':

    ##############################################################################
    # Parameters
    ##############################################################################
    data = 'IP'     # Specify the dataset to be analyzed
    method = 'GSS'      # Specify method name. Options: GSS, PCA (IBRA+PCA), and PLS (IBRA+PLS)
    classifier = 'CNN'  # Specify classifier type
    nbands = 5          # Specify the number of desired bands
    size = 100          # Percentage of the dataset used for the experiments
    transform = False   # If True, transform the bands to simulate multispectral bands
    average = True      # If True, average consecutive bands to reduce the total # of bands to half
    medianF = False     # If True, apply median filter to images for de-noising
    pca, pls = False, False
    if method == 'PCA':
        pca = True      # If True, IBRA forms a set of candidate bands and then we reduce the number of bands using PCA
    elif method == 'PLS':
        pls = True      # If True, IBRA forms a set of candidate bands and then we reduce the number of bands using LDA

    sizestr = ''
    if size != 100:
        sizestr = str(size)
    if data == 'IP' or data == 'PU':
        average = False

    # NN hyperparameters
    epochs = 130
    batch = 128
    if classifier == "ANN":
        epochs = 90
        batch = 2048  # 1024 for Kochia

    ##############################################################################
    # Load data
    ##############################################################################
    dataset = load_predefined_data(flag_average=average, normalization=True, median=medianF, data=data, printInf=True)

    ##############################################################################
    # Initialize IBRA
    ##############################################################################
    interB = InterBandRedundancy(dataset=dataset, printProcess=False)
    th = 12  # Initial VIF threshold

    ##############################################################################
    # Run IBRA+GSS, IBRA+PCA, or IBRA+PLS using different VIF values
    ##############################################################################
    for t in reversed(range(5, th + 1)):  # Test values from th to 5
        print("Testing VIF threshold: " + str(t))

        # Check if the analysis have been made before
        filepreselected = data + "//results//" + method + "//preselection_" + data + "_VIF" + str(t)
        filedistances = data + "//results//" + method + "//distances_" + data + "_VIF" + str(t)
        fileselected = data + "//results//" + method + "//" + str(nbands) + " bands//selection_" + data + sizestr + classifier + \
                       str(nbands) + "bands_VIF" + str(t) + ".txt"

        # If the folder does not exist, create it
        folder = data
        if not os.path.exists(folder):
            os.mkdir(folder)
            os.mkdir(folder + "//results//")
            os.mkdir(folder + "//results//" + method)
        folder = data + "//results//" + method
        if not os.path.exists(folder):
            os.mkdir(folder)

        if os.path.exists(filepreselected):
            with open(filepreselected, 'rb') as f:
                indexes = pickle.load(f)
        else:
            interB.setT(t)
            # Get the distribution of distances
            dist = interB.clusters()
            # Calculate local minima
            dist.insert(0, 100)     # Add high values at the beginning and the end so that initial
            dist.append(100)        # and final bands can be considered as local minima
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
        interB.plotSample(distances=filedistances)

        # Get the k-selected bands based on IE
        new_dataset = load_predefined_data(nbands=nbands, flag_average=average, method=method, transform=transform,
                                           data=data, selection=indexes, median=medianF, vifv=t, pca=pca, pls=pls)
        net = TrainSelection(method=method, classifier=classifier, batch_size=batch, pca=pca, pls=pls,
                             epochs=epochs, plot=False, selection=indexes, th=str(t), dataset=new_dataset, size=size)
        if method == 'GSS':
            index, entr = net.selection(select=nbands)
            # Save selected bands as txt file
            with open(fileselected, 'w') as x_file:
                x_file.write(str(index))
            # Save scores of each of the bands
            with open(data + "//results//" + method + "//bandScores_" + data + sizestr + classifier + "_VIF" + str(t), 'wb') as fi:
                pickle.dump(entr, fi)
        else:
            if pca:
                print("Applying PCA over the IBRA-preselected bands and training a classifier")
            else:
                print("Applying PLS over the IBRA-preselected bands and training a classifier")

        # Train selected bands if the selected set of bands was not trained before
        if not os.path.exists(data + "//results//" + method + "//" + str(nbands) + " bands//classification_report5x2_" + sizestr +
                              classifier + method + str(nbands) + data + str(t) + ".txt"):
            np.random.seed(7)  # Re-Initialize seed to get reproducible results
            torch.manual_seed(7)
            torch.cuda.manual_seed(7)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            net = TrainSelection(method=method, classifier=classifier, batch_size=batch, pca=pca, pls=pls,
                                 epochs=epochs, plot=False, selection=indexes, th=str(t), dataset=new_dataset, size=size)
            net.train()
            # net.validate()  # Store the evaluation metrics
