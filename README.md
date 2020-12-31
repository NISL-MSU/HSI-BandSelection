# Hyperspectral Band Selection: Greedy Spectral Selection 

## Description

In recent years, Hyperspectral Imaging (HSI) has become a powerful source for reliable data in applications such as agriculture, 
remote sensing, and biomedicine. However, hyperspectral images are highly data dense and often benefit from methods to reduce the 
number of spectral bands while retaining the most useful information for a specific application. Here, we present a novel band selection 
method to determine relevant wavelengths obtained from an HSI system in the context of image classification. Our approach consists 
of two main steps: the first, called **Inter-Band Redundancy Analysis** (IBRA), finds relevant spectral bands based on a collinearity analysis between a band and its neighbors in 
order to determine the distance we need to move away from it to find sufficiently distinct bands. The analysis of the distribution 
of this metric across the spectrum helps to remove redundant bands and dramatically reduces the search space. The second, called **Greedy Spectral Selection** (GSS) uses the 
reduced set of bands and selects the top-*k* bands, where *k* is the desired number of bands, according to their information entropy 
values; then, the band that presents the most severe indication of multicollinearity is removed from the current selection and the 
next available pre-selected band is considered if the classification performance improves. We present the classification results 
obtained from our method and compare them to other feature selection methods on two hyperspectral image datasets. Furthermore, we 
use the original hyperspectral data cube to simulate the process of using actual filters in a multispectral imager. We show that 
our method produces more suitable results for a multispectral sensor design. 

## Datasets

We used an in-greenhouse controlled HSI dataset of Kochia leaves in order to classify three different herbicide-resistance levels (herbicide-susceptible, dicamba-resistant, and glyphosate-resistant). 
A total of 76 images of kochia with varying spatial resolution and 300 spectral bands ranging from 387.12 to 1023.5 nm were captured. From these images, which were previously calibrated and converted to reflectance values, we manually extracted 6,316 25x25 pixel overlapping patches. The Kochia dataset can be downloaded from [here](https://montana.box.com/v/kochiadataset).

Furthermore, we used a well-known remote sensing HSI dataset: Indian Pines (IP), which can be downloaded from [here](http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat) or from this [repository](https://github.com/GiorgioMorales/HSI-BandSelection/tree/master/Data). 

## Usage

This repository contains the following scripts:

* `interBandRedundancy.py`: Executes both the pre-selection and final selection method for a desired number ofspectral bands.        
* `TrainSelection.py`: Class used to train and validate a classifier on the selected dataset type. The main parameters of this class are:
        
        *nbands: Desired number of bands.
        *method: Band selection method. Options: 'SSA', 'OCF', 'GA', 'PLS', 'FNGBS', 'FullSpec'.
        *classifier: Type of model used to train the classifiers. Options: 'CNN', 'SVM', 'RF'.
        *transform: Flag used to simulate Gaussian bandwidths.
        *average: Flag used to reduce the number of bands averaging consecutive bands.
        *batch_size: Size of the batch used for training.
        *epochs: Number of epochs used for training.
        *data: Name of the dataset. Options: 'Kochia', 'Avocado'
        *size: Percentage of the dataset used for the experiments.
        *selection: Load only the selected bands from the dataset
        *th: Optional index to add in the end of the generated files
        *median: If True, perform a median filtering on the spectral bands.
        
* `network.py`: Contains all the network architectures used in this work.
* `utils.py`: Additional methods used to transform the data and calculate the metrics.
