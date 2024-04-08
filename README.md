[![Paper](https://img.shields.io/badge/Paper-OpenAccess-b31b1b.svg)](https://www.mdpi.com/2072-4292/13/18/3649)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NISL-MSU/HSI-BandSelection/blob/master/HSIBandSelection.ipynb)


# Hyperspectral Band Selection for Multispectral Image Classification with Convolutional Networks

## Description

We present a novel band selection 
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

Furthermore, we used two well-known remote sensing HSI dataset: Indian Pines (IP) and
Salinas (SA), which can be downloaded from [here](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes) or from the "src/HSIBandSelection/Data" folder from this [repository](https://github.com/GiorgioMorales/HSI-BandSelection/tree/master/Data).

## Installation

The following libraries have to be installed:
* [Git](https://git-scm.com/download/) 
* [Pytorch](https://pytorch.org/)

To install the package, run `!pip install -q git+https://github.com/NISL-MSU/HSI-BandSelection` in the terminal. 

You can also try the package on [Google Colab](https://colab.research.google.com/github/NISL-MSU/HSI-BandSelection/blob/master/HSIBandSelection.ipynb).



## Usage

### Load your data

You can bring your own HSI classification dataset. Format the input data as a set of image data cubes of shape $(N, w, h, b)$, where $N$ is the number of data cubes, $w$ and $h$ are the width and the height of the cubes, and $b$ is the number of spectral bands. You could use the `createImageCubes` method, provided [here](https://github.com/NISL-MSU/HSI-BandSelection/blob/master/src/HSIBandSelection/readSAT.py#L47), as a reference to format your data. 

In this example, we will load the Indian Pines dataset, which is an image with shape `(145, 145, 200)`.

```python
from HSIBandSelection.readSAT import loadata, createImageCubes
X, Y = loadata(name='IP')
print('Initial image shape: ' + str(X.shape))

X, Y = createImageCubes(X, Y, window=5)
print('Processed dataset shape: ' + str(X.shape))
```

In this case, we loaded a HS image saved in our package. It doesn't matter where you bring the data from, you only need to provide the $X$ (input data) and $Y$ (target labels) matrices. In addition, assign your dataset a name; otherwise, it will be called `temp`. With these three elements, we create a data object:

```python
from HSIBandSelection.utils import Dataset
dataset = Dataset(train_x=X, train_y=Y, name='IP')
```

### Execute the Band Selection / Dimensionality Reduction Algorithm

We'll use the `SelectBands` class. **Parameters**:

*   `dataset`: utils.Dataset object
*   `method`: Method name. Options: 'IBRA', 'GSS' (IBRA+GSS), 'PCA' (IBRA+PCA), and 'PLS' (IBRA+PLS). *Default:* 'GSS'
*   `classifier`: Classifier type. Options: 'CNN' (if data is 2D), 'ANN', 'RF', 'SVM'. *Default:* 'CNN'
*   `nbands`: How many spectral bands you want to select or reduce to. *Default:* 5
*   `transform`: If True, the final selected bands will suffer a Gaussian transformation to simulate being a multispectral band. *Default:* False
*   `average`: If True, average consecutive bands to reduce the initial total # of bands to half. *Default:* False
*   `epochs`: Number of iterations used to train the NN models. *Default:* 150
*   `batch_size`: Batch size used to train the NN models. *Default:* 128
*   `scratch`: If True, execute the IBRA process from scratch and replace previously saved results. *Default:* True

```python
from HSIBandSelection.SelectBands import SelectBands
selector = SelectBands(dataset=dataset, method='GSS', nbands=5)
```

From the SelectBands class, we call the `run_selection` method. **Parameters**:

*   `init_vf`: Initial Variance Inflation Factor threshold (used for IBRA). *Default: 12*
*   `final_vf`: Final Variance Inflation Factor threshold (used for IBRA). *Default: 5*

**Return**:


If the selected method is IBRA:

*   `VIF_best`: The VIF threshold at which the best results were obtained
*   `IBRA_best`: The best pre-selected bands using Iner-band redundancy
*   `stats_best`: The best performance metric values obtained after 5x2 CV using the selected bands

If the selected method is GSS:

*   `VIF_best`: The VIF threshold at which the best results were obtained 
*   `IBRA_best`: The best pre-selected bands using Iner-band redundancy
*   `GSS_best`: The best combination of bands obtained using GSS
*   `stats_best`: The best performance metric values obtained after 5x2 CV using the selected bands

If the selected method is PCA or PLS:

*   `VIF_best`: The VIF threshold at which the best results were obtained 
*   `IBRA_best`: The best pre-selected bands using Iner-band redundancy
*   `reduced_dataset`: The reduced dataset after applying PCA or PLS to the pre-selected bands
*   `stats_best`: The best performance metric values obtained after 5x2 CV using the reduced bands

```python
VIF_best, IBRA_best, GSS_best, stats_best = selector.run_selection()
print('The best metrics were obtained using a VIF value of {}'.format(VIF_best))
print('The pre-selected bands obtained by IBRA wew {}'.format(IBRA_best))
print('The pre-selected bands obtained by IBRA+GSS wew {}'.format(GSS_best))
print('The best classification metrics were as follows:')
print(stats_best)
```





In addition, this repository contains the following scripts:

* `interBandRedundancy.py`: Executes the IBRA algorithm.        
* `TrainSelection.py`: Class used to train and validate a classifier on the selected dataset type. The main parameters of this class are:        
        
        * `method`: Band selection method. Options: 'GSS', 'OCF', 'GA', 'PLS', 'FNGBS', 'FullSpec', 'Compressed.
        * `classifier`: Type of model used to train the classifiers. Options: 'CNN', 'SVM', 'RF'.
        * `transform`: Flag used to simulate Gaussian bandwidths.
        * `batch_size`: Size of the batch used for training.
        * `epochs`: Number of epochs used for training.
        * `dataset`: A utils.Dataset object
        * `size`: Percentage of the dataset used for the experiments.
        * `th`: Optional index to add in the end of the generated files
        * `pca`: If True, we use the IBRA method to form a set of candidate bands and then we reduce the number of using PCA.
        * `pls`: If True, we use the IBRA method to form a set of candidate bands and then we reduce the number of using LDA.
        
* `utils.py`: Additional methods used to transform the data and calculate the metrics.   

* `ClassificationStrategy/network.py`: Contains all the network architectures used in this work.  
* `ClassificationStrategy/ModelStrategy.py`: Interface class used to train and validate a classifier on the selected dataset type. The parameters of the constructor of this class are:
        
        *classifier: Type of classifier. Options: 'CNN', 'ANN', 'SVM', or 'RF'.
        *data: Type of data. Options: 'Kochia', 'Avocado', 'IP', or 'SA'.
        *device: Type of device used for training (Used for the CNN).
        *nbands: Number of selected spectral ban.
        *windowSize: Window size (Used for the CNN).
        *train_y: Target data.
        *classes: Number of classes.

   The parameters of the `trainFold` method are:
        
        *trainx: Training set.
        *train_y: Target data of the entire dataset (training + validation sets).
        *train: List of training indexes
        *batch_size: Size of the mini-batch (Used for the CNN).
        *epochs: Number of epochs used to train a CNN.
        *valx: Validation set.
        *test: List of test indexes
        *means: Mean of each spectral band calculated in the training set.
        *stds: Standard deviation of each spectral band calculated in the training set.
        *filepath: Path used to store the trained model.
        *printProc: If True, prints all the training process        

   The parameters of the `evaluateFold` method are:
        
        *valx: Validation set.
        *train_y: Target data.
        *test: List of test indexes
        *means: Mean of each spectral band calculated in the training set.
        *stds: Standard deviation of each spectral band calculated in the training set.
        *batch_size: Size of the mini-batch (Used for the CNN).
   
### Results: Metrics Comparison

<p align="center">
  <img src="Figures/Comparison1.jpg" alt="alt text" width="400">
</p>

<p align="center">
  <img src="Figures/Comparison2.jpg" alt="alt text" width="400">
</p>

# Citation
Use this Bibtex to cite this repository

```
@Article{rs13183649,
AUTHOR = {Morales, Giorgio and Sheppard, John W. and Logan, Riley D. and Shaw, Joseph A.},
TITLE = {Hyperspectral Dimensionality Reduction Based on Inter-Band Redundancy Analysis and Greedy Spectral Selection},
JOURNAL = {Remote Sensing},
VOLUME = {13},
YEAR = {2021},
NUMBER = {18},
ARTICLE-NUMBER = {3649},
URL = {https://www.mdpi.com/2072-4292/13/18/3649},
ISSN = {2072-4292},
DOI = {10.3390/rs13183649}
}
```


```
@INPROCEEDINGS{ijcnn2021,
  author={Morales, Giorgio and Sheppard, John and Logan, Riley and Shaw, Joseph},
  booktitle={2021 International Joint Conference on Neural Networks (IJCNN)}, 
  title={Hyperspectral Band Selection for Multispectral Image Classification with Convolutional Networks}, 
  year={2021},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/IJCNN52387.2021.9533700}}
}
```
