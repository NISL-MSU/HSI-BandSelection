import torch
import copy
from torch import reshape
import numpy as np


def LRP(model, XV, device):
    # Get the list of layers of the network
    layers = [module for module in model.modules() if not isinstance(module, torch.nn.Sequential)][1:]

    # Get the relevances for each image in the set XV
    R = np.zeros((XV.shape[0], XV.shape[2]))
    for i, im in enumerate(XV):
        R[i, :] = LRP_individual(layers, reshape(im, (1, im.shape[0], im.shape[1], im.shape[2], im.shape[3])), device)
        maxind = np.argsort(R[i, :])[-6:]
        R[i, :] = R[i, :] * 0
        R[i, :][maxind] = 1

    sumT = np.zeros((XV.shape[2],))
    for i in range(XV.shape[2]):
        sumT[i] = np.sum(R[:, i])


def LRP_individual(layers, X, device):
    # Propagate the input
    L = len(layers)
    A = [X] + [X] * L  # Create a list to store the activation produced by each layer
    for layer in range(L):
        # After the 4th and 17th layers, we should reshape the tensor
        if layer == 4:
            A[layer] = reshape(A[layer], (A[layer].shape[0], A[layer].shape[2] * 16,
                                          A[layer].shape[3], A[layer].shape[4]))
        elif layer == 17:
            A[layer] = reshape(A[layer], (A[layer].shape[0], A[layer].shape[1]))

        A[layer + 1] = layers[layer].forward(A[layer])

    # Get the relevance of the last layer using the highest classification score of the top layer
    T = A[-1].cpu().detach().numpy().tolist()[0]
    index = T.index(max(T))
    T = np.abs(np.array(T)) * 0
    T[index] = 1
    T = torch.FloatTensor(T)
    R = [T.data] * L + [(A[-1].cpu() * T).data + 1e-6]
    if T[1] != 1:
        return np.zeros((18, ))

    # Propagation procedure from the top-layer towards the lower layers
    for layer in range(0, L)[::-1]:

        if isinstance(layers[layer], torch.nn.Conv2d) or isinstance(layers[layer], torch.nn.Conv3d) \
                or isinstance(layers[layer], torch.nn.AvgPool2d) or isinstance(layers[layer], torch.nn.Linear):

            if 0 < layer <= 13:  # Gamma rule (LRP-gamma)
                rho = lambda p: p + 0.25 * p.clamp(min=0)
                incr = lambda zd: zd + 1e-9
            else:  # Basic rule (LRP-0)
                rho = lambda p: p
                incr = lambda zd: zd + 1e-9

            A[layer] = A[layer].data.requires_grad_(True)
            z = incr(newlayer(layers[layer], rho).forward(A[layer]))  # step 1
            s = (R[layer + 1].to(device) / z).data  # step 2
            (z * s).sum().backward()
            c = A[layer].grad  # step 3
            R[layer] = (A[layer] * c).cpu().data  # step 4

            if layer == 17:
                R[layer] = reshape(R[layer], (R[layer].shape[0], R[layer].shape[1], 1, 1))
            elif layer == 4:
                R[layer] = reshape(R[layer], (R[layer].shape[0], 16, int(R[layer].shape[1] / 16), R[layer].shape[2],
                                              R[layer].shape[3]))
        else:
            R[layer] = R[layer + 1]

    minv = np.min(np.min(np.min(R[0].data.numpy(), axis=3), axis=3), axis=2)
    maxv = np.max(np.max(np.max(R[0].data.numpy(), axis=3), axis=3), axis=2)
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 6)
    count = 0
    im = None
    for i in range(3):
        for j in range(6):
            im = axs[i, j].imshow(R[0].data.numpy()[0, 0, count, :, :], vmin=minv, vmax=maxv)
            count += 1
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    # Calculate relevances per band as the sum of relevances of all the pixels of a band
    return np.sum(np.sum(R[0].data.numpy(), axis=3), axis=3).reshape(X.shape[2])


def newlayer(layer, g):
    """Clone a layer and pass its parameters through the function g."""
    layer = copy.deepcopy(layer)
    layer.weight = torch.nn.Parameter(g(layer.weight))
    layer.bias = torch.nn.Parameter(g(layer.bias))
    return layer
