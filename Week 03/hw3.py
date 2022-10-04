from re import I, X
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

# load the dataset from the provided .npy file, center it around
# the origin, and return it as a numpy array of floats.


def load_and_center_dataset(filename):
    output = np.load(filename)
    return (output - np.mean(output, axis=0))

#  calculate and return the covariance matrix of the dataset as a numpy
# matrix (d × d array).


def get_covariance(dataset):
    row = len(dataset)
    output = np.dot(np.transpose(dataset), dataset)
    return output / (row - 1)

# perform eigendecomposition on the covariance matrix S and return a diagonal matrix
# (numpy array) with the largest m eigenvalues on the diagonal in descending order,
# and a matrix (numpy array) with the corresponding eigenvectors as columns.


def get_eig(S, m):
    eigenValue, eigenVector = eigh(S, subset_by_index=[len(S) - m, len(S) - 1])
    index = eigenValue.argsort()[::-1]
    eigenValue = np.sort(eigenValue)[::-1]

    eigenValue = eigenValue[index]
    eigenValue = np.diagflat(eigenValue)

    eigenVector = eigenVector[:, index]

    return eigenValue, eigenVector

# similar to get_eig, but instead of returning the first m, return all eigenvalues and
# corresponding eigenvectors in a similar format that explain more than a prop proportion
# of the variance (specifically, please make sure the eigenvalues are returned in
# descending order)


def get_eig_prop(S, prop):
    eigenValue, eigenVector = eigh(S)
    index = eigenValue.argsort()[::-1]
    eigenValue = np.sort(eigenValue)[::-1]

    eigenValue = eigenValue[index]
    eigenValue = np.diagflat(eigenValue)
    eigenTotal = np.sum(eigenValue)

    for i in range(eigenValue.size):
        if (eigenValue[i] / eigenTotal <= prop):
            break

    eigenValue = eigenValue[0: i]
    eigenValue = np.diagflat(eigenValue)

    eigenVector = eigenVector[:, i]

    return eigenValue, eigenVector

# project each d*1 image into your m-dimensional subspace (spanned by
# m vectors of size d*1) and return the new representation as a d*1 numpy array


def project_image(image, U):
    return np.dot(image, np.dot(U, np.transpose(U)))

# use matplotlib to display a visual representation of the original image
# and the projected image side-by-side.


def display_image(orig, proj):
    # Reshape images to 32X32 and transpose them
    orig = np.transpose(np.reshape(orig, (32, 32)))
    proj = np.transpose(np.reshape(proj, (32, 32)))
    # Make figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # set title on each subplot
    ax1.set_title('Original')
    ax2.set_title('Projection')
    # put the imput images into figure with colorbar
    img1 = ax1.imshow(orig, aspect='equal')
    fig.colorbar(img1, ax=ax1)
    img2 = ax2.imshow(proj, aspect='equal')
    fig.colorbar(img2, ax=ax2)

    return plt.show()

# Test code
#################################################
# x = load_and_center_dataset("YaleB_32x32.npy")
# S = get_covariance(x)
# Lambda, U = get_eig(S, 2)
# projection = project_image(x[0], U)
# display_image(x[0], projection)
#################################################
