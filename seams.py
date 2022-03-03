import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy import misc


def grad_energy(img, sigma = 3):
    """
    Compute the gradient magnitude of an image by doing
    1D convolutions with the derivative of a Gaussian
    
    Parameters
    ----------
    img: ndarray(M, N, 3)
        A color image
    sigma: float
        Width of Gaussian to use for filter
    
    Returns
    -------
    ndarray(M, N): Gradient Image
    """
    I = 0.2125*img[:, :, 0] + 0.7154*img[:, :, 1] + 0.0721*img[:, :, 2]
    I = I/255
    N = int(sigma*6+1)
    t = np.linspace(-3*sigma, 3*sigma, N)
    dgauss = -t*np.exp(-t**2/(2*sigma**2))
    IDx = convolve2d(I, dgauss[None, :], mode='same')
    IDy = convolve2d(I, dgauss[:, None], mode='same')
    GradMag = np.sqrt(IDx**2 + IDy**2)
    return GradMag


def plot_seam(img, seam):
    """
    Plot a seam on top of the image
    Parameters
    ----------
    I: ndarray(nrows, ncols, 3)
        An RGB image
    seam: ndarray(nrows, dtype=int)
        A list of column indices of the seam from
        top to bottom
    """
    plt.imshow(img)
    X = np.zeros((len(seam), 2))
    X[:, 0] = np.arange(len(seam))
    X[:, 1] = seam
    plt.plot(X[:, 1], X[:, 0], 'r')


if __name__ == '__main__':
    img = plt.imread("LivingRoom.jpg")
    G = grad_energy(img)
    plt.figure(figsize=(10, 6))
    plt.imshow(G, cmap='magma')
    plt.colorbar()
    plt.savefig("Energy.png", bbox_inches='tight')