import numpy as np
import matplotlib.pyplot as plt

def grad_energy(img, sigma = 3, rescale=255):
    """
    Compute the gradient magnitude of an image by doing
    1D convolutions with the derivative of a Gaussian
    
    Parameters
    ----------
    img: ndarray(M, N, 3)
        A color image
    sigma: float
        Width of Gaussian to use for filter
    rescale: float
        Amount by which to rescale the gradient
        
    Returns
    -------
    ndarray(M, N): Gradient Image
    """
    from scipy.signal import convolve2d
    I = 0.2125*img[:, :, 0] + 0.7154*img[:, :, 1] + 0.0721*img[:, :, 2]
    I = I/255
    N = int(sigma*6+1)
    t = np.linspace(-3*sigma, 3*sigma, N)
    dgauss = -t*np.exp(-t**2/(2*sigma**2))
    IDx = convolve2d(I, dgauss[None, :], mode='same')
    IDy = convolve2d(I, dgauss[:, None], mode='same')
    GradMag = np.sqrt(IDx**2 + IDy**2)
    return rescale*GradMag

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
    X[:, 1] = np.array(seam, dtype=int)
    plt.plot(X[:, 1], X[:, 0], 'r')

def read_image(path):
    """
    A wrapper around matplotlib's image loader that deals with
    images that are grayscale or which have an alpha channel

    Parameters
    ----------
    path: string
        Path to file
    
    Returns
    -------
    ndarray(M, N, 3)
        An RGB color image in the range [0, 255]
    """
    img = plt.imread(path)
    if np.issubdtype(img.dtype, np.integer):
        img = np.array(img, dtype=float)/255
    if len(img.shape) == 3:
        if img.shape[1] > 3:
            # Cut off alpha channel
            img = img[:, :, 0:3]
    if img.size == img.shape[0]*img.shape[1]:
        # Grayscale, convert to rgb
        img = np.concatenate((img[:, :, None], img[:, :, None], img[:, :, None]), axis=2)
    return img

if __name__ == '__main__':
    img = read_image("LivingRoom.jpg")
    E = grad_energy(img)
    plt.figure(figsize=(10, 6))
    plt.imshow(E, cmap='magma')
    plt.colorbar()
    plt.savefig("Energy.png", bbox_inches='tight')
