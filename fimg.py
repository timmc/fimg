import sys

import numpy.fft as fft
from skimage.util import img_as_ubyte
from skimage import data, io, filters

def main():
    # TODO arbitrary source image
    # TODO use `from skimage.color import rgb2gray`
    image = data.camera()
    # Ensure intensity is represented as 0-255 int
    image = img_as_ubyte(image)

    
    fourier = fft.fft2(image)
    fourier_shift = fft.fftshift(fourier)

    image_i = fft.irfft2(fourier, image.shape)


    # TODO remove hardcoded path (from here and .gitignore)
    io.imsave('out.png', image_i)


if __name__ == '__main__':
    main(*sys.argv[1:])
