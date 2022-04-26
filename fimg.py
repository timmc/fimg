import sys

import numpy
import numpy.fft as fft
import skimage


def main():
    # TODO arbitrary source image
    # TODO use `from skimage.color import rgb2gray`
    image = skimage.data.camera()
    # Ensure intensity is represented as 0-255 int
    image = skimage.util.img_as_ubyte(image)

    
    fourier = fft.fft2(image)
    image_i = fft.irfft2(fourier, image.shape)


    # Convert from floats back to unsigned bytes for skimage
    out_image_data = numpy.uint8(image_i)
    # TODO remove hardcoded path (from here and .gitignore)
    skimage.io.imsave('out.png', out_image_data)


if __name__ == '__main__':
    main(*sys.argv[1:])
