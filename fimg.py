import sys

import numpy
import numpy.fft as fft
import skimage


def load_image_grayscale():
    # TODO arbitrary source image
    # TODO use `from skimage.color import rgb2gray`
    image = skimage.data.camera()
    # Ensure intensity is represented as 0-255 int
    return skimage.util.img_as_ubyte(image)


def save_image_grayscale(image_data):
    # Convert from floats back to unsigned bytes for skimage
    out_image_data = numpy.uint8(image_data)
    # TODO remove hardcoded path (from here and .gitignore)
    skimage.io.imsave('out.png', out_image_data)


def main(roll_x, roll_y):
    roll_x = int(roll_x)
    roll_y = int(roll_y)

    orig = load_image_grayscale()

    fourier = fft.fft2(orig)
    fourier = numpy.roll(fourier, roll_y, (0,))
    fourier = numpy.roll(fourier, roll_x, (1,))
    image_i = fft.irfft2(fourier, orig.shape)

    save_image_grayscale(image_i)


if __name__ == '__main__':
    main(*sys.argv[1:])
