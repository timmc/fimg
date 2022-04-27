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


def roll(image, x, y):
    image = numpy.roll(image, y, (0,))
    image = numpy.roll(image, x, (1,))
    return image


def spatial_to_freq(spatial):
    return fft.fft2(spatial)


def freq_to_spatial(freq, shape):
    return fft.irfft2(freq, shape)


def freq_to_amp_phase(freq):
    amplitude = abs(freq)
    phase = numpy.angle(freq)
    return amplitude, phase


def amp_phase_to_freq(amplitude, phase):
    real = numpy.cos(phase) * amplitude
    imag = numpy.sin(phase) * amplitude
    return real + imag * complex(0, 1)


def main(roll_x, roll_y):
    orig_image = load_image_grayscale()
    orig_freq = spatial_to_freq(orig_image)
    amplitude, phase = freq_to_amp_phase(orig_freq)

#    amplitude = roll(amplitude, int(roll_x), int(roll_y))
    phase = roll(phase, int(roll_x), int(roll_y))

    recomp_freq = amp_phase_to_freq(amplitude, phase)
    save_image_grayscale(freq_to_spatial(recomp_freq, orig_image.shape))


if __name__ == '__main__':
    main(*sys.argv[1:])
