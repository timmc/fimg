import math
import sys

import numpy
import numpy.fft as fft
import skimage


def load_image_grayscale(src_path):
    image = skimage.io.imread(src_path)
    if len(image.shape) > 2:
        # A third dimension means more than one channel, so how many
        # channels did we get?
        if image.shape[2] == 3:
            image = skimage.color.rgb2gray(image)
        elif image.shape[2] == 4:
            # Flatten with black background, first
            image = skimage.color.rgba2rgb(image, background=(0, 0, 0))
            image = skimage.color.rgb2gray(image)
        else:
            raise Exception(f"Unknown image shape: {image.shape}")
    # Ensures intensity is represented as 0-255 int and not as float or whatever
    return skimage.util.img_as_ubyte(image)


def save_image_grayscale(dest_path, image_data):
    # Convert from floats back to unsigned bytes for skimage
    out_image_data = numpy.uint8(image_data)
    skimage.io.imsave(dest_path, out_image_data)


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


def rotate_phase(phase, frac):
    full_circle = 2 * math.pi
    return (phase + math.pi + (full_circle * frac)) % full_circle - math.pi


def main(src_path, dest_path, phase_const_str):
    orig_image = load_image_grayscale(src_path)
    orig_freq = spatial_to_freq(orig_image)
    amplitude, phase = freq_to_amp_phase(orig_freq)

    phase = phase * 0 + float(phase_const_str) * 2 * math.pi

    recomp_freq = amp_phase_to_freq(amplitude, phase)
    save_image_grayscale(dest_path, freq_to_spatial(recomp_freq, orig_image.shape))


if __name__ == '__main__':
    main(*sys.argv[1:])
