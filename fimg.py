from functools import wraps
from inspect import signature
import math
from random import random
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


def spatial_to_freq(spatial):
    return fft.fft2(spatial)


def freq_to_spatial(freq, shape):
    return fft.irfft2(freq, shape)


def freq_to_amp_phase(freq):
    amp = abs(freq)
    phase = numpy.angle(freq)
    return amp, phase


def amp_phase_to_freq(amp, phase):
    real = numpy.cos(phase) * amp
    imag = numpy.sin(phase) * amp
    return real + imag * complex(0, 1)


def cli(*, name):
    def do_wrap(func):
        func.__cli__ = {
            'name': name,
            'arg_count': len(signature(func).parameters) - 1,
        }
        return func
    return do_wrap


def xform_freq(cmd):
    @wraps(cmd)
    def wrapped(image, *cmd_args):
        src_freq = spatial_to_freq(image)
        out_freq = cmd(src_freq, *cmd_args)
        return freq_to_spatial(out_freq, image.shape)
    return wrapped


def xform_amp(cmd):
    @wraps(cmd)
    @xform_freq
    def wrapped(freq, *cmd_args):
        amp, phase = freq_to_amp_phase(freq)
        return amp_phase_to_freq(cmd(amp, *cmd_args), phase)
    return wrapped


def xform_phase(cmd):
    @wraps(cmd)
    @xform_freq
    def wrapped(freq, *cmd_args):
        amp, phase = freq_to_amp_phase(freq)
        return amp_phase_to_freq(amp, cmd(phase, *cmd_args))
    return wrapped


@cli(name='phase_rotate_angle')
@xform_phase
def rotate_phase(phase, frac_str):
    frac = float(frac_str)
    full_circle = 2 * math.pi
    return (phase + math.pi + (full_circle * frac)) % full_circle - math.pi


@cli(name='plot_amp')
def plot_amp(image):
    """Render amplitude as spatial image data."""
    freq = spatial_to_freq(image)
    amp, _phase = freq_to_amp_phase(freq)

    # Roll by 1/2 along each axis to bring the low frequencies to the center
    amp = fft.fftshift(amp)
    # There's a huge range of variation in amplitude, so use a log transform
    amp = numpy.log2(amp)
    lo = numpy.min(amp)
    hi = numpy.max(amp)
    out = (amp - lo) / (hi - lo) * 255
    return out


def roll_xy(arr, x, y):
    arr = numpy.roll(arr, y, (0,))
    arr = numpy.roll(arr, x, (1,))
    return arr


@cli(name='roll_freq')
@xform_freq
def roll_freq(freq, x, y):
    return roll_xy(freq, int(x), int(y))


@cli(name='roll_amp')
@xform_amp
def roll_amp(amp, x, y):
    return roll_xy(amp, int(x), int(y))


@cli(name='roll_phase')
@xform_phase
def roll_phase(phase, x, y):
    return roll_xy(phase, int(x), int(y))


@cli(name='const_amp')
@xform_amp
def const_amp(amp, val):
    return amp * 0 + float(val)


@cli(name='const_phase')
@xform_phase
def const_phase(phase, frac):
    return phase * 0 + float(frac) * 2 * math.pi


def speckle(val):
    return val * random()


@cli(name='speckle_amp')
@xform_amp
def speckle_amp(amp):
    return numpy.vectorize(speckle)(amp)


@cli(name='speckle_phase')
@xform_phase
def speckle_phase(phase):
    return numpy.vectorize(speckle)(phase)


commands = [
    roll_freq, roll_amp, roll_phase,
    rotate_phase,
    const_amp, const_phase,
    speckle_amp, speckle_phase,
    plot_amp,
]


def print_available_commands():
    print("Commands available:")
    for c in commands:
        print(f"- {c.__cli__['name']}")


def main(src_path, dest_path, cmd, *cmd_args):
    try:
        cmd_f = next(f for f in commands if f.__cli__['name'] == cmd)
    except StopIteration:
        print(f"Unknown command: {cmd}")
        print()
        print_available_commands()
        sys.exit(1)

    if cmd_f.__cli__['arg_count'] != len(cmd_args):
        print("Wrong number of arguments for command; "
              f"expected {cmd_f.__cli__['arg_count']}, got {len(cmd_args)}")
        sys.exit(1)

    src_image = load_image_grayscale(src_path)
    out_image = cmd_f(src_image, *cmd_args)

    save_image_grayscale(dest_path, out_image)


if __name__ == '__main__':
    if len(sys.argv) >= 4:
        main(*sys.argv[1:])
    else:
        print("Expected arguments: <in-path> <out-path> <command> [...]")
        print()
        print_available_commands()
