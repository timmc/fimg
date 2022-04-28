from functools import wraps
from inspect import signature
import math
from random import random
import sys

import click
import numpy
import numpy.fft as fft
import skimage

#===========#
# Utilities #
#===========#


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

#==========#
# CLI crud #
#==========#

@click.group()
def cli():
    pass

# A whole bunch of awful, nested decorators that act as transformation
# lenses so that commands can just focus on the aspect of the data
# they want.

def xform_image(xfunc):
    """
    Wraps a function that handles image spatial data.

    Outermost transform decorator that handles image loading and saving.
    This one even adds its own Click arguments.
    """
    @wraps(xfunc)
    @click.argument('src_path', type=click.Path(exists=True))
    @click.argument('dest_path', type=click.Path())
    def wrapped(src_path, dest_path, *cmd_args, **cmd_kwargs):
        src_image = load_image_grayscale(src_path)
        out_image = xfunc(src_image, *cmd_args, **cmd_kwargs)
        save_image_grayscale(dest_path, out_image)
    return wrapped


def xform_freq(cmd):
    """
    Wrap a function that transforms complex frequency data so it takes image spatial data.
    """
    @wraps(cmd)
    @xform_image
    def wrapped(image, *cmd_args, **cmd_kwargs):
        src_freq = spatial_to_freq(image)
        out_freq = cmd(src_freq, *cmd_args, **cmd_kwargs)
        return freq_to_spatial(out_freq, image.shape)
    return wrapped


def xform_amp(cmd):
    """
    Wrap a function that transforms amplitude so it takes complex frequency data.
    """
    @wraps(cmd)
    @xform_freq
    def wrapped(freq, *cmd_args, **cmd_kwargs):
        amp, phase = freq_to_amp_phase(freq)
        return amp_phase_to_freq(cmd(amp, *cmd_args, **cmd_kwargs), phase)
    return wrapped


def xform_phase(cmd):
    """
    Wrap a function that transforms phase so it takes complex frequency data.
    """
    @wraps(cmd)
    @xform_freq
    def wrapped(freq, *cmd_args, **cmd_kwargs):
        amp, phase = freq_to_amp_phase(freq)
        return amp_phase_to_freq(amp, cmd(phase, *cmd_args, **cmd_kwargs))
    return wrapped

#==========#
# Commands #
#==========#

@cli.command('phase_rotate_angle')
@click.option('--circle-fraction', required=True, type=float)
@xform_phase
def rotate_phase(phase, circle_fraction):
    full_circle = 2 * math.pi
    return (phase + math.pi + (full_circle * circle_fraction)) % full_circle - math.pi


@cli.command('plot_amp')
@xform_image
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
    return (amp - lo) / (hi - lo) * 255


@cli.command('plot_phase')
@xform_image
def plot_phase(image):
    """Render phase as spatial image data."""
    freq = spatial_to_freq(image)
    _amp, phase = freq_to_amp_phase(freq)

    # Roll by 1/2 along each axis to bring the low frequencies to the center
    phase = fft.fftshift(phase)
    # Phase is in radians, so just bring it to range and rescale it.
    phase = phase % (2 * math.pi)
    return phase / (2 * math.pi) * 255


def roll_xy(arr, x, y):
    arr = numpy.roll(arr, y, (0,))
    arr = numpy.roll(arr, x, (1,))
    return arr


@cli.command('roll_freq')
@click.option('--x', required=True, type=int)
@click.option('--y', required=True, type=int)
@xform_freq
def roll_freq(freq, x, y):
    return roll_xy(freq, x, y)


@cli.command('roll_amp')
@click.option('--x', required=True, type=int)
@click.option('--y', required=True, type=int)
@xform_amp
def roll_amp(amp, x, y):
    return roll_xy(amp, x, y)


@cli.command('roll_phase')
@click.option('--x', required=True, type=int)
@click.option('--y', required=True, type=int)
@xform_phase
def roll_phase(phase, x, y):
    return roll_xy(phase, x, y)


@cli.command('const_amp')
@click.option('--value', required=True, type=float)
@xform_amp
def const_amp(amp, value):
    return amp * 0 + value


@cli.command('const_phase')
@click.option('--circle-fraction', required=True, type=float)
@xform_phase
def const_phase(phase, circle_fraction):
    return phase * 0 + circle_fraction * 2 * math.pi


def speckle(val):
    return val * random()


@cli.command('speckle_amp')
@xform_amp
def speckle_amp(amp):
    return numpy.vectorize(speckle)(amp)


@cli.command('speckle_phase')
@xform_phase
def speckle_phase(phase):
    return numpy.vectorize(speckle)(phase)


#=============#
# Entry point #
#=============#

if __name__ == '__main__':
    cli()
