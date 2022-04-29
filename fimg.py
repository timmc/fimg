from functools import wraps
from inspect import signature
import math
from random import random
import sys

import click
import numpy as np
import skimage

#===========#
# Utilities #
#===========#


def spatial_to_freq(spatial):
    return np.fft.fft2(spatial)


def freq_to_spatial(freq, shape):
    return np.fft.irfft2(freq, shape)


def freq_to_amp_phase(freq):
    return abs(freq), np.angle(freq)


def amp_phase_to_freq(amp, phase):
    return amp * (np.cos(phase) + np.sin(phase) * 1j)


def rescale(arr, in_low, in_high, out_low, out_high):
    return (arr - in_low) / (in_high - in_low) * (out_high - out_low) + out_low


#==========#
# CLI crud #
#==========#

@click.group()
@click.argument('src_path', type=click.Path(exists=True))
@click.argument('dest_path', type=click.Path())
@click.option(
    '--out-of-range',
    type=click.Choice([
        'mod', 'clip',
        'percentile-pull-clip',
    ]),
    default='mod',
)
@click.option(
    '--clip-percentile',
    help=("When using percentile-pull-clip for out-of-range inputs, "
          "use this value and 100 minus this value for the percentiles "
          "to be pulled into range."),
    type=float, default='10',
)
@click.pass_context
def cli(ctx, **kwargs):
    ctx.obj = kwargs

# A whole bunch of awful, nested decorators that act as transformation
# lenses so that commands can just focus on the aspect of the data
# they want.

def xform_image(xfunc):
    """
    Wraps a function that handles single-channel image spatial data.

    Outermost transform decorator that handles image loading and saving.
    This one asks for the Click context to get the src/dest paths.
    """
    @wraps(xfunc)
    @click.pass_context
    def wrapped(ctx, *cmd_args, **cmd_kwargs):
        src_path = ctx.obj['src_path']
        dest_path = ctx.obj['dest_path']

        src_image = skimage.io.imread(src_path)
        # Ensures intensity is represented as 0-255 int and not as float or whatever
        src_image = skimage.util.img_as_ubyte(src_image)

        if len(src_image.shape) == 2:
            # If it's a single-channel image, it comes in with only
            # two dimensions.
            src_channels = src_image[np.newaxis, :, :]
        elif len(src_image.shape) == 3:
            if src_image.shape[2] == 4:
                # Probably RGBA -- apply the alpha channel first.
                src_image = skimage.color.rgba2rgb(src_image)
            elif src_image.shape[2] != 3:
                raise Exception(f"Unexpected number of channels: {src_image.shape[2]}")
            # Multi-channel images have the channels as a tail
            # dimension; bring it up front so we can iterate over the
            # channels.
            src_channels = src_image.transpose((2, 0, 1))
        else:
            raise Exception(f"Unexpected image shape: {src_image.shape}")

        out_channels = np.array([
            xfunc(channel, *cmd_args, **cmd_kwargs)
            for channel in src_channels
        ])

        # Put the channel dimension back at the end
        out_image = out_channels.transpose((1, 2, 0))

        if np.amin(out_image) < 0 or np.amax(out_image) > 255:
            oor = ctx.obj['out_of_range']
            if oor == 'mod':
                out_image = np.mod(out_image, 255)
            elif oor == 'clip':
                out_image = np.clip(out_image, 0, 255)
            elif oor == 'percentile-pull-clip':
                # Linear transform pulling low and high
                # percentiles towards [0, 255] unless they're already
                # inside it; then clip as needed.
                ptile_lo = np.percentile(out_image, ctx.obj['clip_percentile'])
                ptile_hi = np.percentile(out_image, 100 - ctx.obj['clip_percentile'])
                rescale_lo = min(ptile_lo, 0)
                rescale_hi = max(255, ptile_hi)
                out_image = rescale(out_image, rescale_lo, rescale_hi, 0, 255)
                out_image = np.clip(out_image, 0, 255)
            else:
                raise Exception(f"Unknown out-of-range option: {oor}")

        # Convert from floats back to unsigned bytes for skimage
        out_image = np.uint8(out_image)
        skimage.io.imsave(dest_path, out_image)
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
    amp = np.fft.fftshift(amp)
    # There's a huge range of variation in amplitude, so use a log transform
    amp += -np.amin(amp) + 1.1
    amp = np.log2(amp)
    # Then just rescale to full intensity range
    return rescale(amp, np.amin(amp), np.amax(amp), 0, 255)


@cli.command('plot_phase')
@xform_image
def plot_phase(image):
    """Render phase as spatial image data."""
    freq = spatial_to_freq(image)
    _amp, phase = freq_to_amp_phase(freq)

    # Roll by 1/2 along each axis to bring the low frequencies to the center
    phase = np.fft.fftshift(phase)
    # Phase is in radians, so just bring it to range and rescale it.
    phase = phase % (2 * math.pi)
    return rescale(phase, 0, 2 * math.pi, 0, 255)


def roll_xy(arr, x, y):
    arr = np.roll(arr, y, (0,))
    arr = np.roll(arr, x, (1,))
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


def speckle(arr):
    # TODO Accept RNG seed in CLI
    rng = np.random.default_rng()
    return arr * rng.random(arr.shape)


@cli.command('speckle_amp')
@xform_amp
def speckle_amp(amp):
    return speckle(amp)


@cli.command('speckle_phase')
@xform_phase
def speckle_phase(phase):
    return speckle(phase)


#=============#
# Entry point #
#=============#

if __name__ == '__main__':
    cli(obj={})
