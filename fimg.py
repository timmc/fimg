from functools import wraps
import math
from random import random
from textwrap import dedent
import sys

import click
import imageio
import numpy as np
import skimage

#===========#
# Utilities #
#===========#


def spatial_to_freq(spatial):
    return np.fft.fft2(spatial)


def freq_to_spatial(freq):
    return np.fft.irfft2(freq, freq.shape)


def freq_to_amp_phase(freq):
    return abs(freq), np.angle(freq)


def amp_phase_to_freq(amp, phase):
    return amp * math.e ** (1j * phase)  # r * e^(i * theta)


def rescale(arr, in_low, in_high, out_low, out_high):
    return (arr - in_low) / (in_high - in_low) * (out_high - out_low) + out_low


#==========#
# CLI crud #
#==========#

@click.group()
@click.argument('src', type=click.File(mode='rb'))
@click.argument('dest', type=click.File(mode='wb', lazy=True))
@click.option(
    '--out-format', type=click.Choice(['jpg', 'png']),
    # Default to JPG because PNGs of really speckly images can be huge
    default='jpg',
)
@click.option(
    '--out-of-range', '-o',
    type=click.Choice(['mod', 'clip','lin-cent',]), default='mod',
    help=dedent("""
    Global: How to handle out-of-range values when saving.

    `mod` takes the modulo of pixel intensity against 255, resulting
    in sharp inversion bands, but no change to in-range pixels.

    `clip` caps high values to 255 and low to 0, resulting in
    completely black or white patches.

    `lin-cent` is a linear rescale based on percentiles:
    If the low or high percentile (defined by --clip-centile) is
    out of range, do a linear rescale with the effect of pulling that
    percentile to 0 or 255, then clip anything still out of range.
    """),
)
@click.option(
    '--clip-centile',
    help=("Global: When using lin-cent for out-of-range inputs, "
          "use this value (and 100 minus this value) for the percentiles "
          "to be pulled into range."),
    type=float, default='10',
)
@click.option(
    '--rescale-channels',
    type=click.Choice(['together', 'separately']), default='together',
    help=(
        "Global: Whether rescaling due to out-of-range intensities, "
        "whether to scale all the color components of an image together "
        "versus on different scales (based on their individual ranges)."
    ),
)
@click.pass_context
def cli(ctx, **kwargs):
    ctx.obj = kwargs


def load_image_channels(src_file):
    src_image = imageio.imread(src_file)
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

    return src_channels


def linear_centile_rescale(data, clip_centile):
    """
    Rescale data in an array based on low and high percentiles.
    """
    ptile_lo = np.percentile(data, clip_centile)
    ptile_hi = np.percentile(data, 100 - clip_centile)
    rescale_lo = min(ptile_lo, 0)
    rescale_hi = max(255, ptile_hi)
    data = rescale(data, rescale_lo, rescale_hi, 0, 255)
    data = np.clip(data, 0, 255)
    return data


def write_image(channels, dest_file, /, out_of_range, clip_centile, rescale_channels, out_format):
    if np.amin(channels) < 0 or np.amax(channels) > 255:
        if out_of_range == 'mod':
            channels = np.mod(channels, 255)
        elif out_of_range == 'clip':
            channels = np.clip(channels, 0, 255)
        elif out_of_range == 'lin-cent':
            if rescale_channels == 'together':
                channels = linear_centile_rescale(channels, clip_centile)
            else:
                channels = np.array([linear_centile_rescale(ch, clip_centile) for ch in channels])
        else:
            raise Exception(f"Unknown out-of-range option: {out_of_range}")

    # Put the channel dimension back at the end
    out_image = channels.transpose((1, 2, 0))

    # Convert from floats back to unsigned bytes for skimage
    out_image = np.uint8(out_image)
    imageio.imwrite(dest_file, out_image, format=out_format)


# A whole bunch of awful, nested decorators that act as transformation
# lenses so that commands can just focus on the aspect of the data
# they want.

def operate_on_image(xfunc):
    """
    Wraps a function that handles single-channel image spatial data.

    Outermost transform decorator that handles image loading and saving.
    This one asks for the Click context to get the src/dest paths.
    """
    @wraps(xfunc)
    @click.pass_context
    def on_image_wrapper(ctx, *cmd_args, **cmd_kwargs):
        src_channels = load_image_channels(ctx.obj['src'])
        out_channels = np.array([
            xfunc(channel, *cmd_args, **cmd_kwargs)
            for channel in src_channels
        ])
        write_image(
            out_channels, ctx.obj['dest'],
            **{k:ctx.obj[k] for k in ['out_of_range', 'clip_centile', 'rescale_channels', 'out_format']}
        )

    return on_image_wrapper


def operate_on_freq(cmd):
    """
    Wrap a function that transforms complex frequency data so it takes image spatial data.
    """
    @wraps(cmd)
    @operate_on_image
    def on_freq_wrapper(image, *cmd_args, **cmd_kwargs):
        src_freq = spatial_to_freq(image)
        out_freq = cmd(src_freq, *cmd_args, **cmd_kwargs)
        assert out_freq is not None
        return freq_to_spatial(out_freq)
    return on_freq_wrapper


def operate_on_amp(cmd):
    """
    Wrap a function that transforms amplitude so it takes complex frequency data.
    """
    @wraps(cmd)
    @operate_on_freq
    def on_amp_wrapper(freq, *cmd_args, **cmd_kwargs):
        amp, phase = freq_to_amp_phase(freq)
        return amp_phase_to_freq(cmd(amp, *cmd_args, **cmd_kwargs), phase)
    return on_amp_wrapper


def operate_on_phase(cmd):
    """
    Wrap a function that transforms phase so it takes complex frequency data.
    """
    @wraps(cmd)
    @operate_on_freq
    def on_phase_wrapper(freq, *cmd_args, **cmd_kwargs):
        amp, phase = freq_to_amp_phase(freq)
        return amp_phase_to_freq(amp, cmd(phase, *cmd_args, **cmd_kwargs))
    return on_phase_wrapper


def opt_angles(cmd):
    """
    Add --radians option, with alternatives --degrees and --turns.

    Performs validation and conversion first.
    """
    @wraps(cmd)
    @click.option('--radians', '--rad', type=float)
    @click.option('--degrees', '--deg', type=float)
    @click.option('--turns', type=float)
    def opt_angles_wrapper(*cmd_args, radians, degrees, turns, **cmd_kwargs):
        angle_count = len([x for x in [radians, degrees, turns] if x is not None])
        if angle_count != 1:
            raise click.UsageError(
                "Exactly one of --radians, --degrees, or --turns "
                f"must be supplied (was {angle_count})"
            )

        if degrees is not None:
            radians = degrees / 180 * math.pi
        elif turns is not None:
            radians = 2 * math.pi * turns

        if radians is None:
            raise Exception("Early validation somehow failed to detect lack of rad, deg, or turn option")

        return cmd(*cmd_args, **dict(**cmd_kwargs, radians=radians))
    return opt_angles_wrapper


#==========#
# Commands #
#==========#

@cli.command('phase-shift')
@opt_angles
@operate_on_phase
def phase_shift(phase, radians):
    """
    Add the given angle to the phase. Effects are along the X axis.

    Previously known as phase_rotate_angle -- can be thought of as rotation in
    the complex plane, or shifting in the spatial plane.
    """
    # We could avoid the extra computation of separating amplitude and phase
    # by operating on freq directly and returning `freq * math.e ** (1j * radians)`
    # but this is just simpler to read.
    return phase + radians


@cli.command('plot-amp')
@operate_on_image
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


@cli.command('plot-phase')
@operate_on_image
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


@cli.command('roll-freq')
@click.option('--x', required=True, type=int)
@click.option('--y', required=True, type=int)
@operate_on_freq
def roll_freq(freq, x, y):
    return roll_xy(freq, x, y)


@cli.command('roll-amp')
@click.option('--x', required=True, type=int)
@click.option('--y', required=True, type=int)
@operate_on_amp
def roll_amp(amp, x, y):
    return roll_xy(amp, x, y)


@cli.command('roll-phase')
@click.option('--x', required=True, type=int)
@click.option('--y', required=True, type=int)
@operate_on_phase
def roll_phase(phase, x, y):
    return roll_xy(phase, x, y)


@cli.command('const-amp')
@click.option('--value', required=True, type=float)
@operate_on_amp
def const_amp(amp, value):
    return amp * 0 + value


@cli.command('const-phase')
@opt_angles
@operate_on_phase
def const_phase(phase, radians):
    return phase * 0 + radians


def speckle(arr):
    # TODO Accept RNG seed in CLI
    rng = np.random.default_rng()
    return arr * rng.random(arr.shape)


@cli.command('speckle-amp')
@operate_on_amp
def speckle_amp(amp):
    return speckle(amp)


@cli.command('speckle-phase')
@operate_on_phase
def speckle_phase(phase):
    return speckle(phase)


def filter_first_axis(image, mode, bounds):
    freq = np.fft.fft2(image, axes=(0, 1))
    (start, until) = bounds
    if mode == 'reject':
        freq[start:until, :] = 0
    elif mode == 'pass':
        freq[:start, :] = 0
        freq[until:, :] = 0
    else:
        raise Exception(f"Unexpected mode: {mode}")
    image = np.fft.irfft2(freq, s=freq.shape, axes=(0, 1))
    return image


@cli.command('band-filter')
@click.argument('mode', required=True, type=click.Choice(['pass', 'reject']))
@click.option('--x', nargs=2, type=int)
@click.option('--y', nargs=2, type=int)
@operate_on_image
def band_filter(image, mode, x, y):
    """
    Pass or reject frequencies in a certain band.

    Each of --x and --y takes a pair [from, until).
    """
    # Handle x and y axes separately, performing the FFT on that axis
    # first in each case.

    if y is not None:
        image = filter_first_axis(image, mode, y)

    if x is not None:
        image = np.transpose(image)
        image = filter_first_axis(image, mode, x)
        image = np.transpose(image)

    return image


#=============#
# Entry point #
#=============#

if __name__ == '__main__':
    cli(obj={})
