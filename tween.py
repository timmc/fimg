"""
Helper to generate a tweening sequence for when there's boring
stuff in the middle of an animation but frames should step through the
parameter space at a linear rate at the beginning and end.
"""

import math
import sys


def tanh_lin(x, compression, steepness):
    # 8 is a sufficiently high steepness to make tanh approach
    # linear at the endpoints. It can be higher to shorten the middle
    # and lengthen the beginning and end.
    return (math.tanh(steepness*(x - 1/2)) + 1)/2 * (compression - 1) + x


def rescaled(f, x, y2):
    """
    Given a function symmetric around (0.5,0.5), rescale to intercept (0,0) and (1,y2).
    """
    at_zero = f(0)
    return (f(x) - at_zero) / (y2 - 2 * at_zero)


if __name__ == '__main__':
    # E.g. $0 512 2 will generate values for a 256 frame animation
    # (512/2=256) including endpoints.
    width = int(sys.argv[1])  # Outputs integers in [0, width]
    compression = int(sys.argv[2])  # Time-compression, should divide width
    steepness = int(sys.argv[3])  # Relative compression of middle, 8+ is better

    f = lambda x: tanh_lin(x, compression, steepness)

    steps = width // compression
    for i in range(0, steps + 1):
        x = i/steps
        y = rescaled(f, x, compression)
        print(round(y * width))
