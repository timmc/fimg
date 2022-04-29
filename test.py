import math
import unittest

import numpy as np

import fimg


class MathTests(unittest.TestCase):
    def test_freq_to_amp_phase(self):
        freq = np.asarray([5 + 6j])
        amp, phase = fimg.freq_to_amp_phase(freq)
        self.assertAlmostEqual(amp[0], math.sqrt(61), places=3)
        self.assertAlmostEqual(phase[0], 0.87606, places=3)
