"""
This package provides a collection of classes and functions for analyzing GADGET2 data.
"""


from .generate_files import generate_files
from . import utils

__all__ = ['generate_files', 'utils', ]


# import all others as submodules
from . import remap, run_h5, range_vs_energy, cnn_images, energy_calibration, energy_spectrum, event_plotting, raw_h5