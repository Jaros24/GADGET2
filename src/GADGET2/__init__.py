"""
This package provides a collection of classes and functions for analyzing GADGET2 data.
"""


from .generate_files import generate_files
from .run_h5 import GadgetRunH5, get_h5_path, get_default_path, run_num_to_str
from .energy_spectrum import plot_1d_hist, show_cut, plot_spectrum, quick_fit_gaussian, multi_fit_from_params, multi_fit_init_guess, fit_multi_peaks
from . import utils

__all__ = [
    'generate_files', # generate_files
    'GadgetRunH5', 'get_h5_path', 'get_default_path', 'run_num_to_str', # GadgetRunH5
    'plot_1d_hist', 'show_cut', 'plot_spectrum', 'quick_fit_gaussian', #energy_spectrum
    'multi_fit_from_params', 'multi_fit_init_guess', 'fit_multi_peaks',
    'utils' # utils
    ]

# import all others as submodules
from . import remap, range_vs_energy, cnn_images, energy_calibration, event_plotting, raw_h5