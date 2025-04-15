"""
This package provides a collection of classes and functions for analyzing GADGET2 data.
"""

from .energy_calibration import to_MeV, to_counts
from .energy_spectrum import plot_1d_hist, show_cut, plot_spectrum, quick_fit_gaussian, multi_fit_from_params, multi_fit_init_guess, fit_multi_peaks
from . import event_plotting # WIP
from .generate_files import generate_files
from . import point_cloud # WIP
from .range_vs_energy import plot_RVE, show_RVE_event, save_cut_files, get_RvE_cut_indexes
from . import raw_h5 # WIP
from . import remap
from .run_h5 import GadgetRunH5, get_h5_path, get_default_path, run_num_to_str
from . import utils

__all__ = [
    'to_MeV', 'to_counts', # energy_calibration
    'plot_1d_hist', 'show_cut', 'plot_spectrum', 'quick_fit_gaussian', # energy_spectrum
    'multi_fit_from_params', 'multi_fit_init_guess', 'fit_multi_peaks',
    'event_plotting', # event_plotting
    'generate_files', # generate_files
    'point_cloud', # point_cloud
    'plot_RVE', 'show_RVE_event', 'save_cut_files', 'get_RvE_cut_indexes', # range_vs_energy
    'raw_h5', # raw_h5
    'remap', # remap
    'GadgetRunH5', 'get_h5_path', 'get_default_path', 'run_num_to_str', # run_h5
    'utils' # utils
]