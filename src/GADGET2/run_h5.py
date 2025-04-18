import socket
import numpy as np
import os
from .energy_calibration import to_MeV
from .raw_h5 import raw_h5_file
import matplotlib.path


def run_num_to_str(run_num):
    run_num = int(run_num)
    return  ('%4d'%run_num).replace(' ', '0')

def get_h5_path(path=None):
    if path is None:
        machine = socket.gethostname()
        if machine == 'tpcgpu':
            return "/egr/research-tpc/shared/Run_Data/"
        elif machine in ['pike', 'steelhead', 'flagtail']:
            return "/mnt/analysis/e21072/h5test/"
        elif machine == 'warbler':
            return "/home/adam/GADGET2/data/"
        else:
            raise ValueError(f"Unknown machine: {machine}")
    else:
        return path

def get_default_path(run_id):    
    run_str = run_num_to_str(run_id)
    return f'{get_h5_path()}run_{run_str}'

class GadgetRunH5:
    def __init__(self, run_num, folder_path=None):
        self.run_num = run_num
        
        if folder_path is None: # if no folder path is provided, use the default
            try:
                folder_path = get_default_path(run_num)
                subdirs = os.listdir(folder_path)
                print(f"Located files for run {run_num}:")
                [print(subdir) for subdir in subdirs]
                for subdir in subdirs:
                    if 'tot_energy.npy' in os.listdir(os.path.join(folder_path, subdir)):
                        folder_path = os.path.join(folder_path, subdir)
                        print(f"Default path chosen: {folder_path}")
                        break
            except:
                raise ValueError(f"Could not locate run {run_num} files using default path.")
        self.folder_path = folder_path
        
        #energy in adc counts
        self.total_energy = np.load(os.path.join(folder_path, 'tot_energy.npy'), allow_pickle=True)
        #
        self.skipped_events = np.load(os.path.join(folder_path, 'skipped_events.npy'), allow_pickle=True)
        #
        self.veto_events = np.load(os.path.join(folder_path, 'veto_events.npy'), allow_pickle=True)
        #list of event numbers selected for inclusion
        self.good_events = np.load(os.path.join(folder_path, 'good_events.npy'), allow_pickle=True)
        #track lengths
        self.len_list = np.load(os.path.join(folder_path, 'len_list.npy'), allow_pickle=True)
        #time series of charge collected
        self.trace_list = np.load(os.path.join(folder_path, 'trace_list.npy'), allow_pickle=True)
        #
        self.angle_list = np.load(os.path.join(folder_path, 'angle_list.npy'), allow_pickle=True)
        self.file_path = get_h5_path() + (f'run_{run_num_to_str(run_num)}.h5')
    
        self.h5_file = raw_h5_file(self.file_path, flat_lookup_csv=f'{__file__[:-9]}/data/flatlookup4cobos.csv')
        self.h5_file.background_subtract_mode='fixed window'
        self.h5_file.data_select_mode='near peak'
        self.h5_file.remove_outliers=True
        self.h5_file.near_peak_window_width = 50
        self.h5_file.require_peak_within= (-np.inf, np.inf)
        self.h5_file.num_background_bins=(160, 250)
        self.h5_file.zscale = 1.45 # Have also used 1.92 # TODO: add as parameter in h5 file?
        self.h5_filelength_counts_threshold = 100
        self.h5_file.ic_counts_threshold = 25
        self.h5_file.include_counts_on_veto_pads = False
        
        self.total_energy_MeV = to_MeV(self.total_energy)

        self.xHit_list = np.load(os.path.join(self.folder_path, 'xHit_list.npy'), allow_pickle=True)
        self.yHit_list = np.load(os.path.join(self.folder_path, 'yHit_list.npy'), allow_pickle=True)
        self.zHit_list = np.load(os.path.join(self.folder_path, 'zHit_list.npy'), allow_pickle=True)
        self.eHit_list = np.load(os.path.join(self.folder_path, 'eHit_list.npy'), allow_pickle=True)

    def get_index(self, event_num):
        """
        Gets the index at which an event number can be found in the data
        """
        return np.where(self.good_events == event_num)[0][0]

    def get_hit_lists(self, event_num):
        """
        I think these are x,y,z positions of points in the point cloud, and 
        the energy associated with each point.
        """
        index = self.get_index(event_num)
        return self.xHit_list[index], self.yHit_list[index], \
            self.zHit_list[index], self.eHit_list[index]

    def get_RvE_cut_indexes(self, points):
        """
        points: list of (energy, range) tuples defining a cut in RvE
        Energy is in MeV, range in mm
        """
        path = matplotlib.path.Path(points)
        to_return = []
        index = 0
        while index < len(self.good_events):
            this_point = (self.total_energy_MeV[index], self.len_list[index])
            if path.contains_point(this_point):
                to_return.append(index)
            index += 1
        return to_return
