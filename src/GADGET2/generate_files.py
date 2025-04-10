"""
This script generates the files needed for the GADGET GUI from the h5 files.
"""

import os
import sys
import h5py
import numpy as np
import math
import tqdm
from BaselineRemoval import BaselineRemoval
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import skspatial.objects

from .cnn_images import remove_noise, smooth_trace
from .run_h5 import run_num_to_str, get_default_path
from .utils import vprint


def generate_files(run_num:int, length=80, ic=800000, pads=21, eps=7, samps=8, poly=2, overwrite=False, verbose=False):
    """
    Generates the files needed for the GADGET GUI from the h5 files.
    
    Parameters
    - run_num: Run number of the h5 file to be processed.
    - length: Maximum length of the track in mm.
    - ic: Maximum energy in ADC counts.
    - pads: Minimum number of pads hit.
    - eps: Maximum distance between two points to be considered in the same cluster.
    - samps: Minimum number of samples in a cluster.
    - poly: Degree of the polynomial used to fit the trace.
    overwrite: whether to overwrite existing files.
    verbose: for debugging purposes, prints more information.
    """
    run_num = run_num_to_str(run_num)
    vprint(f"run_num: {run_num}", verbose)
    #check if files already exist
    mypath = get_default_path(run_num)
    sub_mypath = mypath + f'/len{length}_ic{ic}_pads{pads}_eps{eps}_samps{samps}_poly{poly}'
    vprint(f"mypath: {mypath}", verbose)
    if os.path.isdir(mypath):
        if os.path.isdir(sub_mypath):
            # Give option to overwrite or cancel
            if overwrite == False:
                print("Files Already Exist.")
                overwrite = int(input('Would you like to overwrite (1=yes, 0=no): '))    
                if overwrite == True:
                    print('Overwriting Existing Files')
                else:
                    return
        else:
            os.makedirs(sub_mypath)
        
    else:
        os.makedirs(mypath)
        os.makedirs(sub_mypath)
    
    vprint(f"sub_mypath: {sub_mypath}", verbose)
    
    class HiddenPrints:
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout

    def remove_outliers(xset, yset, zset, eset, pads):
        """
        Uses DBSCAN to find and remove outliers in 3D data
        """

        data = np.array([xset.T, yset.T, zset.T]).T
        DBSCAN_cluster = DBSCAN(eps=eps, min_samples=samps).fit(data)
        del data
    
        if all(element == -1 for element in DBSCAN_cluster.labels_):
            veto = True
        else:
            # Identify largest clusters
            labels = DBSCAN_cluster.labels_
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)

            # Find the two largest clusters
            largest_clusters = sorted(unique_labels, key=lambda x: np.sum(labels == x), reverse=True)[:2]

            # Relabel non-main-cluster points as outliers
            for cluster_label in unique_labels:
                if cluster_label not in largest_clusters:
                    labels[labels == cluster_label] = -1

            # Remove outlier points
            out_of_cluster_index = np.where(labels == -1)
            rev = out_of_cluster_index[0][::-1]
            for i in rev:
                xset = np.delete(xset, i)
                yset = np.delete(yset, i)
                zset = np.delete(zset, i)
                eset = np.delete(eset, i)

            if len(xset) <= pads:
                veto = True
            else:
                veto = False

        return xset, yset, zset, eset, veto

    def track_len(xset, yset, zset):
        """
        Uses PCA to find the length of a track
        """
        veto_on_length = False

        # Form data matrix
        data = np.concatenate((xset[:, np.newaxis], 
               yset[:, np.newaxis], 
               zset[:, np.newaxis]), 
               axis=1)
        
        # Use PCA to find track length
        pca = PCA(n_components=2)
        principalComponents = pca.fit(data)
        principalComponents = pca.transform(data)
        principalDf = pd.DataFrame(data = principalComponents,
                                   columns = ['principal component 1', 'principal component 2'])
        calibration_factor = 1.4

        # Call track_angle to get the angle of the track
        angle_deg = 0# track_angle(xset, yset, zset)

        # Calculate the scale factor based on the angle
        #scale_factor = get_scale_factor(angle_deg)
        #scale_factor_trace = scale_factor * 1.5 

        # Apply the scale factor to the track length
        # track_len = scale_factor * calibration_factor * 2.35 * principalDf.std()[0]
        track_len = calibration_factor * 2.35 * principalDf.std()[0]
        vprint(f"track_len: {track_len}", verbose)

        if track_len > length:
            veto_on_length = True

        return track_len, veto_on_length, angle_deg



    def track_angle(xset, yset, zset):
        """
        Fits 3D track, and determines angle wrt pad plane
        """
        # Form data matrix
        data = np.concatenate((xset[:, np.newaxis], 
                   yset[:, np.newaxis], 
                   zset[:, np.newaxis]), 
                   axis=1)
        vprint(f"data shape: {data.shape}", verbose)
        vprint(f"data: {data}", verbose)

        # Fit regression line
        # TODO: THIS LINE FITTING IS NOT WORKING
        line_fit = skspatial.objects.Line.best_fit(data)
        vprint(f"line_fit: {line_fit}", verbose)

        # Find angle between the vector of the fit line and a vector normal to the xy-plane (pad plane)
        v = np.array([line_fit.vector]).T   # fit line vector
        n = np.array(([[0, 0, 1]])).T       # Vector normal to xy-plane
        dot = np.dot(n.T, v)[0][0]          # Note that both vectors already have a magnitude of 1

        # Clamp the dot variable to be within the valid range
        dot = max(-1.0, min(1.0, dot))

        theta = math.acos(dot)
        track_angle_rad = (math.pi/2 - theta) 
        track_angle_deg = track_angle_rad * (180 / np.pi)
        vprint(f"track_angle_rad: {track_angle_rad}", verbose)

        # Angle should always be less than 90 deg
        if track_angle_deg < 0:
            track_angle_deg = 180 + track_angle_deg 
        if track_angle_deg > 90:
            track_angle_deg = 180 - track_angle_deg

        return track_angle_deg

    def get_scale_factor(angle, angle_min=40, angle_max=90, scale_min=1, scale_max=1.3):
        if angle < angle_min:
            return scale_min
        elif angle > angle_max:
            return scale_max
        else:
            return scale_min + (scale_max - scale_min) * (angle - angle_min) / (angle_max - angle_min)


    def main(h5file, pads, ic):
        """
        This functions does the following: 
        - Converts h5 files into ndarrays. 
        - Removes outliers.
        - Calls PCA to return track length.
        - Sums mesh signal to return energy.
        """
        # Converts h5 files into ndarrays, and output each event dataset as a separte list
        num_events = int(h5file['meta/meta'][2])
        vprint(f"num_events: {num_events}", verbose)

        len_list = []
        good_events = []
        tot_energy = []
        trace_list = []
        xHit_list = []
        yHit_list = []
        zHit_list = []
        eHit_list = []
        angle_list = []

        cloud_missing = 0
        skipped_events = 0
        veto_events = 0
        
        # Veto in "junk" region of plot (low energy, high range)
        # Define veto region
        #y = np.array([20, 40, 60])
        #x = np.array([100000, 150000, 200000])
        #slope, intercept = np.polyfit(x, y, deg=1)

        for i in tqdm.trange(1, num_events+1):
            # Make copy of cloud datasets
            str_cloud = f"evt{i}_cloud"
            vprint(f"str_cloud: {str_cloud}", verbose)
            try:
                cloud = h5file[f'clouds/{str_cloud}']
                cloud = np.array(cloud)
                vprint(f"cloud shape: {cloud.shape}", verbose)
            except:
                vprint(f"cloud not found for event {i}", verbose)
                cloud_missing += 1
                continue
            

            # Make copy of datasets
            cloud_x = cloud[:,0]
            cloud_y = cloud[:,1]
            cloud_z = cloud[:,2]
            #cloud_z = cloud[:,2] - np.min(cloud[:, 2])
            cloud_e = cloud[:,3]
            del cloud
            vprint(f"cloud x: {cloud_x.shape}", verbose)
            vprint(f"cloud y: {cloud_y.shape}", verbose)
            vprint(f"cloud z: {cloud_z.shape}", verbose)
            vprint(f"cloud e: {cloud_e.shape}", verbose)
            
            # Apply veto condition
            R = 36                           # Radius of the pad plane
            r = np.sqrt(cloud_x**2 + cloud_y**2)
            statements = np.greater(r, R)    # Check if any point lies outside of R
            vprint(f"max r: {np.max(r)}", verbose)

            if np.any(statements) == True:
                veto_events += 1
                vprint(f"skipped event {i} because of veto condition", verbose)
                continue
            
            # Apply pad threshold
            x = (35 + cloud_x) * 2 + 42
            y = 145 - (35 + cloud_y) * 2
            xy_tuples = np.column_stack((x, y))
            unique_xy_tuples = set(map(tuple, xy_tuples))
            num_unique_tuples = len(unique_xy_tuples)
            vprint(f"num_unique_tuples: {num_unique_tuples}", verbose)

            if num_unique_tuples <= pads:
                skipped_events += 1
                vprint(f"skipped event {i} because of pad threshold", verbose)
                continue

            """
            # Call remove_outliers to get dataset w/ outliers removed
            cloud_x, cloud_y, cloud_z, cloud_e, veto = remove_outliers(cloud_x, cloud_y, cloud_z, cloud_e, pads)
            if veto == True:
                skipped_events += 1
                vprint(f"skipped event {i} because of outliers")
                continue
            """
            # Move track next to pad plane for 3D view and scale by appropriate factor
            zscale = 1.45 # Have also used 1.92
            cloud_z = (cloud_z  - np.min(cloud_z))*zscale
            vprint(f"cloud_z rescaled: {cloud_z.shape}", verbose)
            #cloud_z = (cloud_z  - np.min(cloud_z ))

            # Call track_len() to create lists of all track lengths
            length, veto_on_length, angle = track_len(cloud_x, cloud_y, cloud_z)
            vprint(f"length: {length}", verbose)
            vprint(f"veto_on_length: {veto_on_length}", verbose)
            vprint(f"angle: {angle}", verbose)
            if veto_on_length == True:
                veto_events += 1
                vprint(f"skipped event {i} because of length threshold", verbose)
                continue 

            str_trace = f"evt{i}_data"
            data_trace = np.array(h5file['get'][str_trace])
            # pad_nums = data_trace[:,4]             

            trace = np.sum(data_trace[:, -512:], axis=0)
            del data_trace


            max_val = np.argmax(trace)
            low_bound = max_val - 75
            if low_bound < 0:
                low_bound = 5
            upper_bound = max_val + 75
            if upper_bound > 512:
                upper_bound = 506
            trace = trace[low_bound:upper_bound]
            
            """
            # Smooth trace
            trace = smooth_trace(trace)

            # Subtract background and fit trace
            polynomial_degree=poly
            baseObj=BaselineRemoval(trace)
            trace=baseObj.IModPoly(polynomial_degree)

            # Remove noise, negative values, and zero consecutive bins
            trace = remove_noise(trace, threshold_ratio=0.01)
            """

            # Here you can apply the scale factor to the total energy
            scaled_energy = np.sum(trace)

            if scaled_energy > ic:
                veto_events += 1
                vprint(f"skipped event {i} because of energy threshold", verbose)
                continue

            """
            # Check to see if point is in "junk" region
            x1 = scaled_energy
            y1 = length
            y_line = slope * x1 + intercept
            if y1 > y_line and y1 > 20:
                veto_events += 1
                continue
            """

            # Call track_angle to create list of all track angles
            angle_list.append(angle)

            # Append all lists
            len_list.append(length)
            tot_energy.append(scaled_energy)
            trace_list.append(trace)
            xHit_list.append(cloud_x)
            yHit_list.append(cloud_y)
            zHit_list.append(cloud_z)
            eHit_list.append(cloud_e)

            # Track original event number of good events
            good_events.append(i)

        print('Starting # of Events:', num_events)
        print('Events Below Threshold:', skipped_events)
        print('Vetoed Events:', veto_events)
        print('Events Missing Cloud:', cloud_missing)
        print('Final # of Good Events:', len(good_events))

        return (tot_energy, skipped_events, veto_events, good_events, len_list, trace_list, xHit_list, yHit_list, zHit_list, eHit_list, angle_list)

    str_file = f"{get_default_path(run_num)}.h5"
    f = h5py.File(str_file, 'r')
    vprint(f"opened file {str_file}", verbose)
    (tot_energy, skipped_events, veto_events, good_events, len_list, trace_list, xHit_list, yHit_list, zHit_list, eHit_list, angle_list) = main(h5file=f, pads=pads, ic=ic)

    # Save Arrays
    if len(good_events) == 0:
        print(f"No good events in run {run_num} found with parameters, not saving files.")
        print(f"len{length}_ic{ic}_pads{pads}_eps{eps}_samps{samps}_poly{poly}")
        # remove directory
        try: os.rmdir(sub_mypath)
        except OSError: pass
        return
    
    tot_energy = np.array(tot_energy)
    skipped_events = np.array(skipped_events)
    veto_events = np.array(veto_events)
    good_events = np.array(good_events)
    len_list = np.array(len_list)
    trace_list = np.array(trace_list, dtype=object)
    xHit_list = np.array(xHit_list, dtype=object)
    yHit_list = np.array(yHit_list, dtype=object)
    zHit_list = np.array(zHit_list, dtype=object)
    eHit_list = np.array(eHit_list, dtype=object)
    angle_list = np.array(angle_list, dtype=object)
    
    
    sub_path = f"{get_default_path(run_num)}/len{length}_ic{ic}_pads{pads}_eps{eps}_samps{samps}_poly{poly}/"
    print(f"DIRECTORY: {sub_path}")
    np.save(f"{sub_path}tot_energy", tot_energy, allow_pickle=True)
    np.save(f"{sub_path}skipped_events", skipped_events, allow_pickle=True)
    np.save(f"{sub_path}veto_events", veto_events, allow_pickle=True)
    np.save(f"{sub_path}good_events", good_events, allow_pickle=True)
    np.save(f"{sub_path}len_list", len_list, allow_pickle=True)
    np.save(f"{sub_path}trace_list", trace_list, allow_pickle=True)
    np.save(f"{sub_path}xHit_list", xHit_list, allow_pickle=True)
    np.save(f"{sub_path}yHit_list", yHit_list, allow_pickle=True)
    np.save(f"{sub_path}zHit_list", zHit_list, allow_pickle=True)
    np.save(f"{sub_path}eHit_list", eHit_list, allow_pickle=True)
    np.save(f"{sub_path}angle_list", angle_list, allow_pickle=True)

    #Delete arrays
    del tot_energy
    del skipped_events
    del veto_events
    del good_events
    del len_list
    del trace_list
    del xHit_list
    del yHit_list
    del zHit_list
    del eHit_list
    del angle_list