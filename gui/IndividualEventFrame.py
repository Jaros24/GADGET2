import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import gadget_widgets
from fit_gui.FitFrame import FitFrame

from skspatial.objects import Line

from raw_viewer.RawEventViewerFrame import RawEventViewerFrame

class IndividualEventFrame(ttk.Frame):
    def __init__(self, parent, run_data):
        super().__init__(parent)
        self.run_data = run_data
        #show background image
        self.background_image = gadget_widgets.get_background_image()
        self.background = ttk.Label(self, image=self.background_image)
        self.background.place(relx=0.5, rely=0.5, anchor='center')

        self.event_select_frame = ttk.LabelFrame(self, text='Event Selection')
        self.event_select_frame.pack()
        ttk.Label(self.event_select_frame, text='Event #').grid(row=0, column=0)
        self.event_num_entry = gadget_widgets.GEntry(self.event_select_frame,
                                                      text='Enter Event #')
        self.event_num_entry.grid(row=0, column=1)

        self.threeD_frame = ttk.LabelFrame(self, text='Point Cloud Viewer')
        track_w_trace_button = ttk.Button(self.threeD_frame,
                                          text='Show Track w/ Trace (Point Cloud)',
                                          command = self.track_w_trace)
        track_w_trace_button.grid(row=0, column=0) #, columnspan=2
        track_w_trace_button_raw = ttk.Button(self.threeD_frame,
                                          text='Show Track w/ Trace (Raw Data) : Unfiltered Trace',
                                          command = self.track_w_trace_raw)
        track_w_trace_button_raw.grid(row=0, column=1)
        track_w_trace_button_raw_smooth = ttk.Button(self.threeD_frame,
                                          text='Show Track w/ Trace (Raw Data) : Smooth Trace',
                                          command = self.track_w_trace_raw_smooth)
        track_w_trace_button_raw_smooth.grid(row=1, column=0)
        ttk.Button(self.threeD_frame, text='3D Point Cloud', 
                   command=self.show_point_cloud).grid(row=2, column=0)
        ttk.Button(self.threeD_frame, text='3D Dense Point Cloud',
                   command=self.plot_dense_3d_track).grid(row=2, column=1)
        self.threeD_frame.pack()

        fitting_frame = ttk.LabelFrame(self, text='Point Cloud Fitting Tools')
        fitting_frame.pack()
        ttk.Label(fitting_frame, text='Bandwidth Factor:').grid(row=0, column=0)
        self.bandwidth_entry = ttk.Entry(fitting_frame)
        self.bandwidth_entry.grid(row=0, column=1)
        ttk.Button(fitting_frame, text='Project onto Principle Axis',
                   command=self.project_trace).grid(row=1, columnspan=2)
        
        trace_frame = ttk.LabelFrame(self, text='Original Trace Data')
        ttk.Button(trace_frame, text='Open Raw Data Viewer', 
                   command=self.open_raw_viewer).grid(row=1, column=0)
        trace_frame.pack()

    def open_raw_viewer(self):
        new_window = tk.Toplevel(self)
        self.viewer_frame = RawEventViewerFrame(new_window, file_path=self.run_data.h5_file_path, flat_lookup_path='raw_viewer/channel_mappings/flatlookup4cobos.csv')
        self.viewer_frame.pack()

    def project_trace(self):
        debug = True

        index = self.run_data.get_index(int(self.event_num_entry.get()))
        bandwidth = float(self.bandwidth_entry.get())

        xHit = self.run_data.xHit_list[index]
        yHit = self.run_data.yHit_list[index]
        zHit = self.run_data.zHit_list[index]
        eHit = self.run_data.eHit_list[index]

        extend_bins = 10 #TODO: this should probably be relaed to the bandwidth
        if debug:
            self.show_plot(xHit, yHit, zHit, eHit)
        
        #TODO: do we want to weight the fit by energy deposition?
        points = np.vstack((xHit, yHit, zHit)).transpose()
        line = Line.best_fit(points)
        pstart, direction = line.point, line.vector
        xProj, yProj, zProj = [], [], []
        for x,y,z in zip(xHit, yHit, zHit):
            v = line.project_vector([x, y, z])
            xProj.append(v[0])
            yProj.append(v[1])
            zProj.append(v[2])
        xProj = np.array(xProj)
        yProj = np.array(yProj)
        zProj = np.array(zProj)
        if debug:
            self.show_plot(xProj, yProj,zProj, eHit)

        dist = np.sqrt((xProj - pstart[0])**2 + (yProj - pstart[1])**2 + (zProj - pstart[2])**2)

        kde = scipy.stats.gaussian_kde(dist, weights=eHit)
        # division_factor = 3
        kde.set_bandwidth(kde.factor / bandwidth)  # You can adjust the bandwidth to control the smoothness

        # Create a dense array of x values for the histogram
        x_dense = np.linspace(np.min(dist) - extend_bins, np.max(dist) + extend_bins, 100)

        # Evaluate the KDE for the dense x values
        y_smooth = kde.evaluate(x_dense)

        new_window = tk.Toplevel()
        FitFrame(new_window, x_dense, y_smooth).pack()



