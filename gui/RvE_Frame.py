import datetime
import random
import os

import tkinter.filedialog
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.path import Path
from GadgetRunH5 import GadgetRunH5
import numpy as np
from tqdm import tqdm
import pickle

import gadget_widgets
from prev_cut_select_window import PrevCutSelectWindow

class RvE_Frame(ttk.Frame):
    def __init__(self, parent, run_data:GadgetRunH5):
        super().__init__(parent)
        self.run_data = run_data
        #show background image
        self.background_image = gadget_widgets.get_background_image()
        self.background = ttk.Label(self, image=self.background_image)
        self.background.place(relx=0.5, rely=0.5, anchor='center')
        #plot settings
        self.plot_settings_frame = ttk.LabelFrame(self,text='plot settings')
        self.plot_settings_frame.grid(row=0)
        self.range_bins_label = ttk.Label(self.plot_settings_frame, text='# range bins:')
        self.range_bins_label.grid(row=0, column=0)
        self.range_bins_entry = gadget_widgets.GEntry(self.plot_settings_frame)
        self.range_bins_entry.grid(row=0, column=1)
        self.range_bins_entry.insert(0,'200')
        self.energy_bins_label = ttk.Label(self.plot_settings_frame, text='# energy bins:')
        self.energy_bins_label.grid(row=0, column=2)
        self.energy_bins_entry = gadget_widgets.GEntry(self.plot_settings_frame)
        self.energy_bins_entry.grid(row=0, column=3)
        self.energy_bins_entry.insert(0,'200')
        self.log_scale_var = tk.BooleanVar(value=True)
        self.scale_label = ttk.Label(self.plot_settings_frame, text='energy scale:')
        self.scale_label.grid(row=1, column=1, sticky=tk.E)
        self.log_scale_radio = ttk.Radiobutton(self.plot_settings_frame,
                                               text='log', variable=self.log_scale_var,
                                               value=True)
        self.lin_scale_radio = ttk.Radiobutton(self.plot_settings_frame,
                                               text='linear', variable=self.log_scale_var,
                                               value=False)
        self.lin_scale_radio.grid(row=1,column=2)
        self.log_scale_radio.grid(row=1,column=3)
        #basic viewing tools
        self.view_frame = ttk.LabelFrame(self, text='viewing tools')
        self.view_frame.grid(row=1)
        self.show_rve_plot_button = ttk.Button(self.view_frame, text='Plot Range vs Energy',
                                               command=self.plot_spectrum)
        self.show_rve_plot_button.grid(row=0, column=0, columnspan=2)
        self.event_num_entry = gadget_widgets.GEntry(self.view_frame, default_text='Event #')
        self.event_num_entry.grid(row=1, column=0)
        self.show_event_button = ttk.Button(self.view_frame, text='show event on RvE plot',
                                            command=self.show_event)
        self.show_event_button.grid(row=1, column=1)
        self.cut_tools_frame = ttk.LabelFrame(self, text='cut tools')
        self.cut_tools_frame.grid(row=2)
        #TODO: implement manual cut, and project to axis
        self.manual_cut_button = ttk.Button(self.cut_tools_frame, text='Manual Cut Selection')
        self.manual_cut_button.grid(row=0, column=0)
        self.from_file_cut_button = ttk.Button(self.cut_tools_frame,
                                               text='Polygon from File',
                                               command=self.cut_from_file)
        self.from_file_cut_button.grid(row=0, column=1)
        self.from_file_cut_button_raw = ttk.Button(self.cut_tools_frame,
                                               text='Polygon (Raw) from File',
                                               command=self.cut_from_file_raw)
        self.from_file_cut_button_raw.grid(row=0, column=2)
        self.prev_cut_button = ttk.Button(self.cut_tools_frame, 
                                          text='Previous Cuts',
                                          command=self.prev_cut)
        self.prev_cut_button.grid(row=1, column=0, columnspan=2)
        self.project_cut_x_ax_button = ttk.Button(self.cut_tools_frame, text='Project Cut to X-axis')
        self.project_cut_x_ax_button.grid(row=2, column=0)
        self.project_cut_y_ax_button = ttk.Button(self.cut_tools_frame, text='Project Cut to Y-axis')
        self.project_cut_y_ax_button.grid(row=2, column=1)
    
    def prev_cut(self):
        PrevCutSelectWindow(self, self.run_data)
    
    def cut_from_file(self):
        '''
        Input files should have an energy (MeV) followed by range (mm) on each line,
        with the values seperated by a space. 
        '''
        fname = tkinter.filedialog.askopenfile(initialdir=os.getcwd())
        points = np.loadtxt(fname)
        self.save_cut_files(points)

    def cut_from_file_raw(self):
        '''
        Input files should have an energy (MeV) followed by range (mm) on each line,
        with the values seperated by a space. 
        '''
        fname = tkinter.filedialog.askopenfile(initialdir=os.getcwd())
        points = np.loadtxt(fname)
        self.save_cut_files(points, use_raw_data=True)