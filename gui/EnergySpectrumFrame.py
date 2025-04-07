import tkinter as tk
from tkinter import ttk
import gadget_widgets
import GADGET2

import matplotlib.pyplot as plt
import numpy as np

import scipy.optimize as opt
import scipy.stats as stats

class EnergySpectrumFrame(ttk.Frame):
    def __init__(self, parent, run_data):
        super().__init__(parent)
        self.run_data = run_data
        #show background image
        self.background_image = gadget_widgets.get_background_image()
        self.background = ttk.Label(self, image=self.background_image)
        self.background.place(relx=0.5, rely=0.5, anchor='center')
        #add GUI elements
        num_cols=1
        current_row = 0
        
        self.hist_settings_frame = ttk.Labelframe(self, text='histogram settings')
        self.hist_settings_frame.grid(row=current_row, column=0)
        current_row += 1
        self.num_bins_label = ttk.Label(self.hist_settings_frame, text='num bins:')
        self.num_bins_label.grid(row=0, column=0, sticky=tk.E)
        self.num_bins_entry = gadget_widgets.GEntry(self.hist_settings_frame, default_text='Enter # of Bins')
        self.num_bins_entry.grid(row=0, column=1, columnspan=num_cols)
        self.units_select_label = ttk.Label(self.hist_settings_frame, text='energy units:')
        self.units_select_label.grid(row=1, column=0, sticky=tk.E)
        self.units_combobox = ttk.Combobox(self.hist_settings_frame, values=['adc counts', 'MeV'])
        self.units_combobox.set('adc counts')
        self.units_combobox.grid(row=1, column=1)
        self.generate_spectrum_button = \
                    ttk.Button(self.hist_settings_frame, text="Show Spectrum", command=GADGET2.EnergySpectrum.plot_spectrum)
        self.generate_spectrum_button.grid(row=2, column=0, columnspan=2)
        
        self.quick_fit_frame = ttk.Labelframe(self, text='quick fit')
        self.quick_fit_frame.grid(row=current_row, column=0)
        current_row += 1
        self.low_cut_entry = gadget_widgets.GEntry(self.quick_fit_frame, default_text='Low Cut Value')
        self.low_cut_entry.grid(row=0, column=0)
        self.high_cut_entry = gadget_widgets.GEntry(self.quick_fit_frame, default_text='High Cut Value')
        self.high_cut_entry.grid(row=0, column=1)
        self.show_cut_button = ttk.Button(self.quick_fit_frame, text='Show Cut Range', command=GADGET2.EnergySpectrum.show_cut)
        self.show_cut_button.grid(row=1, column=0, columnspan=2)
        self.quick_fit_gaus_button = ttk.Button(self.quick_fit_frame, text="Quick Gaussian Fit",
                                command=GADGET2.EnergySpectrum.quick_fit_gaussian)
        self.quick_fit_gaus_button.grid(row=2, column=0, columnspan=2)
        current_row += 1

        self.multi_peak_fit_frame = ttk.Labelframe(self, text='multi-peak fit')
        self.multi_peak_fit_frame.grid(row=current_row)
        current_row += 1
        self.multi_fit_button = ttk.Button(self.multi_peak_fit_frame,
                                 text="Initial Guesses for Multi-peak Fit",
                                 command=GADGET2.EnergySpectrum.multi_fit_init_guess)
        self.multi_fit_button.grid(row=0)
        self.multi_fit_params_entry = gadget_widgets.GEntry(self.multi_peak_fit_frame, 
                'Paste Fit Parameters | Use * in Front of Param to Fix Value', width=42)
        self.multi_fit_params_entry.grid(row=1)
        self.multi_fit_from_params_button = ttk.Button(self.multi_peak_fit_frame, text="Multi-peak Fit from Params",
                 command=GADGET2.EnergySpectrum.multi_fit_from_params)
        self.multi_fit_from_params_button.grid(row=2)

    def get_dataset(self):
        if self.units_combobox.get() == 'MeV':
            return self.run_data.total_energy_MeV
        elif self.units_combobox.get() == 'adc counts':
            return self.run_data.total_energy
        else:
            assert False