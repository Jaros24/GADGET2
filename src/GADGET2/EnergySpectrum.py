import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
from .CurveFit import gaussian

def plot_1d_hist(dataset,
                 fig_name:str="Energy Spectrum",
                 units:str="Integrated Charge (adc counts)",
                 num_bins:int=20,
                 vlines:list[float]=None):
    '''
    Shows a 1d histogram of the dataset.
    
    Methods calling this function should call plt.show() after this function call,
    and making any other desired modifications to the figure.
    
    Parameters
    ----------
    dataset : array-like
        The data to be plotted.
    fig_name : str
        The name of the figure to be displayed.
        Default is 'Energy Spectrum'.
    units : str
        The units of the data to be plotted. Used as x-axis label.
        - Integrated Charge (adc counts)
        - Energy (MeV)
    num_bins : int
        The number of bins to be used in the histogram.
        Default is 20.
    vlines : tuple
        A tuple of x-coordinates where vertical lines should be drawn.
        Default is None.
        Useful for marking cuts or peaks.
    '''
    num_bins = int(num_bins)
    #plt.figure(fig_name, clear=True)
    plt.xlabel(f'{units}', fontdict = {'fontsize' : 20})
    plt.ylabel(f'Counts',fontdict = {'fontsize' : 20})
    plt.hist(dataset, bins=num_bins)
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.title(f'{fig_name}', fontdict = {'fontsize' : 20})
    if vlines is not None:
        for vline in vlines:
            plt.axvline(vline, color='red', linestyle='dashed', linewidth=2)


def show_cut(dataset,
                 vlines:list[float, float],
                 fig_name:str="Energy Spectrum",
                 units:str="Integrated Charge (adc counts)",
                 num_bins:int=20) -> int:
    '''
    Shows a 1d histogram of the dataset with vertical lines at the specified cut values.
    Titles the plot with the number of counts in the cut range.
    Wrapper for the plot_1d_hist function.
    
    Returns the number of counts in the cut range.
    '''
    # Show basic plot
    plot_1d_hist(dataset, fig_name, units, num_bins, vlines)
    # Get number of counts in range, order of vlines doesn't matter
    low_cut = min(vlines)
    high_cut = max(vlines) 
    trimmed_hist = dataset[np.logical_and(dataset>=low_cut, dataset<=high_cut)]
    plt.title(f'Number of Counts in Cut: {len(trimmed_hist)}',fontdict = {'fontsize' : 20})
    plt.show()
    return len(trimmed_hist)


def plot_spectrum(dataset,
                 fig_name:str="Energy Spectrum",
                 units:str="Integrated Charge (adc counts)",
                 num_bins:int=20,
                 vlines:list[float]=None):
    '''
    Displays a 1D histogram of the dataset.
    Wrapper for the plot_1d_hist function.
    '''
    plot_1d_hist(dataset, fig_name, units, num_bins, vlines)
    plt.show()


def quick_fit_gaussian(dataset,
                 units:str="Integrated Charge (adc counts)",
                 num_bins:int=20,
                 vlines:list[float]=None):
    '''
    Fits a Gaussian to the histogram of the dataset.
    Displays the histogram with the fit and the residuals.
    Wrapper for the plot_1d_hist function.
    '''
    
    if vlines is None:
        low_cut_value = np.min(dataset)
        high_cut_value = np.max(dataset)
    else:
        low_cut_value = min(vlines)
        high_cut_value = max(vlines)
        
    hist, bins = np.histogram(dataset, bins=num_bins, range=(low_cut_value, high_cut_value))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    #fit curve
    popt, pcov = opt.curve_fit(gaussian, bin_centers, hist, p0=[1, np.mean(dataset), np.std(dataset)], maxfev=800000)
    amplitude, mu, sigma = popt
    # Calculate chi-squared and p-value
    residuals = hist - gaussian(bin_centers, amplitude, mu, sigma)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((hist - np.mean(hist))**2)
    r_squared = 1 - (ss_res / ss_tot)
    chi_squared = ss_res / (num_bins - 3)
    dof = num_bins - 3
    chi_squared_dof = chi_squared / dof
    p_value = 1 - stats.chi2.cdf(chi_squared, dof)
    
    # Plot the histogram with the fit
    plt.subplot(211)
    plot_1d_hist(dataset, "Quick Gaussian Fit", "", num_bins, vlines)

    x_fit = np.linspace(low_cut_value, high_cut_value, 100)
    plt.plot(x_fit, gaussian(x_fit, amplitude, mu, sigma), 'r-', label='Fit')
    
    # Plot the residuals
    plt.subplot(212)
    plt.plot(bin_centers, residuals, 'b-', label='Residuals')
    plt.axhline(0, color='black', lw=1)
    plt.xlabel(f'{units}', fontdict = {'fontsize' : 20})
    plt.ylabel('Residuals')
    plt.legend()
    plt.tight_layout()
    # Display the fit parameters on the plot
    text = f'Chi-squared: {chi_squared:.2f}\nDegrees of Freedom: {dof}\nChi-squared per DOF: {chi_squared_dof:.2f}\np-value: {p_value:.2f}\nAmplitude: {amplitude:.2f}\nMu: {mu:.2f}\nSigma: {sigma:.2f}'
    plt.text(low_cut_value + 0.05, 0.95, text, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5)) #transform=ax[0].transAxes
    plt.show()

def multi_fit_init_guess(self):
    '''Not Implemented From GitLab Yet'''
    raise NotImplementedError("multi_fit_init_guess is not implemented")

def fit_multi_peaks(self, num_bins, x_hist, y_hist):
    '''Not Implemented From GitLab Yet'''
    raise NotImplementedError("fit_multi_peaks is not implemented")

def multi_fit_from_params(self):
    '''Not Implemented From GitLab Yet'''
    raise NotImplementedError("multi_fit_from_params is not implemented")