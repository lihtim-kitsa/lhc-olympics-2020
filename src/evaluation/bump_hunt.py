import numpy as np
from scipy.optimize import curve_fit
import scipy.stats as stats

class BumpHunter:
    def __init__(self, fit_range=(2500, 4500), signal_window=(3300, 3700), n_bins=40):
        self.fit_range = fit_range
        self.signal_window = signal_window
        self.n_bins = n_bins
        
        # Define bin edges for the full range
        self.bins = np.linspace(self.fit_range[0], self.fit_range[1], self.n_bins + 1)
        self.bin_centers = 0.5 * (self.bins[:-1] + self.bins[1:])
        self.bin_widths = np.diff(self.bins)
        
        # Mask for sideband (excluding signal window)
        self.sideband_mask = (self.bin_centers < self.signal_window[0]) | (self.bin_centers > self.signal_window[1])
        self.signal_mask = ~self.sideband_mask

    @staticmethod
    def dijet_func(x, p0, p1, p2, p3):
        """
        Standard dijet background function:
        f(x) = p0 * (1 - x)^p1 / (x^(p2 + p3*log(x)))
        Here x is scaled mass mjj / 13000
        """
        x_scaled = x / 13000.0
        # Add small epsilon to avoid division by zero or invalid logs
        x_scaled = np.clip(x_scaled, 1e-5, 1.0 - 1e-5)
        return p0 * ((1.0 - x_scaled)**p1) / (x_scaled**(p2 + p3 * np.log(x_scaled)))

    def fit_background(self, mjj_data):
        """
        Fits the background function to the sidebands of the mjj distribution.
        Returns the expected background counts in the signal region.
        """
        # Histogram the data
        counts, _ = np.histogram(mjj_data, bins=self.bins)
        errors = np.sqrt(counts)
        # Avoid division by zero in weights
        errors[errors == 0] = 1.0
        
        # Get sideband data
        x_fit = self.bin_centers[self.sideband_mask]
        y_fit = counts[self.sideband_mask]
        err_fit = errors[self.sideband_mask]
        
        # Initial guess
        p0_guess = np.sum(y_fit) * 10
        p0_init = [p0_guess, 5.0, 5.0, 0.0]
        
        try:
            popt, pcov = curve_fit(
                self.dijet_func, 
                x_fit, 
                y_fit, 
                p0=p0_init, 
                sigma=err_fit, 
                absolute_sigma=True,
                maxfev=10000
            )
        except RuntimeError:
            print("Warning: Curve fit failed to converge. Returning NaNs.")
            return np.nan, np.nan, np.nan
            
        # Predict in the signal region
        x_sig = self.bin_centers[self.signal_mask]
        expected_sig_counts = self.dijet_func(x_sig, *popt)
        total_expected_bkg = np.sum(expected_sig_counts)
        
        # Observed in signal region
        total_observed = np.sum(counts[self.signal_mask])
        
        # Simple significance calculation: S / sqrt(B)
        # Or using Poisson likelihood ratio
        if total_expected_bkg > 0:
            # Local significance using asymptotic formula for Poisson
            # Z = sqrt(2 * (N * ln(N/B) - (N - B)))
            n = total_observed
            b = total_expected_bkg
            if n > b:
                z = np.sqrt(2 * (n * np.log(n / b) - (n - b)))
            else:
                z = 0.0 # No excess
        else:
            z = np.nan
            
        return total_observed, total_expected_bkg, z

def perform_bump_hunt(mjj_data, prefix=""):
    """
    Utility wrapper to run the bump hunt and return a formatted dict.
    """
    hunter = BumpHunter()
    obs, exp, sig = hunter.fit_background(mjj_data)
    
    return {
        f'{prefix}observed_events': float(obs),
        f'{prefix}expected_bkg': float(exp),
        f'{prefix}local_significance': float(sig)
    }
