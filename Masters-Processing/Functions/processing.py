"""
    Processing
    --------------
    Set of functions to pre-process and process the SSVEP data

"""
import numpy as np
import scipy.signal as signal
from scipy.stats import chi2

def butter_filt(data:np.ndarray, fc:list, type:str, srate:float)-> np.ndarray:
    """
        Implements Butterworth digital filter with zero-phase

        Parameters
        ----------
            data: ndarray
                EEG data to filter
            fc: list
                Cut-off frequencies [Hz]
            type: str
                Type of filter to implement. "lowpass", "highpass", "bandpass", "bandstop"
            srate: float
                Sampling rate [Hz]

        Returns
        -------
            filt_data: nd.array
                Filtered EEG data
    """

    # Create filter
    sos = signal.butter(4, fc, type, fs=srate, output="sos")

    # Apply filter to longest dimension (i.e., time samples)
    axis = np.argmax(data.shape)
    filt_data = signal.sosfiltfilt(sos, data, axis=axis)

    return filt_data

def welch_confidence(x:np.ndarray, fs:float, nperseg:int, scaling:str,  pvalue:float = 0.95):
    """
        Implements pwelch estimate with power spectrum (pxx) with confidence interval, similar to MATLAB.
        Implementation shown [here](https://www.mathworks.com/matlabcentral/answers/522047-how-pwelch-computes-confidence-intervals).

        Parameters
        ----------
            x: np.ndarray
                Signal to calculate power spectrum
            fs: float
                Sampling frequency [Hz]
            nperseg: int
                Length of each segment
            scaling: str
                "density" for PSD, "spectrum" for power spectrum
            pvalue: float
                Coverage probability for the true PSD, specified as a scalar in the range (0,1).
                (Default = 0.95)

        Returns
        -------
            f: np.ndarray
                Frequency vector [Hz]
            pxx: np.ndarray
                Mean power spectrum
            pxxc: np.ndarray
                Power spectrum confidence levels. Size will be [pxx[:,0], 2*pxx[:,1]].
                Columns go [low, high] for confidence levels
    """

    # Calculate power spectrum
    [f, pxx] = signal.welch(x, fs, nperseg=nperseg, scaling=scaling)

    # Calculate confidence levels
    pxx_size = np.shape(pxx)
    pxxc = np.zeros((2*pxx_size[0], pxx_size[1]))
    
    x_size = np.shape(x)
    noverlap = nperseg // 2
    nsteps = nperseg - noverlap
    nseg = (np.max(x_size)-nperseg)//nsteps + 1
    k = 2*nseg  # Degrees of freedom

    for j,i in enumerate(range(0, pxx_size[0]+1, 2)):
        pxxc[i,:]   = pxx[j,:]*k/chi2.ppf((1+pvalue)/2, k)
        pxxc[i+1,:] = pxx[j,:]*k/chi2.ppf((1-pvalue)/2, k)

    return f, pxx, pxxc

def ssvep_snr(f:np.ndarray, pxx:np.ndarray, stim_freq:float, noise_band:float, nharms: int, db_out:bool):
    """
        Computes an SSVEP SNR as described in `Norcia et al. 2015`

        Parameters
        ----------
            f: ndarray
                Frequency vector [Hz]
            pxx: ndarray
                Power spectrum of EEG signal [uV^2]
            stim_freq: float
                SSVEP stimulation frequency [Hz]
            noise_band: float
                Range of single sided noise band [Hz]
            nharms: int
                Number of harmonics to use 
            db_out: bool
                Boolean to output in dB
    """
    # Check that the pxx is two dimensional and organize channels as rows
    if (pxx.ndim != 2):
        print("Warning: The input array does not have two dimensions.")
        return None
    
    if (pxx.shape[0] > pxx.shape[1]):
        pxx = pxx.T

    # Preallocate and initialize variables
    peaks_index = np.zeros(nharms+1)  # Peaks of SSVEP, including stimulation frequency
    pxx_signal = 0  # Signal power [V^2]
    pxx_noise = 0   # Noise power [V^2]
    fres = f[1]     # Frequency resolution [Hz]

    for h in range(nharms+1):
        norm_freq = np.abs(f-stim_freq*(h+1))
        peak_array = np.nonzero(norm_freq == np.min(norm_freq)) 
        peaks_index[h] = int(peak_array[0][0])    # Find peaks
        pxx_signal = pxx_signal + pxx[:,int(peaks_index[h])]

        noise_low = pxx[:,int(peaks_index[h]-np.floor(noise_band/fres)-1):int(peaks_index[h])-1]
        noise_high = pxx[:,int(peaks_index[h])+1:int(peaks_index[h]+np.floor(noise_band/fres)+1)]
        pxx_noise = pxx_noise + np.mean(np.concatenate([noise_low, noise_high], axis=1), axis=1)

    if db_out:
        return 10*np.log10(pxx_signal/pxx_noise)
    else:
        return pxx_signal/pxx_noise
    
def bandpass_filter(
    eeg:np.ndarray,
    srate:float,
    f_low:float,
    f_high:float,
    f_order:int = 5
    ) -> np.ndarray:
    """
        Applies a bandpass filter to the EEG data.

        Parameters:
            eeg: np.ndarray
                The EEG data. Shape should be [n_epochs, n_channels, n_samples].
            srate: float
                The sampling rate of the EEG data [Hz].
            f_low: float
                The low cutoff frequency [Hz].
            f_high: float
                The high cutoff frequency [Hz].
            f_order: int
                The order of the filter.

        Returns:
            filtered_data: np.ndarray
                The filtered EEG data.
    """

    # Create the bandpass filter
    sos = signal.butter(
        N = f_order,
        Wn = [f_low, f_high],
        btype = "band",
        fs = srate,
        output = "sos"
        )

    # Implement the bandpass filter
    filtered_data = signal.sosfiltfilt(
        sos= sos,
        x = eeg,
        )

    return filtered_data

def line_filter(eeg, srate, f_notch, f_order):

    # Create the notch filter
    [b,a] = signal.iirnotch(
        w0 = f_notch,
        Q = 30,
        fs = srate,
        )
    
    # Implement notch filter
    filtered = signal.filtfilt(
        b = b,
        a = a,
        x = eeg,
        )

    return filtered
