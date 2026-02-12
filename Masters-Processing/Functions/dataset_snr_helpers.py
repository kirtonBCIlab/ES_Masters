import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================
def trim_epochs(first_stimulus_epochs_all):
    """
    Trim first stimulus epochs across all participants to the minimum sample length.
    
    Parameters:
    -----------
    first_stimulus_epochs_all : list
        List containing first stimulus epochs for each participant.
        Each element should be shape [n_trials, samples, channels]

    Returns:
    --------
    trimmed_epochs : list
        List with trimmed first stimulus epochs.
    min_sample_length : int
        Minimum sample length.
    """
    
    # Find min length across all participants
    min_sample_length = None
    all_lengths = []
    
    for participant_idx, participant_epochs in enumerate(first_stimulus_epochs_all):
        if participant_epochs is not None and len(participant_epochs) > 0:
            sample_length = participant_epochs.shape[-1]  # Get the sample dimension
            all_lengths.append(sample_length)
            
            if min_sample_length is None:
                min_sample_length = sample_length
            else:
                min_sample_length = min(min_sample_length, sample_length)
    
    print(f"Minimum sample length: {min_sample_length}")
    
    # Trim epochs to minimum length
    trimmed_epochs = []
    
    for participant_idx, participant_epochs in enumerate(first_stimulus_epochs_all):
        if participant_epochs is None or len(participant_epochs) == 0:
            trimmed_epochs.append(None)
            continue
        
        current_length = participant_epochs.shape[-1]
        
        if current_length > min_sample_length:
            # Trim from the end: [n_trials, samples, channels] -> trim samples dimension
            trimmed_participant = participant_epochs[:, :, :min_sample_length]
        else:
            trimmed_participant = participant_epochs
        
        trimmed_epochs.append(trimmed_participant)
    
    return trimmed_epochs

# ============================================================================
# SPECTRAL ANALYSIS FUNCTIONS
# ============================================================================
def compute_amplitude_spectrum(data_list, sampling_freq, window='hann'):
    """
    Compute amplitude spectrum for list of data arrays (each shape: [trials, samples]).
    
    Parameters:
    -----------
    data_list : list of numpy arrays
        List where each element is a participant's data with shape [trials, samples]
    sampling_freq : float
        Sampling frequency in Hz
    window : str
        Window function: 'hann', 'hamming', 'blackman', 'flattop', or 'none'
    
    Returns:
    --------
    freqs : numpy array
        Frequency vector
    participant_avg_spectra : list
        Average amplitude spectrum for each participant
    grand_avg_spectrum : numpy array
        Grand average amplitude spectrum across all participants
    participant_std : numpy array
        Standard deviation across participants
    all_individual_spectra : list
        List of ALL individual trial spectra for all participants (flat list)
    """
    participant_avg_spectra = []
    all_individual_spectra = []  # This will store ALL individual spectra from ALL participants
    
    # Process each participant
    for participant_data in data_list:
        if participant_data is None or len(participant_data) == 0:
            participant_avg_spectra.append(None)
            continue
            
        n_trials, n_samples = participant_data.shape
        trial_spectra = []
        
        # Process each trial
        for trial_data in participant_data:
            # Apply window function
            if window == 'hann':
                win = np.hanning(n_samples)
            elif window == 'hamming':
                win = np.hamming(n_samples)
            elif window == 'blackman':
                win = np.blackman(n_samples)
            elif window == 'flattop':
                win = np.flattop(n_samples)
            else:  # 'none' or any other string
                win = np.ones(n_samples)
            
            windowed_data = trial_data * win
            
            # Compute FFT
            fft_vals = np.fft.rfft(windowed_data)
            
            # Correct scaling for amplitude spectrum
            # Divide by N for correct amplitude, multiply by 2 for single-sided (except DC)
            amp_spectrum = np.abs(fft_vals) / n_samples
            amp_spectrum[1:] *= 2  # Double all except DC component
            
            # Store this trial's spectrum
            trial_spectra.append(amp_spectrum)
            all_individual_spectra.append(amp_spectrum)  # Add to the master list
        
        # Average spectra across trials for this participant
        if trial_spectra:
            participant_avg_spectrum = np.mean(trial_spectra, axis=0)
            participant_avg_spectra.append(participant_avg_spectrum)
        else:
            participant_avg_spectra.append(None)
    
    # Get frequency vector (using first valid participant)
    valid_data = [d for d in data_list if d is not None and len(d) > 0]
    if len(valid_data) == 0:
        return np.array([]), [], np.array([]), np.array([]), []
    
    n_samples = valid_data[0].shape[1]  # Assuming all have same sample length
    freqs = np.fft.rfftfreq(n_samples, 1/sampling_freq)
    
    # Average across participants (only valid participants)
    valid_spectra = [s for s in participant_avg_spectra if s is not None]
    if valid_spectra:
        all_spectra_array = np.array(valid_spectra)
        grand_avg_spectrum = np.mean(all_spectra_array, axis=0)
        participant_std = np.std(all_spectra_array, axis=0)
    else:
        grand_avg_spectrum = np.array([])
        participant_std = np.array([])
        
    return freqs, participant_avg_spectra, grand_avg_spectrum, participant_std, all_individual_spectra

def compute_snr_from_spectrum(freqs, spectrum, bandwidth=2, use_dB=True):
    """
    Compute SNR as amplitude at each frequency divided by 
    mean amplitude in the neighboring frequency band [f-1, f+1] Hz.
    
    Parameters:
    -----------
    freqs : numpy array
        Frequency vector
    spectrum : numpy array
        Amplitude spectrum (same length as freqs)
    bandwidth : float
        Bandwidth for noise calculation (default: 2 Hz)
    use_dB : bool
        If True, return SNR in dB (20 * log10(SNR_ratio))
        If False, return SNR as ratio
    
    Returns:
    --------
    snr : numpy array
        SNR values at each frequency (in dB if use_dB=True)
    noise_floor : numpy array
        Noise floor (mean amplitude in neighboring band)
    """
    snr = np.zeros_like(spectrum)
    noise_floor = np.zeros_like(spectrum)
    
    half_bandwidth = bandwidth / 2
    
    for i, f in enumerate(freqs):
        # Find frequencies within [f - half_bandwidth, f + half_bandwidth]
        # Exclude the frequency f itself
        mask = (freqs >= f - half_bandwidth) & (freqs <= f + half_bandwidth) & (freqs != f)
        
        if np.any(mask):
            # Calculate noise as mean amplitude in neighboring band
            noise_floor[i] = np.mean(spectrum[mask])
            
            # Calculate SNR: amplitude at f / noise floor
            # Add small epsilon to avoid division by zero
            if noise_floor[i] > 0:
                snr_ratio = spectrum[i] / noise_floor[i]
                # Convert to dB if requested
                if use_dB:
                    snr[i] = 20 * np.log10(snr_ratio)  # 20*log10 for amplitude ratio
                else:
                    snr[i] = snr_ratio
            else:
                snr[i] = np.nan if use_dB else 0
        else:
            # For edge cases where no neighboring frequencies exist
            noise_floor[i] = np.nan
            snr[i] = np.nan
    
    return snr, noise_floor

def compute_snr(data_list, freqs, individual_spectra, bandwidth=2, use_dB=True):
    """
    Compute SNR for each trial, average over trials for each participant, 
    then average across participants.
    
    Parameters:
    -----------
    data_list : list of numpy arrays
        Original data list with shape [trials, samples] for each participant
        Used to determine how many trials each participant has
    freqs : numpy array
        Frequency vector
    individual_spectra : list
        List of amplitude spectra for each trial (in the same order as data_list)
    bandwidth : float
        Bandwidth for noise calculation (default: 2 Hz)
    use_dB : bool
        If True, return SNR in dB
    
    Returns:
    --------
    grand_avg_snr : numpy array
        Average SNR across all participants (in dB if use_dB=True)
    participant_snrs : list
        SNR for each participant
    grand_avg_noise_floor : numpy array
        Average noise floor across all participants
    snr_se : numpy array
        Standard error of SNR across participants
    """
    participant_snrs = []
    participant_noise_floors = []
    
    # We need to know how many trials each participant has to correctly group the individual spectra
    trial_counts = []
    for participant_data in data_list:
        if participant_data is None or len(participant_data) == 0:
            trial_counts.append(0)
        else:
            trial_counts.append(len(participant_data))
    
    # Calculate SNR for each trial, then average by participant
    start_idx = 0
    for participant_idx, n_trials in enumerate(trial_counts):
        if n_trials == 0:
            participant_snrs.append(None)
            participant_noise_floors.append(None)
            continue
            
        # Extract spectra for this participant
        participant_spectra = individual_spectra[start_idx:start_idx + n_trials]
        trial_snrs = []
        trial_noise_floors = []
        
        # Compute SNR for each trial of this participant
        for spectrum in participant_spectra:
            if spectrum is not None:
                snr, noise_floor = compute_snr_from_spectrum(freqs, spectrum, bandwidth, use_dB)
                trial_snrs.append(snr)
                trial_noise_floors.append(noise_floor)
        
        # Average across trials for this participant
        if trial_snrs:
            snr_array = np.array(trial_snrs)
            noise_array = np.array(trial_noise_floors)
            
            participant_avg_snr = np.nanmean(snr_array, axis=0)
            participant_avg_noise_floor = np.nanmean(noise_array, axis=0)
            
            participant_snrs.append(participant_avg_snr)
            participant_noise_floors.append(participant_avg_noise_floor)
        else:
            participant_snrs.append(None)
            participant_noise_floors.append(None)
        
        # Move to next participant's spectra
        start_idx += n_trials
    
    # Now average across participants (only valid ones)
    valid_snrs = [s for s in participant_snrs if s is not None]
    valid_noise_floors = [n for n in participant_noise_floors if n is not None]
    
    if valid_snrs:
        snr_array = np.array(valid_snrs)
        noise_array = np.array(valid_noise_floors)
        
        grand_avg_snr = np.nanmean(snr_array, axis=0)
        grand_avg_noise_floor = np.nanmean(noise_array, axis=0)
        
        # Calculate standard error
        snr_std = np.nanstd(snr_array, axis=0)
        snr_se = snr_std / np.sqrt(len(valid_snrs))
    else:
        grand_avg_snr = np.array([])
        grand_avg_noise_floor = np.array([])
        snr_se = np.array([])
    
    return grand_avg_snr, participant_snrs, grand_avg_noise_floor, snr_se