import mne
import numpy as np
from Functions import import_data
from Functions import data_tools
import argparse
from autoreject import Ransac 
import pandas as pd
import scipy.signal as signal

def import_eeg_data(file_path):
    """Imports EEG data from an XDF file and creates an MNE Raw object."""
    ch_names = ["Fz", "F4", "F8", "C3", "Cz", "C4", "T8", "P7", "P3", "P4", "P8", "PO7", "PO8", "O1", "Oz", "O2"]
    
    # Read EEG data
    eeg_ts, eeg_data, eeg_fs = data_tools.read_xdf(file_path, picks=ch_names)

    # Create MNE array
    info = mne.create_info(ch_names, eeg_fs, ch_types='eeg')
    mne_raw = mne.io.RawArray(eeg_data, info=info)

    # Set standard channel montage
    mne_raw.set_montage('standard_1020')

    return mne_raw, ch_names, eeg_ts, eeg_fs

def bandpass_filter(raw_data, ch_names, hpf_fc=0.5, lpf_fc=35):
    """Applies a bandpass filter to the EEG data.
        - pass in mne_raw from import_eeg_data()
        - pass in ch_names from import_eeg_data()
    """
    print("Filtering Data...")
    filt_raw = raw_data.copy().filter(l_freq=hpf_fc, h_freq=lpf_fc, picks=ch_names)
    return filt_raw

def detect_bad_channels(filt_raw, ch_names):
    """Detects bad EEG channels using RANSAC and removes them."""
    print("Detecting bad channels using RANSAC...")

    # Initialize RANSAC
    ransac = Ransac(verbose=True, picks="eeg", n_jobs=1, min_corr=0.75, n_resample=100, min_channels=0.25)

    # Create epochs with preloading
    epochs = mne.make_fixed_length_epochs(filt_raw, duration=2, overlap=0.5, preload=True)

    # Fit RANSAC and detect bad channels
    ransac.fit(epochs)
    bad_channels = ransac.bad_chs_

    # Remove bad channels
    filt_clean = filt_raw.copy().drop_channels(bad_channels)
    
    # Save clean channel list
    ch_names_clean = [ch for ch in ch_names if ch not in bad_channels]

    return filt_clean, ch_names_clean

def create_epochs(filt_clean, eeg_ts, file_path):
    """Creates epochs for stimuli and baseline (off) events."""
    
    # Define event markers for stimuli
    list_of_events = [f"Contrast{x+1}Size{y+1}" for x in range(4) for y in range(3)]
    epoch_end = "stimulus ended"

    # Define baseline event
    list_of_off_events = ["stimulus ended"]
    epoch_end_off = "baseline ended"

    # Load markers
    marker_ts, markers = import_data.read_xdf_unity_markers(file_path)

    #TEMP
    markers = markers[:-1] # Dan practice data has an extra marker at the end that is not needed
    marker_ts = marker_ts[:-1] # Remove the last marker timestamp and label


    # Create epochs for stimuli
    events_epochs, eeg_epochs = data_tools.create_epochs(
        eeg_data=filt_clean.get_data(),
        eeg_ts=eeg_ts,
        markers=markers,
        markers_ts=marker_ts,
        events=list_of_events,
        epoch_end=epoch_end
    )

    # Create epochs for baseline (off)
    baseline_epochs, eeg_baseline = data_tools.create_epochs(
        eeg_data=filt_clean.get_data(),
        eeg_ts=eeg_ts,
        markers=markers,
        markers_ts=marker_ts,
        events=list_of_off_events,
        epoch_end=epoch_end_off
    )

    # Map stimuli and baseline
    dict_of_stimuli = {i: event for i, event in enumerate(list_of_events)}
    dict_of_stimuli_off = {0: "stimulus ended"}

    # Organize epochs
    eeg_epochs_organized = data_tools.epochs_stim(
        eeg_epochs=eeg_epochs,
        labels=events_epochs,
        stimuli=dict_of_stimuli
    )

    baseline_eeg_epochs_organized = data_tools.epochs_stim(
        eeg_epochs=eeg_baseline,
        labels=baseline_epochs,
        stimuli=dict_of_stimuli_off
    )

    return eeg_epochs_organized, dict_of_stimuli, baseline_eeg_epochs_organized, dict_of_stimuli_off

def prepare_data_structures(eeg_epochs_organized, dict_of_stimuli, 
                            baseline_eeg_epochs_organized, dict_of_stimuli_2, 
                            file_name, eeg_fs, ch_names, bad_channels, ch_names_clean, markers):

    # Structure for baseline epochs
    baseline_data = {stim_label: np.array(baseline_eeg_epochs_organized[stim_idx]) 
                     for stim_idx, stim_label in dict_of_stimuli_2.items()}

    # Structure for stimulus epochs
    stimulus_data = {stim_label: np.array(eeg_epochs_organized[stim_idx]) 
                     for stim_idx, stim_label in dict_of_stimuli.items()}

    # Structure for JSON-equivalent settings
    settings_data = {
        "file_name": file_name,
        "eeg_srate": eeg_fs,
        "ch_names": ch_names,
        "bad_chans": bad_channels,
        "new_ch_names": ch_names_clean,
        "labels": markers,
        "stimuli": dict_of_stimuli
    }

    return stimulus_data, baseline_data, settings_data

def average_occipital(settings_data, stimulus_data, baseline_data):
    """Averages occipital channels for each stimulus and baseline."""
    selected_channels = ['O1', 'Oz', 'O2']
    # make sure the channels are present in the new channel names, skip if not, get the indices of the channels that are present
    selected_channels = [channel for channel in selected_channels if channel in settings_data['new_ch_names']]
    #print if any channels are missing
    if len(selected_channels) < 3:
        print(f"Warning: Some occipital channels are missing. Only using {len(selected_channels)} channels.")
    channel_indices = [settings_data['new_ch_names'].index(channel) for channel in selected_channels]

    occipital_epochs =  {} # List of dictionaries for each file
    for stim_label in stimulus_data:
            # Extract occipital channels
            occipital_epochs[stim_label] = stimulus_data[stim_label][:, channel_indices, :]

            # Average over occiopital channels for each epoch for each stimulus
            occipital_epochs[stim_label] = np.mean(occipital_epochs[stim_label], axis=1)

    occipital_epochs_baseline =  {} # List of dictionaries for each file
    for stim_label in baseline_data:
            # Extract occipital channels
            occipital_epochs_baseline[stim_label] = baseline_data[stim_label][:, channel_indices, :]

            # Average over occiopital channels for each epoch for each stimulus
            occipital_epochs_baseline[stim_label] = np.mean(occipital_epochs_baseline[stim_label], axis=1)

    return occipital_epochs, occipital_epochs_baseline

def psd(occipital_epochs, settings_data):
     # PSD settings
    window_size = 10  # 10 = 0.1 Hz resolution, 5 = 0.2 Hz resolution, 2 = 0.5 Hz resolution

    # Preallocate variables
    eeg_f = [None] 
    eeg_pxx = [None] 

    eeg_f = {}
    eeg_pxx = {}

    # Compute PSD for each stimulus
    for stim_label, epochs in occipital_epochs.items():  # Use occipital_epochs instead of eeg_epochs
        eeg_f[stim_label] = []
        eeg_pxx[stim_label] = []

        # Compute PSD for each epoch
        for epoch in epochs:  # Each epoch is now a 1D array (num_samples,)
            f_values, pxx_values = signal.welch(
            x=epoch,  # 1D array (samples,)
                fs=settings_data["eeg_srate"],
                nperseg=window_size * settings_data["eeg_srate"],
                noverlap=(window_size * settings_data["eeg_srate"]) * 0.5,  # 50% overlap
            )
            eeg_f[stim_label].append(f_values)
            eeg_pxx[stim_label].append(pxx_values)

        # Convert lists to arrays for consistency
        eeg_f[stim_label] = np.array(eeg_f[stim_label])  # Shape: (num_epochs, num_frequencies)
        eeg_pxx[stim_label] = np.array(eeg_pxx[stim_label])  # Shape: (num_epochs, num_frequencies)

    return eeg_f, eeg_pxx

def baseline_mean_sd(occipital_epochs_baseline, settings_data):
    # PSD settings
    window_size = 10  # 10 = 0.1 Hz resolution, 5 = 0.2 Hz resolution, 2 = 0.5 Hz resolution

    # Preallocate variables
    baseline_f = [None] 
    baseline_pxx = [None] 

    # Compute PSD for each file
    baseline_f = {}
    baseline_pxx = {}

    # Compute PSD for each stimulus
    for stim_label, epochs in occipital_epochs_baseline.items():  # Use occipital_epochs instead of eeg_epochs
        baseline_f[stim_label] = []
        baseline_pxx[stim_label] = []

        # Compute PSD for each epoch
        for epoch in epochs:  # Each epoch is now a 1D array (num_samples,)
            f_values, pxx_values = signal.welch(
                x=epoch,  # 1D array (samples,)
                fs=settings_data["eeg_srate"],
                nperseg=window_size * settings_data["eeg_srate"],
                noverlap=(window_size * settings_data["eeg_srate"]) * 0.5,  # 50% overlap
            )
            baseline_f[stim_label].append(f_values)
            baseline_pxx[stim_label].append(pxx_values)

        # Convert lists to arrays for consistency
        baseline_f[stim_label] = np.array(baseline_f[stim_label])  # Shape: (num_epochs, num_frequencies)
        baseline_pxx[stim_label] = np.array(baseline_pxx[stim_label])  # Shape: (num_epochs, num_frequencies)

        # get the mean and std of the baseline data
        baseline_mean = np.mean(baseline_pxx[stim_label], axis=0)
        baseline_std = np.std(baseline_pxx[stim_label], axis=0)

    return baseline_mean, baseline_std

def get_zscore(eeg_pxx, eeg_f, baseline_mean, baseline_std):
    eeg_zscore_mean_pxx = {}  # Dictionary to store per-stimulus z-scores averaged across epochs

    # Compute Z-score for each stimulus
    for stim_label in eeg_pxx.keys():
        eeg_pxx_values = eeg_pxx[stim_label]  # Already averaged over channels, shape (num_epochs, freqs)

        zscores_per_epoch = []  # Store z-scores for each epoch

        for epoch in range(len(eeg_pxx_values)):  
            # Compute z-score for the current epoch
            zscore_epoch = (eeg_pxx_values[epoch, :] - baseline_mean) / baseline_std  # Shape: (freqs,)
            zscores_per_epoch.append(zscore_epoch)  # Store the z-scores for this epoch

        # Convert to array for consistency (epochs, freqs)
        zscores_per_epoch = np.array(zscores_per_epoch)  # Shape (num_epochs, 1281)

        # Calculate the mean z-score for each stimulus
        mean_zscore = np.mean(zscores_per_epoch, axis = 0) # Shape (num_epochs, 1281)

        # Store z-scores per stimulus (averaged across epochs)
        eeg_zscore_mean_pxx[stim_label] = mean_zscore

    freqs = eeg_f[stim_label][0] 

    # Create a mask for both 10 Hz and 20 Hz
    fmask = (freqs == 10.0) | (freqs == 20.0)
    ten_twenty_mean_zscores = {}


    # Compute filtered mean Z scores for each stimulus
    for stim_label in eeg_zscore_mean_pxx.keys():
        zscores = eeg_zscore_mean_pxx[stim_label]

        # Use the mask to get values at 10 Hz and 20 Hz
        selected = zscores[fmask]

        # Store as a list or tuple
        ten_twenty_mean_zscores[stim_label] = selected.tolist()

    stimuli = ten_twenty_mean_zscores.keys()

    mean_zscores = {stim: [[], []] for stim in stimuli}  # index 0 = 10Hz, 1 = 20Hz

    for f in range(len(ten_twenty_mean_zscores)):
        for stim in stimuli:
            mean_zscores[stim][0].append(ten_twenty_mean_zscores[stim][0])  # 10 Hz
            mean_zscores[stim][1].append(ten_twenty_mean_zscores[stim][1])  # 20 Hz

    z_scores = pd.DataFrame([
        [stim for stim in mean_zscores.keys()],
        [mean_zscores[stim][0][0] for stim in mean_zscores.keys()],  # 10 Hz
        [mean_zscores[stim][1][0] for stim in mean_zscores.keys()]   # 20 Hz
    ])    

    return z_scores

def get_mean_comfort(comfort_file):
    absolute_comfort_data = pd.read_csv(comfort_file)
    
    # Compute the mean Comfort_Value for each (Contrast, Size) pair
    df_grouped = absolute_comfort_data.groupby(["Contrast", "Size"])["Comfort_Value"].mean().reset_index()
    df_pivot = df_grouped.pivot_table(index=None, columns=["Contrast", "Size"], values="Comfort_Value")
    df_pivot.columns = [f"Contrast{c}Size{s}" for c, s in df_pivot.columns]
    df_final = df_pivot.reset_index(drop=True)

    return df_final


def get_best_stim(absolute_comfort_data, z_scores):
    best_stim_absolute = None
    kept_stim = []

    stim_name_list = ["Contrast1Size1", "Contrast1Size2", "Contrast1Size3", "Contrast2Size1", "Contrast2Size2", "Contrast2Size3", "Contrast3Size1", "Contrast3Size2", "Contrast3Size3", "Contrast4Size1", "Contrast4Size2", "Contrast4Size3"]

    for idx, row in absolute_comfort_data.iterrows():
        absolute_comfort_list = [row.Contrast1Size1, row.Contrast1Size2, row.Contrast1Size3, row.Contrast2Size1, row.Contrast2Size2, row.Contrast2Size3, row.Contrast3Size1, row.Contrast3Size2, row.Contrast3Size3, row.Contrast4Size1, row.Contrast4Size2, row.Contrast4Size3]
        
    # Get z-score 10 Hz and 20 Hz rows explicitly
    row_10 = z_scores.iloc[1]
    row_20 = z_scores.iloc[2]

    absolute_comfort_dict = dict(zip(stim_name_list, absolute_comfort_list))
    z_score_dict_10 = dict(zip(stim_name_list, row_10))
    z_score_dict_20 = dict(zip(stim_name_list, row_20))

    # Filter stims with z-score >= 2
    for stim, z_score10 in z_score_dict_10.items():
        for stim, z_score20 in z_score_dict_20.items():
            if z_score10 >= 2 or z_score20 >= 2:
                # Check if the stim is already in the kept_stim list
                if stim not in kept_stim:
                    # Append the stim to the kept_stim list
                    kept_stim.append(stim)

    # Find the stim (with Z-score > 2) with the highest absolute comfort score
    max_absolute_comfort = float("-inf")

    for stim in kept_stim:
        if absolute_comfort_dict[stim] > max_absolute_comfort:
            max_absolute_comfort = absolute_comfort_dict[stim]
            best_stim_absolute = stim

    return best_stim_absolute

def main(file_path, comfort_file):
    # Import EEG data
    mne_raw, ch_names, eeg_ts, eeg_fs = import_eeg_data(file_path)
    
    # Apply bandpass filter to EEG data
    filt_raw = bandpass_filter(mne_raw, ch_names)
    
    # Detect bad channels using RANSAC
    filt_clean, ch_names_clean = detect_bad_channels(filt_raw, ch_names)
    
    # Create epochs for stimuli and baseline events
    eeg_epochs_organized, dict_of_stimuli, baseline_eeg_epochs_organized, dict_of_stimuli_off = create_epochs(filt_clean, eeg_ts, file_path)
    
    # Prepare the data structures for stimuli and baseline epochs
    stimulus_data, baseline_data, settings_data = prepare_data_structures(
        eeg_epochs_organized, dict_of_stimuli, baseline_eeg_epochs_organized, dict_of_stimuli_off,
        file_name=file_path, eeg_fs=eeg_fs, ch_names=ch_names, bad_channels=[], ch_names_clean=ch_names_clean,
        markers=[]
    )
    
    # Average occipital channels for each stimulus and baseline
    occipital_epochs, occipital_epochs_baseline = average_occipital(settings_data, stimulus_data, baseline_data)
    
    # Compute Power Spectral Density (PSD)
    eeg_f, eeg_pxx = psd(occipital_epochs, settings_data)
    
    # Compute baseline mean and standard deviation
    baseline_mean, baseline_std = baseline_mean_sd(occipital_epochs_baseline, settings_data)
    
    # Get z-scores for the stimuli
    z_scores = get_zscore(eeg_pxx, eeg_f, baseline_mean, baseline_std)
    
    # Get the best stimulus based on the comfort and z-scores
    abs_comfort = get_mean_comfort(comfort_file)
    best_stim = get_best_stim(abs_comfort, z_scores)

    stimuli = ["Contrast1Size1", "Contrast1Size2", "Contrast1Size3", "Contrast2Size1", "Contrast2Size2", "Contrast2Size3", "Contrast3Size1", "Contrast3Size2", "Contrast3Size3", "Contrast4Size1", "Contrast4Size2", "Contrast4Size3"]

    # Print list of stimuli with z-scores and comfort values
    print("Stimuli with Z-scores and Comfort Values:")
    for idx, stim in enumerate(stimuli):  # Use enumerate to get both index and stimulus
        print(f"{stim}: Z-score 10Hz: {z_scores.iloc[1, idx]}, Z-score 20Hz: {z_scores.iloc[2, idx]}")
    print(f"The best stimulus based on comfort and z-scores: {best_stim}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Stimulus Comfort Analysis")
    parser.add_argument("file_path", type=str, help="Path to the EEG data file")
    parser.add_argument("comfort_file", type=str, help="Path to the comfort data file (CSV)")

    args = parser.parse_args()

    # Run the main function
    main(args.file_path, args.comfort_file)