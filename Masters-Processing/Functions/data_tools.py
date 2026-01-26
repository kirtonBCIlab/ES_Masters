# Import libraries
import os
import mne
import pyxdf
import numpy as np
import pandas as pd
import scipy.signal as signal

def import_data(xdf_file: str) -> tuple[mne.io.Raw, pd.DataFrame, pd.DataFrame]:
    """
    Imports xdf file and returns the EEG stream, Python response, and Unity stream

    Parameters
    ----------
        xdf_file: str

    Returns
    -------
        eeg_mne: mne.io.Raw
            EEG data in an MNE format
        python_stream: pd.DataFrame
            DataFrame with the stream of events sent from the BCI-essentials-python backend
        unity_stream: pd.DataFrame
            DataFrame with the stream of events sent from the BCI-essentials-unity frontend
    """

    [xdf_data, _] = pyxdf.load_xdf(xdf_file)

    # First, determine first sample that needs to be subtracted from all streams
    for i in range(len(xdf_data)):
        stream_name = xdf_data[i]["info"]["name"][0]
        if (stream_name == "DSI24") | (stream_name == "DSI7"):
            first_sample = float(xdf_data[i]["footer"]["info"]["first_timestamp"][0])

    # Second, separate all streams and save data
    for i in range(len(xdf_data)):
        stream_name = xdf_data[i]["info"]["name"][0]

        # Data stream
        if (stream_name == "DSI24") | (stream_name == "DSI7"):
            eeg_np = xdf_data[i]["time_series"].T
            srate = float(xdf_data[i]["info"]["nominal_srate"][0])
            n_chans = len(xdf_data[i]["info"]["desc"][0]["channels"][0]["channel"])
            ch_names = [
                xdf_data[i]["info"]["desc"][0]["channels"][0]["channel"][c]["label"][0]
                for c in range(n_chans)
            ]
            ch_types = "eeg"

            # Drop trigger channel
            eeg_np = eeg_np[:-1, :]
            ch_names = ch_names[:-1]

            # Create MNE object
            info = mne.create_info(ch_names, srate, ch_types)
            eeg_mne = mne.io.RawArray(eeg_np, info)

        # Unity stream
        if stream_name == "PythonResponse":
            python_series = xdf_data[i]["time_series"]
            python_time_stamps = xdf_data[i]["time_stamps"]

            dict_python = {
                "Time stamps": np.array(python_time_stamps) - first_sample,
                "Signal value": np.zeros(len(python_time_stamps)),
                "Event": [event[0] for event in python_series],
            }

            python_stream = pd.DataFrame(dict_python)

        # Unity events
        if stream_name == "UnityEvents":
            unity_series = xdf_data[i]["time_series"]
            unity_time_stamps = xdf_data[i]["time_stamps"]

            dict_unity = {
                "Time stamps": np.array(unity_time_stamps) - first_sample,
                "Signal value": np.zeros(len(unity_time_stamps)),
                "Event": [event[0] for event in unity_series],
            }

            unity_stream = pd.DataFrame(dict_unity)

    return (eeg_mne, python_stream, unity_stream)

def create_epochs(
    eeg_data: mne.io.Raw,
    eeg_ts: float,
    markers: pd.DataFrame,
    markers_ts: np.ndarray,
    events:list[str], 
    epoch_end: str,
    max_length: int= -1000,
    baseline: bool = False
    ) -> [list, np.ndarray]:
    """
    Creates Epoch data from the desired markers (can be Unity or Python stream)

    Parameters
    ----------
        eeg: mne.io.fiff.raw.Raw
            EEG raw array in an MNE format. Must include time vector.
        marker_ts: np.ndarray
            Time stamps of the markers
        markers: pd.DataFrame
            Markers used to create the epochs
        events: list[str]
            List of strings with the events to use for the epochs
        epoch_end: str
            The epochs will be trimmed to the next time this marker is found

    Returns
    -------
        events_list: list
        eeg_epochs: mne.Epochs
    """
    # Make sure that eeg is [channels, samples]
    if eeg_data.shape[0] > eeg_data.shape[1]:
        eeg_data = eeg_data.T

    # Initialize empty list to store epochs
    epochs = []
    event_list = []
    min_epoch_length = np.inf
    
    # Create event mask
    for event in events:
        event_mask = [event in marker_str[0].split(', ') for marker_str in markers]
        event_times = markers_ts[event_mask]

        for start_time in event_times:
            end_mask = [epoch_end in marker_str[0].split(', ') for marker_str in markers]
            end_times = markers_ts[end_mask]
            end_time = end_times[end_times > start_time].min()

            # Trim to indices in eeg time stamps
            start_idx = np.where(eeg_ts >= start_time)[0][0]
            end_idx = np.where(eeg_ts >= end_time)[0][0]
            epoch = eeg_data[:, start_idx:end_idx]

            # Store epoch and corresponding event
            epochs.append(epoch)
            event_list.append(event)

            # Update minimum epoch length
            min_epoch_length = min(min_epoch_length, epoch.shape[1])   

    # Trim baseline epochs to same length as max "on" epoch length supplied as max_length
    if baseline:
        if max_length < min_epoch_length:
            numpy_epochs = np.zeros((len(epochs), eeg_data.shape[0], max_length))
            for (e,epoch) in  enumerate(epochs):
                numpy_epochs[e,:,:] = epoch[:, :max_length]
        else:
            numpy_epochs = np.zeros((len(epochs), eeg_data.shape[0], int(min_epoch_length)))
            for (e,epoch) in  enumerate(epochs):
                numpy_epochs[e,:,:] = epoch[:, :min_epoch_length]

    # Trim epochs to same length (minimum length of "on" epochs)
    else:
        numpy_epochs = np.zeros((len(epochs), eeg_data.shape[0], int(min_epoch_length)))
        for (e,epoch) in  enumerate(epochs):
            numpy_epochs[e,:,:] = epoch[:, :min_epoch_length]

    return [event_list, numpy_epochs]

def create_epochs_resting(
    eeg_data: mne.io.Raw,
    eeg_ts: float,
    markers: pd.DataFrame,
    markers_ts: np.ndarray,
    events:list[str], 
    epoch_end: str
    ) -> [list, np.ndarray]:
    """
    Creates Epoch data from the desired markers (can be Unity or Python stream)

    Parameters
    ----------
        eeg: mne.io.fiff.raw.Raw
            EEG raw array in an MNE format. Must include time vector.
        marker_ts: np.ndarray
            Time stamps of the markers
        markers: pd.DataFrame
            Markers used to create the epochs
        events: list[str]
            List of strings with the events to use for the epochs
        epoch_end: str
            The epochs will be trimmed to the next time this marker is found

    Returns
    -------
        events_list: list
        eeg_epochs: mne.Epochs
    """
    # Make sure that eeg is [channels, samples]
    if eeg_data.shape[0] > eeg_data.shape[1]:
        eeg_data = eeg_data.T

    # Initialize empty list to store epochs
    epochs = []
    event_list = []
    min_epoch_length = np.inf
    
    # Create event mask
    for event in events:
        event_mask = [event in marker_str[0] for marker_str in markers]
        event_times = markers_ts[event_mask]

        for start_time in event_times:
            end_mask = [epoch_end in marker_str[0] for marker_str in markers]
            end_times = markers_ts[end_mask]
            end_time = end_times[end_times > start_time].min()

            # Trim to indices in eeg time stamps
            start_idx = np.where(eeg_ts >= start_time)[0][0]
            end_idx = np.where(eeg_ts >= end_time)[0][0]
            epoch = eeg_data[:, start_idx:end_idx]

            # Store epoch and corresponding event
            epochs.append(epoch)
            event_list.append(event)

            # Update minimum epoch length
            min_epoch_length = min(min_epoch_length, epoch.shape[1])   

    # Trim epochs to same length
    numpy_epochs = np.zeros((len(epochs), eeg_data.shape[0], int(min_epoch_length)))
    for (e,epoch) in  enumerate(epochs):
        numpy_epochs[e,:,:] = epoch[:, :min_epoch_length]

    return [event_list, numpy_epochs]

def split_epoch_into_multiple(epoch: np.ndarray, num_epochs: int) -> np.ndarray:
    """
    Splits a single epoch into multiple smaller epochs of equal length.

    Parameters
    ----------
    epoch: np.ndarray
        A 3D numpy array representing the original epoch with the shape (trials, channels, time_points).
    num_epochs: int
        The number of smaller epochs to split the original epoch into.

    Returns
    -------
    np.ndarray
        A 4D numpy array with the shape (trials, num_epochs, channels, time_points_per_epoch),
        where `time_points_per_epoch` is the length of the original epoch divided by num_epochs.
    """
    # Get the total number of time points in the original epoch
    total_time_points = epoch.shape[2]  # The third dimension represents time points
    
    # Calculate the time points per smaller epoch
    time_points_per_epoch = total_time_points // num_epochs
    
    # Initialize an array to store the split epochs
    split_epochs = np.zeros((epoch.shape[0], num_epochs, epoch.shape[1], time_points_per_epoch))
    
    # Split the epoch into smaller epochs
    for i in range(num_epochs):
        start_idx = i * time_points_per_epoch
        end_idx = (i + 1) * time_points_per_epoch
        split_epochs[:, i, :, :] = epoch[:, :, start_idx:end_idx]
    
    return split_epochs

def combine_epochs_by_label(events_epochs_individual, eeg_epochs_individual):
    """
    Combines epochs based on their labels by concatenating them along the time axis.

    Parameters
    ----------
    events_epochs_individual : list of str
        List of event labels corresponding to each epoch.
    eeg_epochs_individual : list of np.ndarray
        List of EEG epochs where each entry is (channels, time_samples).

    Returns
    -------
    unique_events : list of str
        List of unique event labels.
    combined_epochs : list of np.ndarray
        List of concatenated EEG epochs for each event.
    """
    unique_events = sorted(set(events_epochs_individual))
    combined_epochs = []

    for event in unique_events:
        event_indices = [i for i, evt in enumerate(events_epochs_individual) if evt == event]
        event_epochs = [eeg_epochs_individual[i] for i in event_indices]

        # Concatenate epochs along the time axis
        combined_epochs.append(np.concatenate(event_epochs, axis=1))

    return unique_events, combined_epochs


# EKL Additions
def import_data_byType_noPython(
    xdf_file: str,
) -> tuple[mne.io.Raw, pd.DataFrame, pd.DataFrame]:
    """
    Imports xdf file and returns the EEG stream, Python response, and Unity stream

    Parameters
    ----------
        xdf_file: str

    Returns
    -------
        eeg_mne: mne.io.Raw
            EEG data in an MNE format
        python_stream: pd.DataFrame
            DataFrame with the stream of events sent from the BCI-essentials-python backend
        unity_stream: pd.DataFrame
            DataFrame with the stream of events sent from the BCI-essentials-unity frontend
    """

    [xdf_data, _] = pyxdf.load_xdf(xdf_file)

    # First, we need to figure out the general type of data so we know what is what
    for stream in xdf_data:
        stream_type = stream["info"]["type"]

    # First, determine first sample that needs to be subtracted from all streams
    for i in range(len(xdf_data)):
        stream_type = xdf_data[i]["info"]["type"][0]
        if stream_type == "EEG":
            first_sample = float(xdf_data[i]["footer"]["info"]["first_timestamp"][0])

    # Second, separate all streams and save data
    for i in range(len(xdf_data)):
        stream_type = xdf_data[i]["info"]["type"][0]
        stream_name = xdf_data[i]["info"]["name"][0]

        # Data stream
        if stream_type == "EEG":
            headset = xdf_data[i]["info"]["name"][0]
            eeg_np = xdf_data[i]["time_series"].T
            srate = float(xdf_data[i]["info"]["nominal_srate"][0])
            n_chans = len(xdf_data[i]["info"]["desc"][0]["channels"][0]["channel"])
            ch_names = [
                xdf_data[i]["info"]["desc"][0]["channels"][0]["channel"][c]["label"][0]
                for c in range(n_chans)
            ]
            ch_types = "eeg"

            # Drop trigger channel
            eeg_np = eeg_np[:-1, :]
            ch_names = ch_names[:-1]

            # Create MNE object
            info = mne.create_info(ch_names, srate, ch_types)
            eeg_mne = mne.io.RawArray(eeg_np, info)

        # Unity events
        if stream_type == "LSL_Marker_Strings":
            unity_series = xdf_data[i]["time_series"]
            unity_time_stamps = xdf_data[i]["time_stamps"]

            dict_unity = {
                "Time Stamps": np.array(unity_time_stamps) - first_sample,
                "Signal Value": np.zeros(len(unity_time_stamps)),
                "Event": [event[0] for event in unity_series],
            }

            unity_stream = pd.DataFrame(dict_unity)

    # Changing return right now as a hardcode - but there should be a way to optionally include the outputs
    return (eeg_mne, unity_stream)

def create_epochs_es(
    eeg: mne.io.Raw, markers: pd.DataFrame, events=list[str]
) -> mne.Epochs:
    """
    Documentation will go here one day.
    This function is hardcoded right now to support Emily Schrag's Summer project.
    """

    eeg_epochs = 0
    return eeg_epochs

def epochs_from_unity_markers(
        eeg_time: np.ndarray,
        eeg_data: np.ndarray,
        marker_time: np.ndarray,
        marker_data: list[str]
        ) -> tuple((list[list[np.ndarray]], list)):
    """
    This function returns a list of EEG epochs and a list of marker names, based on
    the marker data provided.

    Notes
    -----
        The marker data must have repeated markers
    """
    # Make sure that data is in shape [samples, channels]
    if eeg_data.shape[0] < eeg_data.shape[1]:
        eeg_data = eeg_data.T

    # Initialize empty list
    eeg_epochs = []

    (repeated_markers, repeated_labels) = find_repeats(marker_data)

    # Trim EEG data to marker data times
    for m in range(np.shape(repeated_markers)[0]):
        eeg_mask_time = (eeg_time >= marker_time[repeated_markers[m, 0]]) & (
            eeg_time <= marker_time[repeated_markers[m, 1]]
        )

        eeg_epochs.append(eeg_data[eeg_mask_time, :])

    return (eeg_epochs, repeated_labels)


def find_repeats(marker_data: list) -> tuple((np.ndarray, list)):
    """
    Finds the repeated values in the marker data

    Returns
    -------
        - `repeats`: Numpy array with n-rows for repeated values [start, stop]
        - `order`: List with the `marker_data` labels of the repeated values.
    """

    repeats = []
    start = None

    for i in range(len(marker_data) - 1):
        if marker_data[i] == marker_data[i + 1]:
            if start is None:
                start = i
        elif start is not None:
            repeats.append((start, i))
            start = None

    if start is not None:
        repeats.append((start, len(marker_data) - 1))

    repeats = np.array(repeats)
    labels = [marker_data[i][0] for i in repeats[:, 0]]

    return repeats, labels

def get_tvep_stimuli(labels: list[str]) -> dict:
    """
        Returns a dictionary of unique labels of the stimulus of labels that begin with "tvep"

        Parameters
        ----------
            labels: list[str]
                Complete list of labels from Unity markers

        Returns
        -------
            unique_labels: list[str]
                List of unique labels of stimulus that begin with "tvep"
    """

    tvep_labels = []

    for label in labels:
        if label.split(",")[0] == "tvep":
            tvep_labels.append(label.split(",")[-1])
  
    dict_of_stimuli = {i: v for i, v in enumerate(list(set(tvep_labels)))}

    return dict_of_stimuli

def epochs_stim_freq(
    eeg_epochs: list,
    labels: list,
    stimuli: dict,
    freqs: dict,
    mode: str = "trim",
    ) -> list:
    """
        Creates EEG epochs in a list of lists organized by stimuli and freqs

        Parameters
        ----------
            eeg_epochs: list 
                List of eeg epochs in the shape [samples, chans]
            labels: list
                Complete list of labels from Unity markers
            stimuli: dict
                Dictionary with the unique stimuli labels
            freqs: dict
                Dictionary with the uniquie frequency labels
            mode: str
                Mode to convert all epochs to the same length,'trim' (default) or 'zeropad'

        Returns
            eeg_epochs_organized: list
                List of organized eeg epochs in the shape [stimuli][freqs][trials][samples, chans]
    """
    # Preallocate list for organized epochs
    eeg_epochs_organized = [[[] for j in range(len(freqs))] for i in range(len(stimuli))]
    mode_options = {"trim": np.min, "zeropad": np.max}
    mode_nsamples = {"trim": np.inf, "zeropad": 0}
    min_samples = np.inf

    # Organize epochs by stim and freq
    for e, epoch in enumerate(labels):
        for s, stim in stimuli.items():
            for f, freq in freqs.items():
                if epoch == f"tvep,1,-1,1,{freq},{stim}":
                    eeg_epochs_organized[s][f].append(np.array(eeg_epochs[e]))

                    # Get number of samples based on mode
                    nsamples = int(mode_options[mode]((mode_nsamples[mode], eeg_epochs[e].shape[0])))
                    mode_nsamples[mode] = nsamples

    # Change length of array based on mode
    for s, _ in stimuli.items():
        for f, _ in freqs.items():
            for t in range(3):  # For each trial
                if (mode == "trim"):
                    eeg_epochs_organized[s][f][t] = eeg_epochs_organized[s][f][t][:min_samples, :].T
                elif (mode == "zeropad"):
                    pad_length = nsamples - eeg_epochs_organized[s][f][t].shape[0]
                    pad_dimensions = ((0, pad_length), (0, 0))
                    eeg_epochs_organized[s][f][t] = np.pad(eeg_epochs_organized[s][f][t], pad_dimensions, 'constant', constant_values=0).T

    return np.array(eeg_epochs_organized)

def epochs_stim(eeg_epochs: list, labels: list, stimuli: dict) -> list:
    """
    Organizes EEG epochs into a list of lists based on stimulus.

    Parameters
    ----------
    eeg_epochs : list 
        List of EEG epochs in the shape [samples, chans].
    labels : list
        List of labels corresponding to each epoch.
    stimuli : dict
        Dictionary mapping unique stimuli indices to labels.

    Returns
    -------
    eeg_epochs_organized : list
        List of organized EEG epochs in the shape [stimuli][trials][samples, chans].
    """
    # Preallocate list for organized epochs
    eeg_epochs_organized = [[] for _ in range(len(stimuli))]

    # Organize epochs by stimulus
    for e, label in enumerate(labels):
        for stim_idx, stim_label in stimuli.items():
            if label == stim_label:
                eeg_epochs_organized[stim_idx].append(np.array(eeg_epochs[e]))
                break

    return [np.array(stim_list) for stim_list in eeg_epochs_organized]

def labels_to_dict_and_array(labels:list) -> tuple[dict, np.ndarray]:
    """
        Returns dictionary of labels with numeric code and numpy
        array with the label codes
    """
    # Create an empty dictionary for labels
    label_dict = {}
  
    # Loop through the list of strings
    for label in labels:
        # if the string is not in the dictionary, assign a new code to it
        if label not in label_dict:
            label_dict[label] = len(label_dict)
    
    # Create a numpy array with the codes of the strings
    arr = np.array([label_dict[label] for label in labels])
    
    return [label_dict, arr]

def trim_epochs(epochs:list) -> np.ndarray:
    """
        Takes a list of epochs of different length and trims to the shorter
        epoch. 
        
        Returns
        -------
            trimmed_epochs: array with shape [epochs, channels, samples]
    """
    # Initialize samples and channels counter
    min_samples = np.inf
    nchans = np.zeros(len(epochs), dtype=np.int16)
    
    # Get number of minimum samples
    for [e,epoch] in enumerate(epochs):
        epoch_shape = epoch.shape
        epoch_len = int(np.max(epoch_shape))
        nchans[e] = int(np.min(epoch_shape))

        min_samples = int(np.min((min_samples, epoch_len)))

    # Check that all epochs have same number of channels
    if (np.sum(np.abs(np.diff(nchans))) != 0):
        print("Not all epochs have the same number of channels")
        return None

    # Preallocate and fill output array
    trimmed_epochs = np.zeros((len(epochs), nchans[0], min_samples))
    for [e,epoch] in enumerate(epochs):
        # Make sure epoch is in shape [chans, samples]
        epoch_shape = epoch.shape
        if epoch_shape[0] > epoch_shape[1]:
            epoch = epoch.T

        trimmed_epochs[e,:,:] = epoch[:,:min_samples]

    return trimmed_epochs

def drop_epochs_by_label(epochs:list[np.ndarray], labels:list[str], label_to_drop:list[str]) -> tuple[list[np.ndarray], list[str]]:
    """
    Drops epochs with a specific label from the list of epochs.

    Parameters
    ----------
    epochs : List[np.ndarray]
        List of EEG epochs.
    labels : List[str]
        List of corresponding marker labels.
    label_to_drop : str
        Label of the epochs to be dropped.

    Returns
    -------
    Tuple[List[np.ndarray], List[str]]
        A tuple containing the modified list of EEG epochs and labels.
    """
    # Use list comprehension to filter out epochs with the specified label
    filtered_epochs = [epoch for epoch, label in zip(epochs, labels) if label not in label_to_drop]

    # Update the list of labels accordingly
    filtered_labels = [label for label in labels if label not in label_to_drop]

    return filtered_epochs, filtered_labels

def normalize_epochs_length(
    epochs:list,
    mode:str = "trim"
    ) -> np.ndarray:
    """
    Takes a list of epochs of different length and trims or zeropads all into the specified number of seconds 
    
    Parameters
    ----------
    epochs : list
        List of epochs where each epoch is an array with [channels, samples].
    mode : str
        Mode in which to normalize the epochs. Can be either "trim" or "zeropad".

    Returns
    -------
    normalized_epochs: np.ndarray
        array with shape [epochs, channels, samples] with the length of the 
        shortest ("trim") or longest ("zeropad") epoch.
    """
      
    # Preallocate list for organized epochs
    normalized_epochs = []
    transpose_flag = False
    mode_options = {"trim": np.min, "zeropad": np.max}
    mode_nsamples = {"trim": np.inf, "zeropad": 0}

    # If the first epoch is [samples, channels] assume that they will all be
    if epochs[0].shape[0] > epochs[0].shape[1]:
        transpose_flag = True

    # Get number of samples based on mode
    for epoch in epochs:
        epoch_nsamples = int(np.max(epoch.shape))
        nsamples = int(mode_options[mode]((mode_nsamples[mode], epoch_nsamples)))
        mode_nsamples[mode] = nsamples

    for epoch in epochs:
        # Transpose if needed for correct shape
        if transpose_flag:
            epoch = epoch.T

        # Trim or zeropad epochs to desired length
        if (mode == "trim"):
            normalized_epochs.append(epoch[:, :nsamples]) 
        elif (mode == "zeropad"):
            pad_length = nsamples - epoch.shape[-1]
            pad_dimensions = ((0,0), (0, pad_length))
            normalized_epochs.append(np.pad(
                epoch,
                pad_dimensions,
                'constant',
                constant_values = 0))

    # Return epochs to correct shape if needed
    normalized_epochs = np.array(normalized_epochs)

    return normalized_epochs