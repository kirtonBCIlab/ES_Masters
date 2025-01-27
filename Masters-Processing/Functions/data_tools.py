"""
    Set of functions to work with the Boccia software pilot data
"""

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
    eeg: mne.io.Raw, markers: pd.DataFrame, time: list[float], events=list[str]
) -> mne.Epochs:
    """
    Creates MNE epoch data from the desired markers (can be Unity or Python stream)

    Parameters
    ----------
        eeg: mne.io.fiff.raw.Raw
            EEG raw array in an MNE format
        markers: pd.DataFrame
            Markers used to create the epochs
        time: list[float]
            Times used for epoch data, must be len==2
        events: list[str]
            List of strings with the events to use for the epochs

    Returns
    -------
        eeg_epochs: mne.Epochs
    """
    markers_np = np.zeros(markers.shape)
    markers_np[:, :2] = markers.iloc[:, [0, 1]]
    event_categories = pd.Categorical(markers["Event"])
    markers_np[:, 2] = event_categories.codes

    events_ids = np.zeros(markers.shape[0])
    for event in events:
        # - Missing implementation
        # We have the categories in the data frame, where the 3rd column can be set as categories
        # the code should look for the matching categories and provide just the indices that match the list of
        # events. Then, create the epochs based on the markers from the selected events

        # categories_of_interest = event_categories.categories.str.contains(event)
        # events_ids[markers_np[categories_of_interest.tolist(), :]] = 1
        break

    # eeg_epochs = mne.Epochs(eeg, )

    eeg_epochs = 0
    return eeg_epochs


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


def build_events_list_es(eeg: mne.io.Raw, markers: pd.DataFrame):
    """
    Documentation to be here one day.
    This just is a hardcoded build of a basic event list for Emily Schrag's summer project
    """
    unity_markers = markers["Event"]
    unity_time_stamps = markers["Time Stamps"]
    # using the below functions to find calls
    idx_markers = find_indices_of_repeated_values(unity_markers)
    idx_markers.sort()

    # Stimulus off calls happen every other block. So we will move those into their own index array, and not include the first one as it is before the stim started. Note that the first value has to be included for it to fold properly
    stim_off_markers = idx_markers[1:-1:2]
    # Make sure that the stim_off markers are true and only the stimulus off ones.
    for m in stim_off_markers:
        assert unity_markers[m] == "Stimulus Off"

    # Put them into a 2D matrix of shape Nx4, ignoring the first off call (pre-marker), so these now correspond to the "off" times between each of the 24 different blocks of stimulus
    grouped_stimOff_array = group_into_new_dimension(stim_off_markers, 4)

    # Repeat with the stim markers
    stim_markers = idx_markers[2:-1:2]
    # Ok ok ok. No we group this as well using our grouped call above, so we have each dimension as one of the "blocks", with the index for where each start of that block.
    grouped_stim = group_into_new_dimension(stim_markers, 4)

    # This should be length 24 right now
    # for i in range(0,len(ts_grouped_markers)):

    #     #Handle edge case here, cause the final one will require something different and we need data from the next value up in the groups.
    #     if i < len(ts_grouped_markers):
    #         nextOffVar = np.array(ts_grouped_offStim[i+1][0])
    #         offVar = ts_grouped_offStim[i][1:4]
    #         offVar = np.append(offVar,nextOffVar)
    #         #Create your local variables based on the outer loop
    #         stimVar = ts_grouped_markers[i]

    #         #Create an empty output for stim/off values for a given block
    #         output = [[],[]]

    #         #Now have an internal second loop grabbing the data and putting it into a different array
    #         for jj in range(len(stimVar)):
    #             output[0].extend(eeg[stimVar[jj]:offVar[jj]])
    #             if jj < len(stimVar)-1:
    #                 output[1].extend(eeg[offVar[jj]:stimVar[jj+1]])

    #         # print(ts_grouped_markers[i])
    #     else:
    #         #Handle the last case
    #         stimVar = ts_grouped_markers[i]
    #         offVar = ts_grouped_offStim[i][1:3]
    #         #Create an empty output for stim/off values for a given block
    #         output = [[],[]]
    #         #Now have an internal second loop grabbing the data and putting it into a different array
    #         for jj in range(len(offVar)):

    #             if jj < len(offVar):
    #                 output[0].extend(eeg[stimVar[jj]:offVar[jj]])
    #                 output[1].extend(eeg[offVar[jj]:stimVar[jj+1]])
    #             else:
    #                 output[0].extend(eeg[stimVar[jj]:-1])
    #                 output[1].extend(eeg[offVar[jj]:stimVar[jj+1]])

    return idx_markers, grouped_stim, grouped_stimOff_array


##EKL Additions
def find_indices_of_repeated_values(arr):
    result = []  # List to store the indices of the first occurrence
    previous_str = None

    for idx, item in enumerate(arr):
        item_str = str(item)  # Convert to string representation

        if item_str != previous_str:
            # If the current string is different from the previous one,
            # add the index to the result
            result.append(idx)

        previous_str = item_str  # Update the previous string

    return result


def find_indices_with_substring(strings_list, substring):
    indices = [idx for idx, string in enumerate(strings_list) if substring in string]
    return indices


def group_into_new_dimension(input_array, group_size):
    return [
        input_array[i : i + group_size] for i in range(0, len(input_array), group_size)
    ]


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

def fix_labels(labels: list[str]) -> list[str]:
    """
        Fix labels in pilot data (e.g., "tvep,1,-1,1,2Min", should be 
        "tvep,1,-1,1,2, Min")

        Parameters
        ----------
            labels: list[str]
                Original set of labels found in Unity LSL stream

        Returns
        -------
            fixed_labels: list[str]
                List of labels with mistakes fixed
    """

    # Preallocate output
    fixed_labels = []

    for label in labels:
        if label == "tvep,1,-1,1,2Min":
            fixed_labels.append("tvep,1,-1,1,2, Min")
        elif label == "tvep,1,-1,1,9.6Min":
            fixed_labels.append("tvep,1,-1,1,9.6, Min")
        elif label == "tvep,1,-1,1,16Min":
            fixed_labels.append("tvep,1,-1,1,16, Min")
        elif label == "tvep,1,-1,1,36Min":
            fixed_labels.append("tvep,1,-1,1,36, Min")
        else:
            fixed_labels.append(label)

    return fixed_labels

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
    # if transpose_flag:
    #     normalized_epochs = np.transpose(normalized_epochs, (0, 2, 1))

    return normalized_epochs

def read_xdf(file: str, picks: list[str]="all"):
    """
        Imports a .XDF file and returns the data matrix [channels x samples] and sample rate [Hz]

        Parameters
        ----------
            - file: str
                Full directory of the file to import
            - picks: list[str] = ["all"]
                List of strings with the names of the channels to import. Default will import all EEG channels
            - return_marker_data: bool
                If enabled, the function also returns the marker data and time stamps

        Returns
        -------
            - `eeg_ts`: EEG time stamps [sec]
            - `eeg`: np.ndarray [channels x samples]
                EEG raw data
            - `srate`: double
                Sampling rate [Hz]
            
    """
    file_path = os.path.normpath(file)  # Normalize path OS agnostic
    [data, header] = pyxdf.load_xdf(file_path, verbose=False)
    
    for stream in data:
        # Obtain data for SMARTING headset
        if (stream["info"]["source_id"][0]=="SMARTING" and stream["info"]["type"][0]=="EEG"):
            eeg_ts = stream["time_stamps"]
            eeg_np = stream["time_series"]
            srate = float(stream["info"]["nominal_srate"][0])
            break

        source_id_list = stream["info"]["source_id"][0].split("_")
        if source_id_list[0] == 'gUSBamp' and source_id_list[-1] != "markers":
            eeg_ts = stream["time_stamps"]
            eeg_np = stream["time_series"]
            srate = float(stream["info"]["nominal_srate"][0])
            break


    # Obtained from:
    # - https://mbraintrain.com/wp-content/uploads/2021/02/RBE-24-STD.pdf
    n_chans = len(stream['info']['desc'][0]['channels'][0]['channel'])
    chans_names = [stream['info']['desc'][0]['channels'][0]['channel'][i]['label'][0] for i in range(n_chans)]

    eeg_pd = pd.DataFrame(data=eeg_np, columns=chans_names)

    if picks != "all":
        eeg_pd = eeg_pd[picks]                    

    return eeg_ts, eeg_pd.to_numpy().T, srate

def read_xdf_unity_markers(file: str) -> tuple[np.ndarray, list[str]]:
    """
        This function returns the time stamps and markers from the Unity stream of an xdf file

        Returns
        -------
            - `marker_time`. Numpy vector with the time stamps of the Unity stream markers.
            - `marker_data`. List with the string of markers.
    """

    file_path = os.path.normpath(file)  # Normalize path OS agnostic
    [data, _] = pyxdf.load_xdf(file_path, verbose=False)

    for stream in data:
        if stream["info"]["name"][0] == 'UnityMarkerStream':
            marker_time = stream["time_stamps"]
            marker_data = stream["time_series"]  

    return marker_time, marker_data