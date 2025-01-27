"""
    Set of functions to import data

"""

import os
import mne
import numpy as np
import pandas as pd
import pyxdf

def select_importer(file: str, picks: list[str]="all"):
    """
        Automatically selects the right function to import data

        Parameters
        ----------
            file: str
                Complete file name of the file to import. Must have the file extension
            picks: list[str]
                List of strings with names of channels to import. Defaults to "all" channels
    """

    function_dict = {
        "edf":read_edf,
        "txt":read_openBCI,
        "xdf":read_xdf,
    }

    symbol = "\\"
    folder = symbol.join(file.split(symbol)[:-1])

    for format in function_dict.keys():
        temp_file = f"{file.split(symbol)[-1]}.{format}"
        if (temp_file in os.listdir(folder)):
            break

    extension = temp_file.split(".")
   

    [eeg, srate] = function_dict[extension[-1]](f"{file}.{extension[1]}", picks)

    return eeg, srate


def read_edf(file: str, picks: list[str] = ["all"]):
    """
        Imports a .EDF and returns the data matrix [channels x samples] and sample rate [Hz]
        
        Parameters
        ----------
            - file: str
                Full directory of file to import
            - picks: list
                List of strings with the names of the channels to import. Default will import all channels

        Returns
        -------
            - eeg: np.ndarray [channels x samples]
                EEG raw data
            - srate: double
                Sampling rate [Hz]

    """

    # if file.split(".")[-1] != "edf":
        # file = f"{file}.edf"

    edf_data = mne.io.read_raw_edf(file, verbose=False)
    eeg = edf_data.get_data(picks)       # EEG [V]
    srate = edf_data.info['sfreq']  # Sampple rate [Hz]

    return eeg, srate

def read_openBCI(file: str, picks: list[str] = "all"):
    """
        Imports a .TXT file and returns the data matrix [channels x samples] and sample rate [Hz]

        Parameters
        ----------
            - file: str
                Full directory of the file to import
            - picks: list[str] = ["all"]
                List of strings with the names of the channels to import. Default will import all EEG channels

        Returns
        -------
            - eeg: np.ndarray [channels x samples]
                EEG raw data
            - srate: double
                Sampling rate [Hz]
    """

    full_data = pd.read_csv(file, header=4)

    f = open(file)
    content = f.readlines()
    nchans = int(content[1].split(" = ")[1])                # Number of channels [int]
    srate = float(content[2].split(" = ")[1].split(" ")[0]) # Sampling rate [Hz]
    
    # Select only EEG channels or a subset of EEG channels
    eeg = full_data.iloc[:,1:nchans+1]
    chans_dict = {
        " EXG Channel 0":"FP1", " EXG Channel 1":"FP2", " EXG Channel 2":"F7", " EXG Channel 3":"F3",
        " EXG Channel 4":"F4", " EXG Channel 5":"F8", " EXG Channel 6": "T7", " EXG Channel 7":"C3", 
        " EXG Channel 8":"C4", " EXG Channel 9":"T8", " EXG Channel 10":"P7", " EXG Channel 11":"P3",
        " EXG Channel 12":"P4", " EXG Channel 13":"P8", " EXG Channel 14":"O1", " EXG Channel 15":"O2"
        }
    eeg.rename(columns=chans_dict, inplace=True)

    if picks != "all":
        eeg = eeg[picks]
    
    return eeg.to_numpy().T, srate

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

    [data, header] = pyxdf.load_xdf(file, verbose=False)
    
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
    # chans_dict = {
    #     0:"Fp1", 1:"Fp2", 2:"F3", 3:"F4", 4:"C3", 5:"C4", 6:"P3",
    #     7:"P4", 8:"O1", 9:"O2", 10:"F7", 11:"F8", 12:"T7", 13:"T8",
    #     14:"P7", 15:"P8", 16:"Fz", 17:"Cz", 18:"Pz", 19:"M1", 20:"M2",
    #     21:"AFz", 22:"CPz", 23:"POz",
    #     }

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

    [data, _] = pyxdf.load_xdf(file, verbose=False)

    for stream in data:
        if stream["info"]["name"][0] == 'UnityMarkerStream':
            marker_time = stream["time_stamps"]
            marker_data = stream["time_series"]  

    return marker_time, marker_data






