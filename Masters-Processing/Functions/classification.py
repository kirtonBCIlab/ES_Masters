"""
    Set of functions to classify the EEG data
"""

# Import libraries
import mne
import numpy as np
from sklearn.pipeline import make_pipeline

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from moabb.pipelines import ExtendedSSVEPSignal
from sklearn.model_selection import cross_val_predict
from .ssvep_transformer import SSVEPTransformer

def rg_logreg(
    eeg_data: np.ndarray,
    labels: np.ndarray,
    cv:int = 3,
    ) -> np.ndarray:
    """
        Implements Riemmanian Geometry + Logistic regression classifier.
        Follows the example found in the [MOABB](https://moabb.neurotechx.com/docs/auto_examples/plot_cross_subject_ssvep.html#sphx-glr-auto-examples-plot-cross-subject-ssvep-py)
        SSVEP functions.

        Parameters
        ----------
        eeg_data: np.ndarray
            The EEG data. Shape should be [n_epochs, n_channels, n_samples].
        labels: np.ndarray
            The labels for each epoch.
        cv: int
            The number of cross-validation folds. Must be at least 2 and at most the number of samples.

        Returns
        -------
        predictions: np.ndarray
            The predicted labels.
    """

    # Apply MOABB pipeline
    pipe =  make_pipeline(
        Covariances(estimator="lwf"),
        TangentSpace(),
        LogisticRegression(solver="lbfgs", multi_class="auto")
        )
    
    predictions = cross_val_predict(pipe, eeg_data, labels, cv=cv)
    # predictions = pipe.fit(eeg_data, labels).predict(eeg_data)

    return predictions

def fb_rg_logreg(
    eeg_data:np.ndarray,
    stim_freqs:list[float],
    eeg_channels:list[str],
    srate:float,
    labels:np.ndarray,
    cv:int = 3,
    ) -> np.ndarray:
    """
        Implements Filter bank with  Riemmanian Geometry + Logistic regression classifier.
        Follows the example found in the [MOABB](https://moabb.neurotechx.com/docs/auto_examples/plot_cross_subject_ssvep.html#sphx-glr-auto-examples-plot-cross-subject-ssvep-py)
        SSVEP functions.

        Parameters
        ----------
        eeg_data: np.ndarray
            The EEG data. Shape should be [n_epochs, n_channels, n_samples].
        stim_freqs: list[float]
            The list of stimulation frequencies [Hz].
        eeg_channels: list[str]
            The list of EEG channel names.
        srate: float
            The sampling rate of the EEG data [Hz].
        labels: np.ndarray
            The labels for each epoch.
        cv: int
            Cross-validation folds.

        Returns
        -------
        predictions: np.ndarray
            The predicted labels.        
    """
    # Create filter bank signal
    filter_bank_eeg = filter_bank(eeg_data, stim_freqs, eeg_channels, srate)

    # Apply MOABB pipeline
    pipe =  make_pipeline(
        SSVEPTransformer(shape=filter_bank_eeg.shape[1:]),
        ExtendedSSVEPSignal(),
        Covariances(estimator="lwf"),
        TangentSpace(),
        LogisticRegression(solver="lbfgs", multi_class="auto")
        )
    
    # Set up stratified k-fold
    # stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    reshaped_filter_bank = np.reshape(filter_bank_eeg, (filter_bank_eeg.shape[0], -1))
    predictions = cross_val_predict(pipe, reshaped_filter_bank, labels, cv=cv)
    # predictions = pipe.fit(filter_bank_eeg, labels).predict(filter_bank_eeg)
    # predictions = pipe.fit(eeg_data, labels).predict(eeg_data)

    return predictions

def filter_bank(
    eeg_data:np.ndarray,
    freqs:list[float],
    eeg_channels:list[str],
    srate:float
    ):
    """
        Implements a filter bank to the EEG data.
        The filtered frequencies have cut-off frequencies of freqs +/- 0.5 Hz.
    """
    # Assuming the eeg_data is a numpy array with shape (n_epochs, n_channels, n_samples)
    n_epochs = eeg_data.shape[0]
    n_channels = eeg_data.shape[1]
    n_samples = eeg_data.shape[2]
    filtered_data = np.zeros((n_epochs, n_channels, n_samples, len(freqs)))

    for (e,epoch) in enumerate(eeg_data):
        for (f,freq) in enumerate(freqs):
            # Create an MNE Raw object
            info = mne.create_info(
                ch_names = eeg_channels,
                sfreq = srate,
                ch_types = "eeg",
                )
            raw = mne.io.RawArray(epoch, info, verbose=False)

            # Apply bandpass filter
            raw.filter(l_freq=freq-0.5, h_freq=freq+0.5, verbose=False)

            # Append filtered data to list
            filtered_data[e,:,:,f] = raw.get_data()

    return filtered_data

def expand_ssvep(eeg_data):
    """ Expands the input EEG data from shape [samples] into [frequencies, times, filter-bank]. """




