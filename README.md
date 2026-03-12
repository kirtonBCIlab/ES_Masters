# ES_Masters

This repository contains a processing pipeline for EEG data and comfort‑ratings used to find each participant's Personal SSVEP stimulus.

## Directory structure

```
ES_Masters
├── Data/                  *****suggested file structure for using the processing notebooks
│   ├── EEG/
│   │   └── P001/…          
│   └── Comfort/
│       └── P001/…              
├── Functions/                  ← custom python modules used by notebooks
│   └── data_tools.py 
│   └── dataset_snr_helpers.py 
│   └── elo.py  
│   └── import_data.py          ← helper functions for filtering, averaging
│   └── processing.py           ← helper functions for filtering, averaging

├── Masters-Processing/         ← folder that contains the jupyter notebooks for the processing of EEG and Comfort data 
│   ├── preprocess.ipynb        ← import & clean raw EEG data, epoching, baselining
│   ├── zscore.ipynb            ← compute PSDs, z‑scores, export z-score.csv
│   └── get-best-stim.ipynb     ← merge z‑scores with comfort ratings, pick best stim
│   └── get-best-stim.py        ← online version of the processing pipeline, functionality of preprocess.ipynb, zscore.ipynb, and get-best-stim.ipynb
├── Masters-Unity/              ← folder that contains the unity project used to present the SSVEP Personalization Pipeline
│   └── ... 
└── README.md                   ← this file
```

## Data
- **`Data/EEG`** – raw EEG data organized by paarticipant ID. Exported raw
  `.npz` epoch files, baseline files, and `z-score.csv` live here.
- **`Data/Comfort`** – comfort‑rating CSVs organised by participant ID.

## Code
- **`Functions/processing.py`** – contains any reusable functions that the
  notebooks import (e.g. filtering routines, PSD helpers).
- **`Functions/data_tools.py`** - 
- **`Functions/dataset_snr_helpers.py`** - 
- **`Functions/elo.py`** - 
- **`Functions/import_data.py`** - 

- **`best-stim-real-time.py`**

## Notebooks
1. **`preprocess.ipynb`** – first notebook you run. It reads the raw
   `.vhdr/.eeg` files, applies filters, epochs the data, and saves compressed
   numpy files and a JSON settings file for each subject.

2. **`zscore.ipynb`** – second step. Loads the preprocessed npz/json files,
   averages occipital channels, computes power‑spectral densities, compares to
   a baseline to create z‑scores, and finally exports a `z-score.csv` per
   subject (with 10 Hz/20 Hz values for each stimulus).

3. **`get-best-stim.ipynb`** – final step for offline processing. It reads the
   `z-score.csv` produced above along with the comfort‑rating data, filters
   stimuli that exceed a z‑score threshold, and selects the stimulus with the
   lowest comfort score.

4. **`batch-dataset-snr.ipynb`** - 


## Usage
1. Install the Python dependencies listed in the notebooks (pandas, numpy,
   scipy, matplotlib, etc.).
    * Use dependencies.yml
2. Set up data in the suggested structure
3. Open `preprocess.ipynb` in VS Code and execute all cells to generate the
   `.npz`/`.json` files.
4. Run `zscore.ipynb` to compute and save the z‑scores.
5. Finally run `get-best-stim.ipynb` to derive each participant’s personal
   stimulus.

Modify the list of files/subjects at the top of each notebook to process different sets of participants.

---
