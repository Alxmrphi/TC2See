import os
import pickle 
import pandas as pd
import numpy as np

from pathlib import Path
from nilearn import surface as surf

# Suppress warnings
pd.options.mode.chained_assignment = None


idx_to_fname = pickle.load(open('idx_to_fname.pkl', 'rb'))
fname_to_idx = pickle.load(open('fname_to_idx.pkl', 'rb'))

# Helpers to add columns to the events dataframe
def process_stim_column(stimulus):
    assert type(stimulus) == str
    return stimulus.split('.')[1:-1][0]
    
def stimulus_to_class(stimulus):
    return fname_to_idx[stimulus]


def get_data_and_design_matrices(sub_no: int,
                                 func_folder: os.PathLike,
                                 events_folder: os.PathLike,
                                 hemi: str,
                                 run_no: int,
                                 n_timepoints: int) -> tuple:
    
    """ This helper function specifies a subject, a run, a hemisphere and returns
    the design matrix and functional data for that subject, run and hemisphere.
    
    Args:
        sub_no (int): The subject number
        func_folder (os.Pathlike): The path to the folder containing the functional data
        events_folder (os.Pathlike): The path to the folder containing the events data
        hemi (str): The hemisphere of the brain. Must be in {lh, rh}
        run_no (int): The run number
        n_timepoints (int): The number of timepoints in the functional data
    
    """

    assert hemi in ['lh', 'rh'], "Argument 'hemi' must be either 'lh' or 'rh'"
    hemi = 'L' if hemi == 'lh' else 'R'

    # Load functional data
    func_file = Path(f'sub-{sub_no}_task-bird_run-{run_no}_hemi-{hemi}_space-fsaverage_bold.func.gii')
    data = surf.load_surf_data(func_folder / func_file)

    # Load events data and process it
    file = Path(f'TC2See_{sub_no}_{run_no}_result_store.csv')
    path = events_folder / file
    df = pd.read_csv(path, sep='\t')
    events_df = df[df['stimulus'].str.endswith('png')] 
    events_df['filename'] = events_df['stimulus'].apply(process_stim_column)
    events_df['file_id'] = events_df['filename'].apply(stimulus_to_class)

    # Create design matrix
    n_conds = len(idx_to_fname.keys())
    assert n_conds == 300, "Number of conditions must be 300"
    design = np.zeros((n_timepoints, n_conds))

    # Each row in the events dataframe details presentation of a bird image.
    # For each row, we want to extract which TR this occurred at and a file ID.
    # The file ID is the mapping of the stimulus filename to the global stimulus class ID
    # found in `idx_to_fname.pkl` and `fname_to_idx.pkl`.
    # We then set the corresponding entry in the design matrix to 1.
    for t in range(len(events_df)):
        tr, idx = events_df.iloc[t][['tr', 'file_id']]
        tr = int(tr)
        design[tr, idx] = 1

    return design, data