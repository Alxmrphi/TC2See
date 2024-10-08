import os
import sys
import time
from pprint import pprint
from pathlib import Path
from random import randint
import traceback

import numpy as np
import pandas as pd
# from ipywidgets import interact
from tqdm import tqdm
import nibabel as nib
# import glmsingle
# from glmsingle.glmsingle import GLM_single
import bids
from bids import BIDSLayout
from scipy.ndimage import zoom, binary_dilation
import h5py
import nibabel as nib
# from einops import rearrange

dir2 = os.path.abspath('../..')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: 
    sys.path.append(dir1)
    
from noise_ceiling import (
    compute_ncsnr,
    compute_nc,
)

from tc2see import load_data


dataset_root = Path('../data')

tc2see_version = 3 
derivatives_path = dataset_root / 'processed/fmriprep_vols'
num_runs = 6 if tc2see_version in (1, 3) else 8

# Initialize BIDSLayouts for querying files.
dataset_layout = BIDSLayout(dataset_root / 'raw_data/bids_data')
derivatives_layout = BIDSLayout(derivatives_path, derivatives=True, validate = False)

task = "bird"
space = 'T1w'
subjects = ['35', '36', '37', '38', '39', '40'] #, '04'] # ['05', '06', '07'] # Subject ID to process data for
tr = 2. #1.97 # TR duration (in seconds)
mask_dilations = 3 # Number of dilation iterations for the brain mask
num_stimuli = 75 #112  # Total number of different stimuli
num_trs = 236 #231 #229  # Total number of TRs in the fMRI data

# Load stimulus images and create a mapping of stimulus names to unique identifiers
stimulus_images = h5py.File(dataset_root / 'stimulus-images.hdf5', 'r')
stimulus_id_map = {name: i for i, name in enumerate(stimulus_images.attrs['stimulus_names'])}

# Create an HDF5 file to store preprocessed fMRI data
with h5py.File(f'/home/jamesmck/scratch/fmri_processing/results/derivatives_TC2See/fmriprep/tc2see-v3-bold.hdf5', 'a') as f:
    for sub in subjects:
        if f'sub-{sub}' not in list(f.keys()):
            try:
                group = f.require_group(f'sub-{sub}')
                
                # mask_image = derivatives_layout.get(
                #     subject=sub,
                #     run=1,
                #     task=task,
                #     space=space, 
                #     desc='brain',
                #     extension='nii.gz',
                # )[0].get_image()
                mask_image_files = derivatives_layout.get(
                    subject=sub,
                    run=1,
                    task=task,
                    space=space, 
                    extension='nii.gz'
                )
                mask_image = [file for file in mask_image_files if "space-T1w_desc-brain_mask.nii.gz" in file.filename]
                mask_image = mask_image[0].get_image()
                fmri_mask = mask_image.get_fdata().astype(bool)
                fmri_mask = binary_dilation(fmri_mask, iterations=mask_dilations)

                num_voxels = fmri_mask.sum()
                
                # If necessary attributes and datasets don't exist in the group, create them
                if 'affine' not in group:
                    group['affine'] = mask_image.affine
                
                #H, W, D = fmri_mask.shape
                if 'fmri_mask' not in group:
                    group['fmri_mask'] = fmri_mask
                    
                group.require_dataset('bold', shape=(num_runs, num_trs, num_voxels), dtype='f4')
                group.require_dataset('bold_mean', shape=(num_runs, num_voxels), dtype='f4')
                group.require_dataset('bold_std', shape=(num_runs, num_voxels), dtype='f4')
                group.require_dataset('bold_trend', shape=(num_runs, 2, num_voxels), dtype='f4')
                group.require_dataset('bold_trend_std', shape=(num_runs, num_voxels), dtype='f4')
                group.require_dataset('stimulus_trs', shape=(num_runs, num_stimuli), dtype='f4')
                group.require_dataset('stimulus_ids', shape=(num_runs, num_stimuli), dtype='i4')
                
                for run_id in range(num_runs):
                    # Load the preprocessed fMRI data for the current subject and run
                    bids_image_files = derivatives_layout.get(
                        subject=sub,
                        run=run_id + 1,
                        space=space, 
                        task=task, 
                        extension='nii.gz',
                    )
                    bids_image = [file for file in bids_image_files if "space-T1w_desc-preproc_bold.nii.gz" in file.filename]
                    bids_image = bids_image[0]
                    
                    bold = bids_image.get_image().get_fdata()
                    bold = bold[fmri_mask].T  # Extract the relevant voxels
                    num_trs_run = bold.shape[0]
                    trend_coeffs = np.stack([np.arange(num_trs_run), np.ones(shape=num_trs_run)], axis=1)
                    
                    # Perform linear detrending on the bold data
                    bold_trend = np.linalg.lstsq(trend_coeffs, bold, rcond=None)[0]
                    bold_predicted = trend_coeffs @ bold_trend
                    bold_detrend = bold - bold_predicted

                    # Load events data for the current subject and run
                    events_file = dataset_layout.get(
                        subject=sub,
                        run=run_id + 1,
                        task=task,
                        extension='tsv'
                    )[0]
                    
                    events_df = pd.read_csv(events_file.path, sep='\t')
                    events_df = events_df[events_df['stimulus'] != '+']
                    stimulus_names = [Path(stimulus_path).stem for stimulus_path in events_df['stimulus']]
                    stimulus_names = [
                        name[:name.find('hash')-1] if "hash" in name else name
                        for name in stimulus_names
                    ]
                    stimulus_ids = [stimulus_id_map[name] for name in stimulus_names]
                    
                    stimulus_trs = np.array(events_df['tr']).astype(np.float32)
                    
                    # Store various datasets in the HDF5 file
                    group['bold'][run_id, :num_trs_run] = bold
                    group['bold_mean'][run_id] = bold.mean(axis=0)
                    group['bold_std'][run_id] = bold.std(axis=0)
                    group['bold_trend'][run_id] = bold_trend
                    group['bold_trend_std'][run_id] = bold_detrend.std(axis=0)
                    group['stimulus_trs'][run_id] = stimulus_trs
                    group['stimulus_ids'][run_id] = stimulus_ids

            except Exception as e:
                print(f"\nError processing {sub}: {e}")
                traceback.print_exc()
                del f[f'sub-{sub}']
                continue
        else:
            print(f"Subject {sub} already exists")
            print(f[f'sub-{sub}']['bold'].shape)