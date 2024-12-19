from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
# from scipy.ndimage import zoom, binary_dilation
import h5py
import nibabel as nib
from fracridge import FracRidgeRegressorCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import pickle
from bids import BIDSLayout
import argparse
from tc2see import load_data
from metrics import (
    cosine_distance, two_versus_two,
)
from noise_ceiling import (
    compute_ncsnr,
    compute_nc,
)

parser = argparse.ArgumentParser()
parser.add_argument('-subj', type=str, help='Subject number', required=True)
args = parser.parse_args()

subj = args.subj

tc2see_version = 3
glm_data_dir = Path("../../results/glm_single")
data_path = Path('../../data')
output_path = Path(f"../../results/roi_decoding_accuracies/individual_accuracy_dicts")
dataset_layout = BIDSLayout(data_path / 'raw_data/bids_data')
idx_to_fname = pickle.load(open(data_path / 'idx_to_fname.pkl', 'rb'))
fname_to_idx = pickle.load(open(data_path / 'fname_to_idx.pkl', 'rb'))
condition_tr_mapping = pickle.load(open(glm_data_dir / f'sub_{subj}/left_hem/sub-{subj}_condition_tr_mapping.pkl', 'rb'))

averaged = True
value_selection = 'R2'


ROIs = {
    "FFC": [18],"V1": [1],"V2": [4],"V3": [5],"V3A": [13],"V3B": [19],"V3CD": [158],"V4": [6],"V6": [3],"V7": [16],
    "V8": [7], "VMV1": [153],"VMV2": [160],"VMV3": [154],"LO1": [20],"LO2": [21],"PIT": [22],"VVC": [163], 
    "LO1": [20],"LO2": [21], "LO3": [159], "PHA1": [126], "PHA2": [155], "PHA3": [127], "IPS1": [17], "MT": [23]
}

ROI_combos = {
    "all_rois": [ roi for roi in range(1,181)],

    # High level Visual
    "FFC": [18], 
    "VVC": [163], 
    "LO1": [20], "LO2": [21], "LO3": [159], "PHA1": [126], 
    "PHA2": [155], "PHA3": [127], "IPS1": [17], "MT": [23],
    "LOC1_to_LOC3": [20, 21, 159],
    "PHA1_to_PHA3": [126, 155, 127],

    # Low level Visual
    "V1": [1],
    "V2": [4],
    "V3_V3A_V3B_V3CD": [5, 13, 19, 158],
    "V4": [6],
    "V6": [3],
    "V7": [16],
    "V8": [7]
}

glasser_L = nib.freesurfer.io.read_annot(data_path / "lh.HCPMMP1.annot")
glasser_R = nib.freesurfer.io.read_annot(data_path / "rh.HCPMMP1.annot")

ROI_masks = {}

for key, vals in ROI_combos.items():

    # mask glasser atlas to mark current loop ROI as 1s
    L_mask = np.isin(glasser_L[0], vals) # vals is a list of ROIs to set as 1
    R_mask = np.isin(glasser_R[0], vals)
    print(f"{key}: {sum(L_mask) + sum(R_mask)}")
    
    # concatenate left and right hemispheres 
    L_R_concat_mask = np.concatenate([L_mask, R_mask], axis=0)
    ROI_masks[key] = L_R_concat_mask

model_name = 'ViT-B=32'
embedding_name = 'embedding' 
num_runs = 6
tr = 2 # 1.97

with h5py.File(data_path / f'{model_name}-features.hdf5', 'r') as f:
    stimulus = f[embedding_name][:]

stimulus_images = h5py.File(data_path / 'stimulus-images.hdf5', 'r')
stimulus_id_map = {name.split('.')[1]: i for i, name in enumerate(stimulus_images.attrs['stimulus_names'])}

accuracies = {}
top_n_nc_vals = 256

print(f"====== Subject {subj} ======")
accuracies[subj] = {}
subject = f'sub-{subj}'


# Need stimulus information from the events file
events_files = dataset_layout.get(
    subject=str(subj),
    task="bird",
    extension='tsv'
)

event_dfs = []
for events_file in events_files:
    event_df = pd.read_csv(events_file, delimiter='\t')
    event_df["run"] = events_file.filename.split("_")[-2].split("-")[1]        
    event_dfs.append(event_df)
events = pd.concat(event_dfs, ignore_index=True)
events = events[events['stimulus'] != '+'].reset_index()[["stimulus", "run"]]
events["stimulus"] = events["stimulus"].apply(lambda x: x.split('/')[2].split('.')[1])


if averaged:    
    events = events.drop_duplicates(subset='stimulus', keep='first') # Drop duplicates when averaging betas
    event_stims_in_order = [idx_to_fname[cond_idx] for cond_idx, _ in condition_tr_mapping]
    events["betas_idx"] = events["stimulus"].apply(lambda x: event_stims_in_order.index(x))
else:
    events["betas_idx"] = events.index

events["embedding_idx"] = events["stimulus"].apply(lambda x: stimulus_id_map[x])
stimulus_ids = events["embedding_idx"].values


def get_file(file_path):
    with open(file_path, 'rb') as file:
        file_contents = pickle.load(file) 
    return file_contents

betas_lh = get_file(glm_data_dir / f'sub_{subj}/left_hem/sub-{subj}_hemi-lh_betas_averaged_.pkl')
betas_rh = get_file(glm_data_dir / f'sub_{subj}/right_hem/sub-{subj}_hemi-rh_betas_averaged_.pkl')

glm_single_R2_lh = get_file(glm_data_dir / f'sub_{subj}/left_hem/sub-{subj}_hemi-lh_glmsingle_results.pkl')
glm_single_R2_lh = np.squeeze(glm_single_R2_lh['typed']['R2'])
glm_single_R2_rh = get_file(glm_data_dir / f'sub_{subj}/right_hem/sub-{subj}_hemi-rh_glmsingle_results.pkl')
glm_single_R2_rh = np.squeeze(glm_single_R2_rh['typed']['R2'])

all_betas_lh = get_file(glm_data_dir / f'sub_{subj}/left_hem/sub-{subj}_hemi-lh_glmsingle_results.pkl')
all_betas_lh = np.squeeze(all_betas_lh['typed']['betasmd']).T
all_betas_rh = get_file(glm_data_dir / f'sub_{subj}/right_hem/sub-{subj}_hemi-rh_glmsingle_results.pkl')
all_betas_rh = np.squeeze(all_betas_rh['typed']['betasmd']).T


glm_single_R2 = np.concatenate((glm_single_R2_lh, glm_single_R2_rh), axis=0)
if averaged:
    betas = np.concatenate((betas_lh, betas_rh), axis=1)
else:
    betas = np.concatenate((all_betas_lh, all_betas_rh), axis=1)


accuracies_list = []
std_list = []

for ROI, ROI_mask in ROI_masks.items():
    print(f"\n- Decoding {ROI}...\n")
    accuracies[subj][ROI] = {}
    ROI_accuracies_list = []

    true_indices = np.where(ROI_mask)[0]
    masked_values = glm_single_R2[true_indices]
    top_256_indices_within_mask = np.argsort(masked_values)[-256:]  # Get indices of the top 256 values
    top_256_indices = true_indices[top_256_indices_within_mask]
    new_mask = np.zeros_like(ROI_mask, dtype=bool)
    new_mask[top_256_indices] = True
    
    # Cross validation. Use every run as test data once.
    for test_run_id in range(num_runs):
        training_run_ids = list(range(num_runs))
        training_run_ids.remove(test_run_id) # Remove the test data id 

        test_df = events[ events["run"] == str(test_run_id + 1) ]
        bold_test = np.array([betas[betas_idx] for betas_idx in test_df["betas_idx"].values])
        stimulus_ids_test = test_df["embedding_idx"].values

        train_df = events[events["run"].isin([str(run_num + 1) for run_num in training_run_ids])]
        bold_train = np.array([betas[betas_idx] for betas_idx in train_df["betas_idx"].values])
        stimulus_ids_train = train_df["embedding_idx"].values


        if value_selection == 'noise_ceiling':
            ncsnr = compute_ncsnr(bold_train, stimulus_ids_train) # Compute noise ceiling noise ratio
            nc = compute_nc(ncsnr, num_averages=1)

            nc[~ROI_mask] = 0
            nc_list = list(nc)  # Convert nc to a Python list
            indices = list(range(len(nc_list)))
            argsort_ids = sorted( indices, key=lambda i: -nc_list[i] ) # Sort accending

            if np.count_nonzero(nc) < top_n_nc_vals:
                argsort_ids = argsort_ids[:np.count_nonzero(nc)] 
            else:
                argsort_ids = argsort_ids[:top_n_nc_vals] 

            X_train = bold_train[:, argsort_ids] # Uses noise ceiling values to select top 256 values
            X_test = bold_test[:, argsort_ids] # Uses noise ceiling values to select top 256 values
        

        elif value_selection == 'R2':
            X_train = bold_train[:, new_mask] # Uses R2 values to select top 256 values
            X_test = bold_test[:, new_mask] # Uses R2 values to select top 256 values
        
        
        elif value_selection == 'no_selection':
            X_train = bold_train
            X_test = bold_test

        X_nan_train = np.isnan(X_train) # Checks if any not a number values in x and sets those to zero
        X_train[X_nan_train] = 0.

        X_nan_test = np.isnan(X_test) # Checks if any not a number values in x and sets those to zero
        X_test[X_nan_test] = 0.

        Y_train = stimulus[stimulus_ids_train] 
        Y_test = stimulus[stimulus_ids_test]

        model = FracRidgeRegressorCV()
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test) 

        y_test_array = torch.from_numpy(Y_test[None]).double()       
        y_pred_array = torch.from_numpy(Y_pred[:, None]).double()

        distances = cosine_distance(
            y_test_array,      
            y_pred_array
        ) 

        accuracy = round(two_versus_two(distances, stimulus_ids=stimulus_ids).item() * 100, 2) 
        ROI_accuracies_list.append(accuracy)

    # get average of accuracies for this iteration of runs
    ROI_accuracy_avg = np.mean(ROI_accuracies_list)
    accuracies[subj][ROI] = round(ROI_accuracy_avg, 3)
    print(f"   Mean Accuracy: {round(ROI_accuracy_avg, 3)}")

    # get standard deviation of accuracies for this iteration of runs
    ROI_accuracy_std = np.std(ROI_accuracies_list)
    print(f"   Std: {round(ROI_accuracy_std, 3)}\n")

    if averaged:
        with open(output_path / f'glm_avg_betas_{value_selection}/ROI_decoding_accs_{subj}.json', 'w') as json_file:
            json.dump(accuracies, json_file, indent=4)
    else:
        with open(output_path / f'glm_non_avg_betas_{value_selection}/ROI_decoding_accs_{subj}.json', 'w') as json_file:
            json.dump(accuracies, json_file, indent=4)

print(accuracies)