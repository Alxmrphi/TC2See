from pathlib import Path
import json
import numpy as np
import torch
from tqdm import tqdm
# from scipy.ndimage import zoom, binary_dilation
import h5py
# import nibabel as nib
from fracridge import FracRidgeRegressorCV

from tc2see import load_data
from metrics import (
    cosine_distance, two_versus_two,
)

from noise_ceiling import (
    compute_ncsnr,
    compute_nc,
)

tc2see_version = 3

accuracies = {}

subjs = ['17']
for subj in tqdm(subjs):
    tr = 2 # 1.97
    subject_no = subj 
    subject = f'sub-{subject_no}'

    bold, stimulus_ids, mask, affine = load_data(
        f'../data/processed/hdf5s/tc2see-v{tc2see_version}-bold.hdf5', 
        subject,
        tr_offset=6 / tr,
        run_normalize='linear_trend',
        interpolation=False,
    )


    model_name = 'ViT-B=32'
    embedding_name = 'embedding' 

    # load the clip embeddings
    with h5py.File(f'../data/{model_name}-features.hdf5', 'r') as f:
        stimulus = f[embedding_name][:]
    Y = stimulus[stimulus_ids] # get the stimulus representations to decode


    subject = f'sub-{subject_no}'
    # 6 Runs - 1 run as the test each time (a run is each time the person gets into the scanner and looks into the scanner for a certain amount of time ~ approx 6 mins)
    results = dict
    permutation_test = False
    iterations = 1
    num_runs = 6

    all_itters_avg = 0
    all_itters_var = 0
    all_itters_std = 0
    all_itters_max = 0
    all_itters_min = 0

    for iteration in tqdm(range(iterations)):
        itter_accuracy = 0
        itter_variance = 0
        
        # Cross validation. Use every id as test data once.
        for test_run_id in tqdm(range(num_runs)):
            training_run_ids = list(range(num_runs))
            training_run_ids.remove(test_run_id) # Remove the test data id 

            load_data_params = dict(
                path = f'../data/processed/hdf5s/tc2see-v{tc2see_version}-bold.hdf5', 
                subject = subject,
                tr_offset = num_runs / tr,
                run_normalize='linear_trend',
                interpolation=False,
            )

            bold_train, stimulus_ids_train, mask, affine = load_data(
                **load_data_params,
                run_ids = training_run_ids
            )

            mask = mask[mask] # flatten mask

            bold_test, stimulus_ids_test, _, _ = load_data(
                **load_data_params,
                run_ids = [test_run_id]
            )
            
            ncsnr = compute_ncsnr(bold_train, stimulus_ids_train) # Compute noise ceiling noise ratio
            nc = compute_nc(ncsnr, num_averages=1)

            nc_vc = nc.copy() 
            nc_vc[~mask] = 0 # Set values not in mask to zero 
            argsort_ids = np.argsort(-nc_vc) # Default ascending, make descending 
            argsort_ids = argsort_ids[:256] 

            X_train = bold_train[:, argsort_ids]    
            X_nan_train = np.isnan(X_train) # Checks if any not a number values in x and sets those to zero
            X_train[X_nan_train] = 0.

            X_test = bold_test[:, argsort_ids]
            X_nan_test = np.isnan(X_test) # Checks if any not a number values in x and sets those to zero
            X_test[X_nan_test] = 0.

            with h5py.File(f'../data/{model_name}-features.hdf5', 'r') as f:
                stimulus = f[embedding_name][:]
            Y_train = stimulus[stimulus_ids_train] 
            Y_test = stimulus[stimulus_ids_test]

            if permutation_test:
                ids = np.arange(Y_train.shape[0])
                np.random.shuffle(ids)
                Y_train = Y_train[ids]

            model = FracRidgeRegressorCV()
            model.fit(X_train, Y_train)
            Y_test_pred = model.predict(X_test) # Y_test and Y_test_pred are n x 512 matrics (n is the number of birds).

            distances = cosine_distance(
                torch.from_numpy(Y_test[None]).float(), 
                torch.from_numpy(Y_test_pred[:, None]).float()
            ) # Y_test(1, N, 512) & Y_test_pred(N, 1, 512) converted to pytorch arrays from np

            # Chance is 50% (above 50% is good, below not great, if really close ex. 54% or 52%, prove statistically above chance)
            accuracy = round(two_versus_two(distances, stimulus_ids=stimulus_ids).item() * 100, 2) 
            itter_accuracy += accuracy
            
            if accuracy < all_itters_min:
                min = accuracy
            if accuracy > all_itters_max:
                max = accuracy
        
        all_itters_avg += itter_accuracy

        print(f"Iteration {iteration} avg accuracy: ", itter_accuracy/num_runs)

    accuracies[subj] = all_itters_avg/(num_runs*iterations)
    total_accuracy = all_itters_avg/(num_runs*iterations)
    print("Total Accuracy: ", total_accuracy)

print(accuracies)

with open('ROI_accuracies_vols.json', 'w') as json_file:
    json.dump(accuracies, json_file, indent=4)