import os
import glob
from pathlib import Path
import shutil
import re
import shutil
import nilearn
import bids
import numpy as np
import pandas as pd
from nilearn import image, plotting
from matplotlib import pyplot as plt
import nilearn as ni
import bids
from bids import BIDSLayout
from scipy.stats import mannwhitneyu

from fmri_processing.data_loading import Data
from nilearn.glm.first_level import make_first_level_design_matrix

def enumerate_events(events):
    index = 0
    new_events = []
    for event in events:
        if event != "fixation_cross":
            event = f"a{index}_{event}"
            index += 1
        new_events.append(event)
    return new_events

subject = "01"
run = "1"
task = "bird"
#space = "T1w"
space = "MNI152NLin2009cAsym"
tr = 1.97
project = "TC2See_prdgm"

mask_type = "gm"
mask_type = "visual"

rsm_function = "cosine"
rsm_function = "pearson"

data = []
for i in range(8):
    data.append(Data(project, subject, task, str(i+1), space))

from fmri_processing.analysis import get_design_matrix, fit_glm, get_glm_activations, get_rsm, plot_rsm
#for r in [0.9, 0.8, 0.7, 0.6, 0.5]:
all_data = []
for r in [0.5]:
    for index, d in enumerate(data):

        design_matrix, names, vectors = get_design_matrix(frame_times=np.arange(d.fmri_img.shape[3])*tr, events=d.events, keep_order=False)
        output = Path(f"../results/derivatives_{project}/mask-{mask_type}_sub-{subject}_space-{space}_out{index}.npy")
        if 1:#not output.exists() and 1:
            fmri_glm = fit_glm(d.fmri_img, design_matrix, d.mask)

            #try:
            activations = get_glm_activations(d, fmri_glm, vectors, mask_type)
            #except ValueError:
            #    continue
            print("activations", activations.shape)
            all_data.append([names, activations])
            continue
            #np.save(str(output), activations)
        #activations = np.load(str(output))

        events = d.events.query("trial_type != 'fixation_cross'")
        order = []
        for n in names:
            order.append(np.where(events.trial_type == n)[0][0])

        if 0:
            plt.figure(1)
            rsm = get_rsm(activations[order, :], rsm_function)

            plot_rsm(rsm, np.array(names)[order])

            plt.savefig(f"../results/derivatives_{project}2/ordered_mask-{mask_type}_sub-{subject}_space-{space}_rsmcom-{rsm_function}_rsm{index}.png")

        plt.figure(2)
        rsm = get_rsm(activations[:, :], rsm_function)

        plot_rsm(rsm, np.array(names)[:])

        plt.savefig(f"../results/derivatives_{project}2/ordered_mask-{mask_type}_sub-{subject}_space-{space}_rsmcom-{rsm_function}_rsm{index}.png")

main_names = all_data[2][0]
joined_act = []
for i in range(4):
    order = []
    for ii, n in enumerate(all_data[i][0]):
        if n in main_names:
            order.append(ii)
    joined_act.append(all_data[i][1][np.array(order)])

joined_act = np.concatenate(joined_act)
rsm = get_rsm(joined_act[:, :], rsm_function)

plot_rsm(rsm, np.array(main_names)[:])