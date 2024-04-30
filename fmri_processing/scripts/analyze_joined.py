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

subject = "02"
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

print("load data")
data = []
for i in range(8):
    data.append(Data(project, subject, task, str(i+1), space))
#raise
from fmri_processing.analysis import get_design_matrix, fit_glm, get_glm_activations, get_rsm, plot_rsm
from nilearn.plotting import plot_design_matrix

print("calc design matrix")
design_matrices = []
for index, d in enumerate(data):
    design_matrix, names, vectors = get_design_matrix(frame_times=np.arange(d.fmri_img.shape[3])*tr, events=d.events)
    for col in design_matrix.columns:
        if col.startswith("drift") or col == "constant":
            design_matrix[col+"_"+str(index)] = design_matrix[col]
            design_matrix = design_matrix.drop(columns=col)
    design_matrices.append(design_matrix)

d2 = pd.concat(design_matrices)
d2 = d2.fillna(0)
#plot_design_matrix(d2)

print("calc fmri_data_joined")
fmri_data_joined = [d.fmri_img.get_fdata() for d in data]
fdata_joned = np.concatenate(fmri_data_joined, axis=-1)
fmri_img_joined = ni.image.new_img_like(data[0].fmri_img, fdata_joned)

print("calc gm_mask_joined")
gm_mask_joined = ni.image.math_img("a+b+c+d+e+f >= 0.5*6",
                            a=ni.image.resampling.resample_to_img(data[0].gm_probseg, data[0].fmri_img),
                            b=ni.image.resampling.resample_to_img(data[1].gm_probseg, data[1].fmri_img),
                            c=ni.image.resampling.resample_to_img(data[2].gm_probseg, data[2].fmri_img),
                            d=ni.image.resampling.resample_to_img(data[3].gm_probseg, data[3].fmri_img),
                            e=ni.image.resampling.resample_to_img(data[4].gm_probseg, data[4].fmri_img),
                            f=ni.image.resampling.resample_to_img(data[5].gm_probseg, data[5].fmri_img),
                            )

print("FirstLevelModel")
from nilearn.glm.first_level import FirstLevelModel
fmri_glm = FirstLevelModel()
fmri_glm2 = fmri_glm.fit(fmri_img_joined, design_matrices=d2)

print("calc vectors")
vectors = []
names = []
for i, name in enumerate(d2.columns):
    print(name, name == "fixation_cross", name.startswith("drift"), name.startswith("constant"))
    if name == "fixation_cross" or name.startswith("drift") or name.startswith("constant"):
        continue
    zeros = np.zeros(d2.shape[1])
    zeros[i] = 1
    vectors.append(zeros)
    names.append(name)

mask_type = "gm"
visual = None
def get_glm_activations(d, fmri_glm, vectors, mask_type="gm", r=None):
    global visual

    activations = []
    for vec in vectors:
        activations_1 = fmri_glm.compute_contrast(vec)

        if mask_type == "visual":
            if visual is None:
                dataset_ju = ni.datasets.fetch_atlas_juelich('maxprob-thr0-1mm')
                visual = ni.image.math_img("(a >= 48)*(a < 48+5)", a=dataset_ju.maps)
            gm_mask = ni.image.resample_to_img(visual, d.fmri_img, interpolation="nearest")
            #gm_mask = ni.image.math_img("a * b", a=gm_mask, b=r2_mask)
            act = ni.masking.apply_mask(activations_1, gm_mask)
        if mask_type == "non-visual":
            if visual is None:
                dataset_ju = ni.datasets.fetch_atlas_juelich('maxprob-thr0-1mm')
                visual = ni.image.math_img("((a >= 48)*(a < 48+5)) < 0.5", a=dataset_ju.maps)
            gm_mask = ni.image.resample_to_img(visual, d.fmri_img, interpolation="nearest")
            #gm_mask = ni.image.math_img("a * b", a=gm_mask, b=r2_mask)
            act = ni.masking.apply_mask(activations_1, gm_mask)
        elif mask_type == "gm":
            #gm_mask = ni.image.math_img("a >= 0.5", a=ni.image.resampling.resample_to_img(d.gm_probseg, d.fmri_img))
            act = ni.masking.apply_mask(activations_1, gm_mask_joined)
        else:
            act = activations_1.get_fdata().reshape(-1)
        activations.append(act)

    activations = np.array(activations)
    return activations
acts = get_glm_activations(d, fmri_glm, vectors, mask_type=mask_type)
rsm = get_rsm(acts, rsm_function)

plot_rsm(rsm, names)
Path(f"../results/derivatives_{project}3").mkdir(exist_ok=True)
plt.savefig(f"../results/derivatives_{project}3/all_mask-{mask_type}_sub-{subject}_space-{space}_rsmcom-{rsm_function}.png")


x = np.arange(0, 1, 0.001)
y = np.tanh(x)
plt.subplot(121)
plt.plot(x, y)
plt.plot(x, x)
plt.subplot(122)
plt.plot(x, y/x)
plt.ylim(0, 1)