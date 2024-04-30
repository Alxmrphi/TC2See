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

from glmsingle.glmsingle import getcanonicalhrflibrary
from glm_single_3_include import *

subject = "03"
run = "1"
task = "bird"
space = "T1w"
#space = "MNI152NLin2009cAsym"
tr = 2
project = "TC2See_prdgm"

mask_type = "gm"
mask_type = "visual"

rsm_function = "cosine"
rsm_function = "pearson"

data = []
for i in range(6):
    data.append(Data(project, subject, task, str(i+1), space))

from fmri_processing.analysis import fit_glm, get_glm_activations, get_rsm, plot_rsm
for i, d in enumerate(data):
    d.duration = 0.5
    if i == 2 or i == 5:
        d.duration = 5.5

##
data_list = [data[0], data[1], data[2], data[3], data[4], data[5]]

mean_mask = average_masks(data_list)

img = join_fmri_img(data_list, mean_mask)



##
for col in d.confounds.columns:
    print(col)
confounds = []#"rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
confounds = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]

r2s = []
betas = []
dms = []
for i in range(20):
    print("HRF i", i)
    design_matrix = get_joined_design_matrix(data_list, i, confounds, False, 0)
    dms.append(design_matrix)

    #nilearn.masking.unmask
    beta, r2 = get_beta_r2(img, design_matrix)
    betas.append(beta)
    r2s.append(r2)
betas = np.array(betas)
r2s = np.array(r2s)
max_index = np.argmax(r2s, axis=0)

##
betas = []
for i in range(20):
    print("HRF i", i)
    design_matrix = get_joined_design_matrix(data_list, i, confounds, False, 0, True)
    beta, r2 = get_beta_r2(img, design_matrix)
    betas.append(beta)
betas = np.array(betas)
beta = np.array([betas[v, :, i] for i, v in enumerate(max_index)]).T

##
from fmri_processing.analysis import get_rsm, plot_rsm
names = get_names(design_matrix)
order = np.argsort(names)
rsm_function = "cosine"
rsm_function = "pearson"

snr_flat = signal_noise_ratio(design_matrix, beta)
print("snr", snr_flat.max(), snr_flat.min())
best_100_i = np.argsort(snr_flat)[::-1][:10]

rsm = get_rsm(beta[order[:100]][:, best_100_i], rsm_function)
plot_rsm(rsm, np.array(names)[order[:100]])

##
names = get_names(design_matrix)
animate = {
    'aardvark': True,
    'air_pump': False,
    'alpaca': True,
    'anchor': False,
    'antelope': True,
    'anvil': False,
    'badger': True,
    'bag': False,
    'basketball': False,
    'beanbag': False,
    'bear': True,
    'biscuit': False,
    'bison': True,
    'box': False,
    'brush': False,
    'bull': True,
    'camel': True,
    'cat': True,
    'cheesecake': False,
    'chick': True,
    'cow': True,
    'dagger': False,
    'dalmatian': True,
    'dartboard': False,
    'deer': True,
    'doorknob': False,
    'drum': False,
    'duck': True,
    'elephant': True,
    'emerald': False,
    'faucet': False,
    'ferret': True,
    'fish': True,
    'flan': False,
    'gavel': False,
    'gorilla': True,
}

##

include = []
category = []
for ii, name in enumerate(names):
    if name not in animate.keys():
        if name not in ["rot", "drift", "trans", "constant"]:
            print("IGNORE", name)
        continue
    include.append(ii)
    category.append(animate[name])

import tensorflow as tf
#snr_flat = signal_noise_ratio(design_matrix, beta)
snr_flat = signal_noise_ratio_names(names, beta)
best_100_i = np.argsort(snr_flat)[::-1][:200]

if 0:
    x = beta[:, best_100_i][include]
    y = tf.keras.utils.to_categorical(category)
else:
    x = beta[include]
    y = tf.keras.utils.to_categorical(category)

frac = int(np.floor(0.5*x.shape[0]))
i = np.random.permutation(np.arange(x.shape[0]))
x = x[i]
y = y[i]
x_train, x_test = x[:frac], x[frac:]
y_train, y_test = y[:frac], y[frac:]

print(x_train.shape, x_test.shape)

##
model = tf.keras.models.Sequential([
    #tf.keras.layers.Dense(1000, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation="softmax"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="CategoricalCrossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

##
if 0:
    from sklearn.linear_model import Ridge
    clf = Ridge(alpha=1.0)
    clf.fit(x_train, y_train)
    clf.score(x_test, y_test)
    clf.predict(x_test)

##

if 0:
    plt.clf()
    i = 0
    design_matrix1 = get_joined_design_matrix(data_list, i, confounds, False, 0, True)
    beta1, r2 = get_beta_r2(img, design_matrix1)
    plt.subplot(221)
    ncsnr1 = signal_noise_ratio(design_matrix1, beta1, mean_mask)
    print(np.mean(ncsnr1))
    plt.imshow(ncsnr1[:, :, 40], vmin=0, vmax=20)

    #beta2, r2 = get_beta_r2(img, design_matrix2)
    plt.subplot(222)
    ncsnr2 = signal_noise_ratio(design_matrix1, beta1, mean_mask, randomize=True)
    print(np.mean(ncsnr2))
    plt.imshow(ncsnr2[:, :, 40], vmin=0, vmax=20)

design_matrix = get_joined_design_matrix(data_list, i, confounds, False, 0, True)

plt.subplot(221)
ncsnr = signal_noise_ratio(design_matrix, beta, mean_mask)
plt.imshow(ncsnr[:, :, 40], vmin=0, vmax=20)
plt.subplot(222)
ncsnr = signal_noise_ratio(design_matrix, betas[0], mean_mask)
plt.imshow(ncsnr[:, :, 40], vmin=0, vmax=20)
plt.subplot(223)
ncsnr = signal_noise_ratio(design_matrix, betas[10], mean_mask)
plt.imshow(ncsnr[:, :, 40], vmin=0, vmax=20)
plt.subplot(224)
ncsnr = signal_noise_ratio(design_matrix, betas[19], mean_mask)
plt.imshow(ncsnr[:, :, 30], vmin=0, vmax=20)
plt.savefig(f"signal-to-noise-sub-{subject}.png")


if 0:
    ##
    offset = 0
    all_names = []
    all_indices = []
    for d in data:
        length = d.fmri_img.shape[-1]
        events = d.events[d.events.onset < (length - 6) * d.tr]
        events = events[events.stimulus != "+"]
        events = events.copy()

        def get_stim(name, index):
            m = re.match(r"images/(.*)_\d*\D(?:_hash_\d)?.jpg", name)
            return m.groups()[0]

        events["stim"] = [get_stim(n, i) for i, n in enumerate(events.stimulus)]
        events["trial_type"] = events["stim"]
        names = events["stim"]

        indices = np.array((events.onset / d.tr)).astype(np.int)
        all_indices.extend(indices + offset)
        all_names.extend(names)
        offset += length

    beta = img[np.array(all_indices)]


    ##
    design_matrix = get_joined_design_matrix_grouped(data_list, i)#, confounds, False, 0, False)
    beta, r2 = get_beta_r2(img, design_matrix)
    names = get_names(design_matrix)

    ##
    ni.plotting.plot_design_matrix(design_matrix)