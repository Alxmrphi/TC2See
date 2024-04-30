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

subject = "04"
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
#for i in range(4, 4+4):
for i in range(6):
    data.append(Data(project, subject, task, str(i+1), space))

from fmri_processing.analysis import fit_glm, get_glm_activations, get_rsm, plot_rsm
for i, d in enumerate(data):
    d.duration = 0.5
    if i == 2 or i == 5:
        d.duration = 5.5

if 0:
    d = data_list[0]
    length = d.fmri_img.shape[-1]
    events = d.events[d.events.onset < (length-6)*d.tr]
    events = events[events.stimulus != "+"]
    events = events.copy()
    def get_stim(name):
        m = re.match(r"images/(.*)_\d*\D(?:_hash_\d)?.jpg", name)
        return m.groups()[0]
    events["stimulus"] = [get_stim(n) for n in events.stimulus]

    indices = np.array((events.onset / d.tr)).astype(np.int)

    m = d.mask.get_fdata()
    im = ni.masking.apply_mask(d.fmri_img, d.mask)
    im = detrend(im)
    if 0:
        im2 = x*b+a
        plt.plot(im[:, 1000])
        plt.plot(im2[:, 1000])

        plt.plot(img[:, 1000])

    gamma = im[indices + 2]
    names = events.stimulus
    order = np.argsort(names)
    rsm_function = "pearson"
    rsm = get_rsm(gamma, rsm_function)
    plot_rsm(rsm, np.array(names))

    rsm = get_rsm(gamma[order], rsm_function)
    plot_rsm(rsm, np.array(names)[order])

    gammas = []
    var_activ = []
    for name in np.unique(names):
        activations = gamma[names == name]
        var_activ.append(np.var(activations, axis=0))
        gammas.append(activations)
    var_activ = np.array(var_activ)
    mean_var_active = np.mean(var_activ, axis=0)
    for g in gammas:
        print(g[:, 56413])

    def diagonal_ordering(m2):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        i = reverse_cuthill_mckee(csr_matrix(m2))
        m2 = m2[i][:, i]
        return m2, i


    rsm2, index2 = diagonal_ordering(rsm)
    plot_rsm(rsm2, np.array(names)[index2])

if 0:
    d = data_list[0]
    d = data_list[0]
    length = d.fmri_img.shape[-1]
    events = d.events[d.events.onset < (length - 6) * d.tr]
    events = events[events.stimulus != "+"]
    events = events.copy()

    def get_stim(name, index):
        m = re.match(r"images/(.*)_\d*\D(?:_hash_\d)?.jpg", name)
        return f"_{index:03d}_"+m.groups()[0]

    def get_stim2(name, index):
        m = re.match(r"images/(.*)_\d*\D(?:_hash_\d)?.jpg", name)
        return m.groups()[0]

    events["stim"] = [get_stim(n,i) for i, n in enumerate(events.stimulus)]
    events["stim2"] = [get_stim2(n,i) for i, n in enumerate(events.stimulus)]
    events["trial_type"] = events["stim"]
    names = events["stim2"]

    m = d.mask.get_fdata()
    im = ni.masking.apply_mask(d.fmri_img, d.mask)
    im = detrend(im)
    if 0:
        im2 = x * b + a
        plt.plot(im[:, 1000])
        plt.plot(im2[:, 1000])

        plt.plot(img[:, 1000])

    confounds = []  # "rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
    confounds = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
    design_matrix = make_first_level_design_matrix(np.arange(d.fmri_img.shape[3]) * d.tr, events[["onset", "duration", "trial_type"]],
                                                   add_regs=d.confounds[confounds],)
    ni.plotting.plot_design_matrix(design_matrix)
    print(design_matrix.columns)

    beta, r2 = get_beta_r2(im, design_matrix)

    order = np.argsort(names)
    rsm_function = "pearson"
    rsm = get_rsm(beta, rsm_function)
    plot_rsm(rsm, np.array(names))

    rsm = get_rsm(beta[order], rsm_function)
    plot_rsm(rsm, np.array(names)[order])
if 0:
    data_list = [data[0], data[1], data[2], data[3], data[4], data[5]]

    mean_mask = average_masks(data_list)

    img = join_fmri_img(data_list, mean_mask)


    for col in d.confounds.columns:
        print(col)
    confounds = []#"rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]

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

    betas = []
    for i in range(20):
        print("HRF i", i)
        design_matrix = get_joined_design_matrix(data_list, i, confounds, False, 0, True)
        beta, r2 = get_beta_r2(img, design_matrix)
        betas.append(beta)
    betas = np.array(betas)
    beta = np.array([betas[v, :, i] for i, v in enumerate(max_index)]).T

    from fmri_processing.analysis import get_rsm, plot_rsm
    names = get_names(design_matrix)
    order = np.argsort(names)
    rsm_function = "cosine"
    rsm_function = "pearson"
    rsm = get_rsm(beta[order[:100]][:, best_100_i], rsm_function)
    plot_rsm(rsm, np.array(names)[order[:100]])

    snr_flat = signal_noise_ratio(design_matrix, beta)
    best_100_i = np.argsort(snr_flat)[::-1][:100]

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

    plt.subplot(221)
    ncsnr = signal_noise_ratio(design_matrix, beta_rand, mean_mask, randomize=True)
    plt.imshow(ncsnr[:, :, 40], vmin=0, vmax=20)
    plt.subplot(222)
    ncsnr = signal_noise_ratio(design_matrix, betas[0], mean_mask, randomize=True)
    plt.imshow(ncsnr[:, :, 40], vmin=0, vmax=20)
    plt.subplot(223)
    ncsnr = signal_noise_ratio(design_matrix, betas[10], mean_mask, randomize=True)
    plt.imshow(ncsnr[:, :, 40], vmin=0, vmax=20)
    plt.subplot(224)
    ncsnr = signal_noise_ratio(design_matrix, betas[19], mean_mask, randomize=True)
    plt.imshow(ncsnr[:, :, 30], vmin=0, vmax=20)
    plt.savefig(f"signal-to-noise-sub-{subject}_random.png")

    ncsnr = noise_ceilling(design_matrix, beta, mean_mask)
    plt.subplot(221)
    plt.imshow(ncsnr[:, :, 40], vmin=0, vmax=10)
    plt.subplot(222)
    ncsnr = noise_ceilling(design_matrix, betas[0], mean_mask)
    print(np.nansum(ncsnr))
    plt.imshow(ncsnr[:, :, 40], vmin=0, vmax=100)
    plt.subplot(223)
    ncsnr = noise_ceilling(design_matrix, betas[10], mean_mask)
    plt.imshow(ncsnr[:, :, 40], vmin=0, vmax=100)
    plt.subplot(224)
    ncsnr = noise_ceilling(design_matrix, betas[19], mean_mask)
    plt.imshow(ncsnr[:, :, 30], vmin=0, vmax=100)
    plt.savefig(f"noise_ceilling-sub-{subject}.png")

    compute_ncsnr(beta, [np.array(a) for a in id_order])

    img_ = nilearn.masking.unmask(img, mean_mask).get_fdata()
    y = 23; x = 10; z = 40
    plt.plot(img_[y, x, z])
    for i in range(20):
        print("HRF i", i)
        design_matrix = get_joined_design_matrix(data_list, i, confounds, False, 0)
        d2 = get_beta_r2_plot(img, design_matrix)
        d2 = nilearn.masking.unmask(d2, mean_mask).get_fdata()
        plt.plot(d2[y, x, z])
        plt.draw()

    r2s = [nilearn.masking.unmask(r2, mean_mask).get_fdata() for r2 in r2s]
    r2s = np.array(r2s)
    max_index = np.argmax(r2s, axis=0)
    plt.imshow(np.max(r2s, axis=0)[:, :, 40], cmap="viridis")
    plt.imshow(max_index[:, :, 40], vmin=0, vmax=19, cmap="viridis")
    print(np.percentile(r2s, 80))
    plt.savefig(f"sub-{subject}_run-3_offset5s.png")

    if 0:
        r2s = []
        for i in range(20):
            print("HRF i", i)
            dm = []
            for d in data_list:
                design_matrix, names, vectors = get_design_matrix(frame_times=np.arange(d.fmri_img.shape[3]) * tr,
                                                                  events=d.events, keep_order=False,
                                                                  glm_single_index=i, tr=1.97, duration=d.duration)
                dm.append(design_matrix)

            dm2 = pd.concat(dm).fillna(0)

            fmri_glm = fit_glm(img, dm2, mean_mask)
            r2s.append(fmri_glm.r_square[0].get_fdata())
            continue



            for vec in vectors:
                activations_1 = fmri_glm.compute_contrast(vec)
        r2s = np.array(r2s)
        max_index = np.argmax(r2s, axis=0)
