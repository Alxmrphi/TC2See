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

if 0:
    plt.figure(3)
    plt.clf()
    hrfs = getcanonicalhrflibrary(32, 1.97/50)
    hrfs = getcanonicalhrflibrary(0.5, 1.97/50)
    cmap = plt.get_cmap("viridis", 20)
    for i, h in enumerate(hrfs):
        plt.plot(np.arange(0, hrfs.shape[1])*1.97/50, h, 'o-', color=cmap(i))
    plt.show()

def hrf(tf, oversampling):
    print(tf, oversampling)
    return hrfs[0]

if 0:
    frame_times = np.arange(0, 100, 1.97)
    onsets = [1, 8, 25, 47, 78]
    events = []
    for i, o in enumerate(onsets):
        events.append(dict(onset=o, duration=2, trial_type="a"+str(i+10)))
    events = pd.DataFrame(events)
    plt.figure(0)
    design_matrix = make_first_level_design_matrix(frame_times, events[["onset", "duration", "trial_type"]]#, hrf_model=hrf
                                                   )

    ni.plotting.plot_design_matrix(design_matrix)

    plt.figure(1)
    design_matrix = make_first_level_design_matrix(frame_times, events[["onset", "duration", "trial_type"]], hrf_model=hrf
                                                   )

    ni.plotting.plot_design_matrix(design_matrix)
    plt.show()

def get_design_matrix(frame_times, events, keep_order=False, glm_single_index=None, tr=None, duration=2, shuffle=False, time_offset=0, add_regs=None):
    from nilearn.glm.first_level import make_first_level_design_matrix

    if glm_single_index is not None:
        hrfs = getcanonicalhrflibrary(duration, tr / 50)
        #for h in hrfs:
        #    plt.plot(np.arange(0, hrfs.shape[1]), h, 'o-')
        #plt.show()

        def hrf(tf, oversampling):
            #print(tf, oversampling)
            return hrfs[glm_single_index]
    else:
        hrf = "spm"

    index = 0
    def category(path):
        nonlocal index
        if path == "+":
            return "fixation_cross"
        if keep_order is True:
            index += 1
            return f"_{index:03d}" + re.match(r"\d*\.(.*)", Path(path).stem).groups()[0]  # .replace(".", "_").replace("_", "")[4:]
        return re.match(r"\d*\.(.*)", Path(path).stem).groups()[0]  # .replace(".", "_").replace("_", "")[4:]

    events = events[events.stimulus != "+"].copy()
    if shuffle is not False:
        np.random.seed(shuffle)
        x = np.array(events["onset"])
        np.random.shuffle(x)
        events["onset"] = x
    events["onset"] += time_offset
    events["trial_type"] = [category(s) for s in events.stimulus]
    design_matrix = make_first_level_design_matrix(frame_times, events[["onset", "duration", "trial_type"]], hrf_model=hrf, add_regs=add_regs
                                                   )
    #plotting.plot_design_matrix(design_matrix)
    # plotting.show()

    vectors = []
    names = []
    for i, name in enumerate(design_matrix.columns):
        if name in ["fixation_cross", "constant"] or name.startswith("drift"):
            continue
        zeros = np.zeros(design_matrix.shape[1])
        zeros[i] = 1
        vectors.append(zeros)
        names.append(name)

    return design_matrix, names, vectors


def enumerate_events(events):
    index = 0
    new_events = []
    for event in events:
        if event != "fixation_cross":
            event = f"a{index}_{event}"
            index += 1
        new_events.append(event)
    return new_events

non_zero_indices = None
non_zero_shape = None
def matrix_to_indices(f):
    return np.where(f[:, :, :, 0] != 0)

def matrix_to_non_zero_list(f):
    global non_zero_indices, non_zero_shape
    if non_zero_indices is None:
        non_zero_indices = matrix_to_indices(f)
        non_zero_shape = f.shape
    return f[non_zero_indices[0], non_zero_indices[1], non_zero_indices[2]]

def non_zero_list_to_matrix(f):
    data = np.zeros(non_zero_shape, f.dtype)
    data[non_zero_indices[0], non_zero_indices[1], non_zero_indices[2]] = f
    return data

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
d = data[3]
for i, d in enumerate(data):
    d.duration = 1
    if i == 3:
        d.duration = 0.5

if 0:
    plt.figure(2)
    design_matrix, names, vectors = get_design_matrix(frame_times=np.arange(data[3].fmri_img.shape[3])*tr, events=d.events, keep_order=False,
                                                             glm_single_index=0, tr=1.97, duration=d.duration)
    ni.plotting.plot_design_matrix(design_matrix)
raise

def average_masks(data_list):
    masks = []
    for d in data_list:
        masks.append(d.mask.get_fdata())
    mean_mask = np.round(np.mean(masks, axis=0))
    mean_mask = ni.image.new_img_like(d.mask, mean_mask)
    return mean_mask

def join_fmri_img(data_list, mean_mask):

    img = []
    for d in data_list:
        im = ni.masking.apply_mask(d.fmri_img, mean_mask)
        im, _ = ni.glm.first_level.mean_scaling(im)
        img.append(im)
    img = np.vstack(img)
    #img, _  = ni.glm.first_level.mean_scaling(img)
    return img
    img = nilearn.masking.unmask(img, mean_mask)

def get_beta_r2(img, dm2):
    dm2 = np.asarray(dm2)
    img = np.asarray(img)
    beta = np.linalg.pinv(dm2) @ img
    d2 = dm2 @ beta
    ss_err = np.mean((d2 - img) ** 2, axis=0)
    ss_tot = np.var(img, axis=0)
    r2 = 1 - ss_err / ss_tot
    return beta, r2

def get_joined_design_matrix(data_list, i, confounds=None, shuffle=False, time_offset=0, keep_order=False):
    dm = []
    for ii, d in enumerate(data_list):
        design_matrix, names, vectors = get_design_matrix(frame_times=np.arange(d.fmri_img.shape[3]) * d.tr,
                                                          events=d.events, keep_order=keep_order,
                                                          glm_single_index=i, tr=1.97, duration=d.duration,
                                                          add_regs=d.confounds[confounds],
                                                          shuffle=shuffle, time_offset=time_offset)
        renames = {}
        for col in design_matrix.columns:
            if keep_order or (col.startswith("drift") or col.startswith("constant") or col in confounds):
                renames[col] = col+"_"+str(ii)
        design_matrix = design_matrix.rename(columns=renames)
        dm.append(design_matrix)

    dm2 = pd.concat(dm).fillna(0)
    return dm2



def get_beta_r2_plot(img, dm2):
    dm2 = np.asarray(dm2)
    img = np.asarray(img)
    beta = np.linalg.pinv(dm2) @ img
    d2 = dm2 @ beta
    return d2

from typing import Sequence, Union
def compute_ncsnr(betas: np.ndarray, stimulus_ids: Union[np.ndarray, Sequence[np.ndarray]]):
    """
    :param betas: Array of betas with shape (num_betas, num_voxels)
    :param stimulus_ids: Array with shape (num_images, num_repetitions)
        Optionally provide a sequence of arrays if some images have a different number of repetitions
    :return:
    """

    if isinstance(stimulus_ids, np.ndarray):
        stimulus_ids = [stimulus_ids]

    betas_var = []
    for ids in stimulus_ids:
        num_images, num_repetitions = ids.shape

        stimulus_betas = betas[ids]  # shape=(num_images, num_repetitions, num_voxels)

        # stimulus_betas.var(axis=1, ddof=1) should work but it doesn't
        # doing it manually instead...
        betas_var.append(((stimulus_betas.mean(axis=1, keepdims=True) - stimulus_betas) ** 2).sum(axis=1) / (num_repetitions - 1))
    betas_var_mean = np.nanmean(np.concatenate(betas_var), axis=0)

    std_noise = np.sqrt(betas_var_mean)

    std_signal = 1. - betas_var_mean
    std_signal[std_signal < 0.] = 0.
    std_signal = np.sqrt(std_signal)
    ncsnr = std_signal / std_noise

    return ncsnr

def signal_noise_ratio(design_matrix, beta, mean_mask):
    ids = {}
    next_id = 0
    id_order = []
    for index, col in enumerate(design_matrix.columns):
        m = re.match("_\d\d\d(.*)_hrf_\d+", col)
        if m:
            name = m.groups()[0]
            if name not in ids:
                ids[name] = next_id
                next_id += 1
                id_order.append([])
            id_order[ids[name]].append(index)
    var = np.zeros(beta.shape[1])
    for a in id_order:
        num_repetitions = len(a)
        stimulus_betas = beta[a]
        var += ((stimulus_betas.mean(axis=0, keepdims=True) - stimulus_betas) ** 2).sum(axis=0) / (num_repetitions - 1)
    var /= len(id_order)
    std_noise = np.sqrt(var)

    std_signal = 1. - var
    std_signal[std_signal < 0.] = 0.
    std_signal = np.sqrt(std_signal)
    ncsnr = std_signal / std_noise
    ncsnr = nilearn.masking.unmask(ncsnr, mean_mask).get_fdata()
    return compute_nc(ncsnr, num_averages=1)

def compute_nc(ncsnr: np.ndarray, num_averages: int):
    ncsnr_squared = ncsnr ** 2
    nc = 100. * ncsnr_squared / (ncsnr_squared + (1. / num_averages))
    return nc

if 0:
    data_list = [data[0], data[1], data[2], data[3]]

    mean_mask = average_masks(data_list)

    img = join_fmri_img(data_list, mean_mask)

    for col in d.confounds.columns:
        print(col)
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

    betas = []
    for i in range(20):
        print("HRF i", i)
        design_matrix = get_joined_design_matrix(data_list, i, confounds, True, 0, True)
        beta, r2 = get_beta_r2(img, design_matrix)
        betas.append(beta)
    betas = np.array(betas)
    beta = np.array([betas[v, :, i] for i, v in enumerate(max_index)]).T

    ncsnr = signal_noise_ratio(design_matrix, beta, mean_mask)
    plt.subplot(221)
    plt.imshow(ncsnr[:, :, 40], vmin=0, vmax=100)
    plt.subplot(222)
    ncsnr = signal_noise_ratio(design_matrix, betas[0], mean_mask)
    plt.imshow(ncsnr[:, :, 40], vmin=0, vmax=100)
    plt.subplot(223)
    ncsnr = signal_noise_ratio(design_matrix, betas[10], mean_mask)
    plt.imshow(ncsnr[:, :, 40], vmin=0, vmax=100)
    plt.subplot(224)
    ncsnr = signal_noise_ratio(design_matrix, betas[19], mean_mask)
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
runs = []
for d in data:
    r2s = []
    ss_tot = None
    ss_res_list = []
    for i in range(20):
        print("HRF i", i)
        design_matrix, names, vectors = get_design_matrix(frame_times=np.arange(d.fmri_img.shape[3])*tr, events=d.events, keep_order=False,
                                                          glm_single_index=i, tr=1.97, duration=d.duration)

        fmri_glm = fit_glm(d.fmri_img, design_matrix, d.mask)

        if ss_tot is None:
            ss_tot = matrix_to_non_zero_list(fmri_glm.predicted[0].get_fdata() + fmri_glm.residuals[0].get_fdata())
        ss_res = matrix_to_non_zero_list((fmri_glm.residuals[0].get_fdata()) ** 2)
        ss_res_list.append(ss_res)
    runs.append([ss_tot, ss_res_list])
    continue
        #r2s.append(1 - ss_res / ss_tot)
    #for i in range(20):
    #    #ss_res = np.sum((glm_list[i].predicted[0].get_fdata() - f_data)**2)
    #    ss_res = np.mean((glm_list[i].residuals[0].get_fdata())**2, axis=-1)
    #
    #    r2s.append(1-ss_res/ss_tot)
    #    print(i, 1-ss_res/ss_tot)
    r2s = np.array(r2s)
    max_index = np.argmax(r2s, axis=0)

    d.max_index = max_index
    d.r2s = r2s
    #d.glm_list = glm_list

r2s_all = []
# iterate over voxels
for i in range(runs[0][1][0].shape[0]):
    r2s = []
    ss_tot = np.concatenate([runs[run_id][0][i] for run_id in range(4)])
    ss_tot = np.mean(ss_tot)
    for hrf_id in range(20):
        ss_res = np.concatenate([runs[run_id][1][hrf_id][i] for run_id in range(4)])
        ss_res = np.var(ss_res)
        r2s.append(1 - ss_res / ss_tot)
    print(r2s)
    r2s_all.append(r2s)

if 0:
    y = 32; x = 12; z = 40
    pred = fmri_glm.predicted[0].get_fdata()
    err = fmri_glm.residuals[0].get_fdata()
    plt.plot(pred[y, x, z])
    plt.plot(pred[y, x, z]+err[y, x, z])

if 0:
    r2s = []
    for i in range(20):
        ss_tot_list = []
        ss_res_list = []
        for ii, d in enumerate(data):
            ss_tot = np.var(d.glm_list[i].predicted[0].get_fdata() + d.glm_list[i].residuals[0].get_fdata(), axis=-1)
            #ss_res = np.sum((glm_list[i].predicted[0].get_fdata() - f_data)**2)
            ss_res = np.mean((d.glm_list[i].residuals[0].get_fdata())**2, axis=-1)

            ss_tot_list.append(ss_tot)
            ss_res_list.append(ss_res)

        ss_tot = np.sum(ss_tot_list, axis=0)
        ss_res = np.sum(ss_res_list, axis=0)
        r2s.append(1-ss_res/ss_tot)
        print(i, 1-ss_res/ss_tot)

for i, d in enumerate(data):
    plt.subplot(2, 2, i+1)
    plt.imshow(d.max_index[:, :, 40], vmin=0, vmax=19, cmap="viridis", interpolation="nearest")
plt.savefig(f"sub-{subject}_run-1-4_unordered.png")

if 0:
    i = 0
    xx = glm_list[i].predicted[0].get_fdata() + glm_list[i].residuals[0].get_fdata()
    i = 2
    xx2 = glm_list[i].predicted[0].get_fdata() + glm_list[i].residuals[0].get_fdata()
    print(xx[35, 12, 40]-xx2[35, 12, 40])
    print()
    #35, 12, 40
    plt.plot()

    predicted_list = []
    for i in range(20):
        predicted_list.append(np.var(glm_list[i].predicted[0].get_fdata(), axis=-1))
    predicted_list = np.array(predicted_list)
    max_index = np.argmax(predicted_list, axis=0)

    max_value = np.max(predicted_list, axis=0)
    min_value = np.min(predicted_list, axis=0)
    ratio = np.abs(max_value)/np.abs(min_value)

    err = fmri_glm.residuals[0].get_fdata()
    pred = fmri_glm.predicted[0].get_fdata()
    orig = d.fmri_img.get_fdata()

    frac_explained = 1 - np.var(err, axis=-1)/(np.var(orig, axis=-1))
    frac_explained = np.clip(frac_explained, -10, 10)
    plt.imshow(frac_explained[:, :, 40])

    plt.subplot(121)
    plt.imshow(np.log(np.var(orig, axis=-1)[:, :, 40]), vmin=0, vmax=10)
    plt.subplot(122)
    plt.imshow(np.log(np.var(pred, axis=-1)[:, :, 40]), vmin=0, vmax=10)

    plt.subplot(121)
    plt.imshow(max_index[:, :, 40], vmin=0, vmax=19, cmap="viridis")
    plt.subplot(122)
    plt.imshow(max_index_1s[:, :, 40], vmin=0, vmax=19, cmap="viridis")

    plt.imshow(ratio[:, :, 40])

    plt.clf()
