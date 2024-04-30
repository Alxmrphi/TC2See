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


def get_stim(name):
    if name == "+":
        return name
    m = re.match(r"images/(.*)_\d*\D(?:_hash_\d)?.jpg", name)
    return m.groups()[0]

def get_design_matrix(frame_times, events, keep_order=False, glm_single_index=None, tr=None, duration=2, shuffle=False, time_offset=0, add_regs=None):
    print("get_design_matrix")
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

        try:
            if keep_order is True:
                index += 1
                return f"_{index:03d}" + path  # .replace(".", "_").replace("_", "")[4:]
            return path  # .replace(".", "_").replace("_", "")[4:]
        except AttributeError:
            raise ValueError(Path(path).stem)

    events = events[events.stimulus != "+"].copy()
    if shuffle is not False:
        np.random.seed(shuffle)
        x = np.array(events["onset"])
        np.random.shuffle(x)
        events["onset"] = x
    events["onset"] += time_offset
    events["trial_type"] = [category(s) for s in [get_stim(ss) for ss in events.stimulus]]
    print("trial_type", events["trial_type"])

    length = frame_times[-1]
    events = events[events.onset < (length - 6)]

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


def average_masks(data_list):
    masks = []
    for d in data_list:
        masks.append(d.mask.get_fdata())
    mean_mask = np.round(np.mean(masks, axis=0))
    mean_mask = ni.image.new_img_like(d.mask, mean_mask)
    return mean_mask

def detrend(im):
    def lin_fit(x_data, y_data, axis=0):
        x_mean = np.mean(x_data, axis=axis)
        y_mean = np.mean(y_data, axis=axis)
        b = np.sum((x_data - x_mean) * (y_data - y_mean), axis=axis) / np.sum((x_data - x_mean) ** 2, axis=axis)
        a = y_mean - (b * x_mean)
        return a, b

    time_steps, voxel_count = im.shape
    x = np.tile(np.arange(time_steps), [voxel_count, 1]).T
    a, b = lin_fit(x, im, axis=0)
    im -= x * b + a

    #im -= np.mean(im, axis=0)[None, :]
    im /= (np.std(im, axis=0)[None, :] + 1e-3)
    return im

def z_score(im):
    im -= np.mean(im, axis=0)[None, :]
    im /= np.std(im, axis=0)[None, :]
    return im

def join_fmri_img(data_list, mean_mask):

    img = []
    for d in data_list:
        im = ni.masking.apply_mask(d.fmri_img, mean_mask)
        im = detrend(im)
        #im, _ = ni.glm.first_level.mean_scaling(im)
        img.append(im)
    img = np.vstack(img)
    #im = z_score(im)
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
                                                          glm_single_index=i, tr=d.tr, duration=d.duration,
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


def get_joined_design_matrix_grouped(data_list, glm_single_index, confounds=None, shuffle=False, time_offset=0, keep_order=False):
    from nilearn.glm.first_level import make_first_level_design_matrix
    dm = []
    counts = {}
    for ii, d in enumerate(data_list):
        length = d.fmri_img.shape[-1]
        duration = d.duration
        events = d.events[d.events.onset < (length - 6) * d.tr]
        events = events[events.stimulus != "+"]
        events = events.copy()

        def get_stim(name, index):
            m = re.match(r"images/(.*)_\d*\D(?:_hash_\d)?.jpg", name)
            name = m.groups()[0]
            if name not in counts:
                counts[name] = 0
            else:
                counts[name] += 1
            return f"_{counts[name] // 1}_"+name

        events["trial_type"] = [get_stim(n, i) for i, n in enumerate(events.stimulus)]

        frame_times = np.arange(d.fmri_img.shape[3]) * d.tr

        if glm_single_index is not None:
            hrfs = getcanonicalhrflibrary(duration, tr / 50)
            def hrf(tf, oversampling):
                return hrfs[glm_single_index]
        else:
            hrf = "spm"

        design_matrix = make_first_level_design_matrix(frame_times, events[["onset", "duration", "trial_type"]],
                                                       hrf_model=hrf, add_regs=d.confounds[confounds] if confounds is not None else None
                                                       )
        renames = {}
        for col in design_matrix.columns:
            if (col.startswith("drift") or col.startswith("constant") or (confounds is not None and col in confounds)):
                renames[col] = col+"_"+str(ii)
        design_matrix = design_matrix.rename(columns=renames)
        dm.append(design_matrix)

    dm2 = pd.concat(dm).fillna(0)
    return dm2




def get_betas_from_data(img, data_list, glm_single_index, confounds=None, keep_order=True):
    from nilearn.glm.first_level import make_first_level_design_matrix
    dm = []
    counts = {}
    name_mapping = {}
    for ii, d in enumerate(data_list):
        length = d.fmri_img.shape[-1]
        duration = d.duration
        events = d.events[d.events.onset < (length - 6) * d.tr]
        events = events[events.stimulus != "+"]
        events = events.copy()

        def get_stim(name, index):
            m = re.match(r"images/(.*)_\d*\D(?:_hash_\d)?.jpg", name)
            name = m.groups()[0]
            if name not in counts:
                counts[name] = 0
            else:
                counts[name] += 1
            colname =  f"_{counts[name] // 1}_"+name
            name_mapping[colname] = name
            return colname

        events["trial_type"] = [get_stim(n, i) for i, n in enumerate(events.stimulus)]

        frame_times = np.arange(d.fmri_img.shape[3]) * d.tr

        if glm_single_index is not None:
            hrfs = getcanonicalhrflibrary(duration, tr / 50)
            def hrf(tf, oversampling):
                return hrfs[glm_single_index]
        else:
            hrf = "spm"

        design_matrix = make_first_level_design_matrix(frame_times, events[["onset", "duration", "trial_type"]],
                                                       hrf_model=hrf, add_regs=d.confounds[confounds] if confounds is not None else None,
                                                       drift_model=None)
        renames = {}
        for col in design_matrix.columns:
            if (col.startswith("drift") or col.startswith("constant") or (confounds is not None and col in confounds)):
                renames[col] = col+"_"+str(ii)
        design_matrix = design_matrix.rename(columns=renames)
        dm.append(design_matrix)

    dm2 = pd.concat(dm).fillna(0)
    beta, r2 = get_beta_r2(img, dm2)
    d2 = dm2 @ beta
    betas = []
    names = []
    dm_new = []
    for i, colname in enumerate(dm2.columns):
        if colname in name_mapping:
            betas.append(beta[i])
            names.append(name_mapping[colname])
            dm_new.append(dm2[colname])
    betas = np.array(betas)
    names = np.array(names)
    d2 = np.asarray(d2)
    dm_new = np.asarray(dm_new)

    return betas, names, r2, dm2, d2, dm_new


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


def get_names(design_matrix):
    names = []
    for index, col in enumerate(design_matrix.columns):
        m = re.match("(?:_\d*)?_?([^_]+)(?:_\d*\D_hash)?(?:_hrf)?", col)
        if m:
            name = m.groups()[0]
            if name == "air":
                name = "air_pump"
            names.append(name)
        else:
            names.append(col)
    return names

def signal_noise_ratio(design_matrix, beta, mean_mask=None, randomize=False):
    ids = {}
    next_id = 0
    id_order = []
    cols = np.array(design_matrix.columns)
    if randomize is True:
        np.random.shuffle(cols)
    for index, col in enumerate(cols):
        #m = re.match("_\d\d\d(.*)_hrf_\d+", col)
        m = re.match("_\d\d\d([^_]*)_.*", col)
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
    if mean_mask is not None:
        ncsnr = nilearn.masking.unmask(ncsnr, mean_mask).get_fdata()
    return ncsnr


def signal_noise_ratio_names(beta, names, mean_mask=None, randomize=False):
    ids = {}
    next_id = 0
    id_order = []
    for index, name in enumerate(names):
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
    if mean_mask is not None:
        ncsnr = nilearn.masking.unmask(ncsnr, mean_mask).get_fdata()
    return ncsnr

def noise_ceilling(design_matrix, beta, mean_mask):
    ncsnr = signal_noise_ratio(design_matrix, beta, mean_mask)
    return compute_nc(ncsnr, num_averages=1)

def beta_per_image(design_matrix, beta, mean_mask):
    ids = {}
    next_id = 0
    id_order = []
    for index, col in enumerate(design_matrix.columns):
        #m = re.match("_\d\d\d(.*)_hrf_\d+", col)
        m = re.match("_\d\d\d([^_]*)_.*", col)
        if m:
            name = m.groups()[0]
            if name not in ids:
                ids[name] = next_id
                next_id += 1
                id_order.append([])
            id_order[ids[name]].append(index)

    sim_beta = []
    for a in id_order:
        stimulus_betas = beta[a]
        sim_beta.append(stimulus_betas)
    sim_beta = np.array(sim_beta)

    std = np.std(sim_beta, axis=1)
    rep = np.mean(std, axis=0)

    rep_brain = nilearn.masking.unmask(rep, mean_mask).get_fdata()
    return sim_beta

def compute_nc(ncsnr: np.ndarray, num_averages: int):
    ncsnr_squared = ncsnr ** 2
    nc = 100. * ncsnr_squared / (ncsnr_squared + (1. / num_averages))
    return nc

def get_activations(d, mask):
    if isinstance(d, list):
        g = []; n = []
        for dd in d:
            gg, nn = get_activations(dd, mask)
            g.extend(gg)
            n.extend(nn)
        return np.array(g), np.array(n)
    length = d.fmri_img.shape[-1]
    events = d.events[d.events.onset < (length - 6) * d.tr]
    events = events[events.stimulus != "+"]
    events = events.copy()

    def get_stim(name):
        m = re.match(r"images/(.*)_\d*\D(?:_hash_\d)?.jpg", name)
        return m.groups()[0]

    events["stimulus"] = [get_stim(n) for n in events.stimulus]

    indices = np.array((events.onset / d.tr)).astype(np.int)

    im = ni.masking.apply_mask(d.fmri_img, mask)
    im = detrend(im)

    gamma = im[indices + 2]
    names = events.stimulus
    return gamma, names

##

def filter_image_types(gamma, all_names, s1, s2, roll):
    included_names = np.roll(list(set(all_names)), roll)[s1:s2]
    index = np.array([a in included_names for a in all_names])
    return gamma[index], np.array(all_names)[index]

def filter_image_repetitions(gamma, all_names, s1, s2, roll):
    valid_counts = np.roll(np.arange(0, 3), roll)[s1:s2]
    counts = {name: 0 for name in set(all_names)}
    indices = []
    for i, name in enumerate(all_names):
        counts[name] += 1
        if counts[name] not in valid_counts:
            continue
        indices.append(i)
    #return indices
    index = np.array(indices)
    #print("index", index)
    return gamma[index], np.array(all_names)[index]

def filter(types=None, repetitions=None):
    def do_filter(x, y):
        if types is not None:
            x, y = filter_image_types(x, y, types[0], types[1], types[2])
        if repetitions is not None:
            #print("repetitions", repetitions)
            x, y = filter_image_repetitions(x, y, repetitions[0], repetitions[1], repetitions[2])
        return x, y
    return do_filter

##
def splitter(s1, e1, r1, s2, e2, r2):
    def split_image(x, y):
        y = np.asarray(y)
        if 0:
            x_train, y_train = filter_image_types(x, y, 0, 30, 0)
            x_test, y_test = filter_image_types(x, y, 30, 30 + 6, 0)
        elif 1:
            x_train, y_train = filter_image_repetitions(x, y, s1, e1, r1)
            x_test, y_test = filter_image_repetitions(x, y, s2, e2, r2)
        else:
            cross = 5
            N = x.shape[0]
            frac = N - (N // cross)
            count_range = np.arange(N)
            acc = []
            i = 1
            indices = np.roll(count_range, i * (N // cross))
            train = indices[:frac]
            test = indices[frac:]

            x_train, y_train = x[train], y[train]
            x_test, y_test = x[test], y[test]

        #print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        #print({i: list(np.array(y_train)).count(i) for i in np.array(y_train)})
        #print({i: list(np.array(y_test)).count(i) for i in np.array(y_test)})

        return (x_train, x_test, x_train), (y_train, y_test, y_train)
    return split_image


def splitter2(s1, e1, r1, s2, e2, r2, s3, e3, r3):
    def split_image(x, y):
        y = np.asarray(y)
        if 0:
            x_train, y_train = filter_image_types(x, y, 0, 30, 0)
            x_test, y_test = filter_image_types(x, y, 30, 30 + 6, 0)
        elif 1:
            x_train, y_train = filter_image_repetitions(x, y, s1, e1, r1)
            x_test, y_test = filter_image_repetitions(x, y, s2, e2, r2)
            x_snr, y_snr = filter_image_repetitions(x, y, s3, e3, r3)
        else:
            cross = 5
            N = x.shape[0]
            frac = N - (N // cross)
            count_range = np.arange(N)
            acc = []
            i = 1
            indices = np.roll(count_range, i * (N // cross))
            train = indices[:frac]
            test = indices[frac:]

            x_train, y_train = x[train], y[train]
            x_test, y_test = x[test], y[test]

        #print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        #print({i: list(np.array(y_train)).count(i) for i in np.array(y_train)})
        #print({i: list(np.array(y_test)).count(i) for i in np.array(y_test)})

        return (x_train, x_test, x_snr), (y_train, y_test, y_snr)
    return split_image

def splitter3(
        filter1,
        filter2,
        filter3
):
    def split_image(x, y):
        y = np.asarray(y)

        x_train, y_train = filter1(x, y)
        x_test, y_test = filter2(x, y)
        x_snr, y_snr = filter3(x, y)

        return (x_train, x_test, x_snr), (y_train, y_test, y_snr)
    return split_image

def split_image(x, y):
    y = np.asarray(y)
    if 0:
        x_train, y_train = filter_image_types(x, y, 0, 30, 0)
        x_test, y_test = filter_image_types(x, y, 30, 30+6, 0)
    elif 1:
        x_train, y_train = filter_image_repetitions(x, y, 0, 6, 0)
        x_test, y_test = filter_image_repetitions(x, y, 6, 8, 0)
    else:
        cross = 5
        N = x.shape[0]
        frac = N - (N // cross)
        count_range = np.arange(N)
        acc = []
        i = 1
        indices = np.roll(count_range, i * (N // cross))
        train = indices[:frac]
        test = indices[frac:]

        x_train, y_train = x[train], y[train]
        x_test, y_test = x[test], y[test]

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    print({i: list(np.array(y_train)).count(i) for i in np.array(y_train)})
    print({i: list(np.array(y_test)).count(i) for i in np.array(y_test)})

    return (x_train, x_test, x_train), (y_train, y_test, y_train)
##

def scramble_embedding(embeddings):
    embeddings_scrambled = {}
    names_scrambled = list(embeddings.keys())
    np.random.shuffle(names_scrambled)
    for name, names2 in zip(embeddings.keys(), names_scrambled):
        embeddings_scrambled[names2] = embeddings[name]

    return embeddings_scrambled

def one_model(x, y, split, embeddings, filter_count=1000, model=None, scramble_emb=False, scramble_brain=False):
    (x_train, x_test, x_snr), (y_train, y_test, y_snr) = split(x, y)

    snr = signal_noise_ratio_names(x_snr, y_snr)
    order = np.argsort(snr)[::-1]
    filter = order[:filter_count]

    x_train = x_train[:, filter]
    x_test = x_test[:, filter]

    if scramble_emb is True:
        embeddings = scramble_embedding(embeddings)
    if scramble_brain is True:
        N = x_train.shape[0]
        random_ind = np.arange(N)
        np.random.shuffle(random_ind)
        x_train = x_train[random_ind]

    y_test = np.asarray([embeddings[name] for name in y_test])
    y_train = np.asarray([embeddings[name] for name in y_train])
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    #y_test = y_test[:, :10]
    #y_train = y_train[:, :10]

    clf = model
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    distances = cosine_distance(y_test[None], y_pred[:, None])
    return two_versus_two(distances)


def one_lasso(x, y, split, embeddings, filter_count=1000):
    (x_train, x_test), (y_train, y_test) = split(x, y)

    snr = signal_noise_ratio_names(x_train, y_train)
    order = np.argsort(snr)[::-1]
    filter = order[:filter_count]

    x_train = x_train[:, filter]
    x_test = x_test[:, filter]

    y_test = np.asarray([embeddings[name] for name in y_test])
    y_train = np.asarray([embeddings[name] for name in y_train])
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    from sklearn.linear_model import Ridge, Lasso
    #clf = Ridge(alpha=1.0)
    clf = Lasso(alpha=1.0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y = y_test

    distances = cosine_distance(y[None], y_pred[:, None])
    return two_versus_two(distances)

def n_folder_ridge(gammas, target_embeddings_scr):
    from sklearn.linear_model import Ridge
    cross = 5
    N = gammas.shape[0]
    frac = N - (N // cross)
    count_range = np.arange(N)
    acc = []
    for i in range(cross):
        indices = np.roll(count_range, i * (N // cross))
        train = indices[:frac]
        test = indices[frac:]

        x_train, y_train = gammas[train], target_embeddings_scr[train]
        x_test, y_test = gammas[test], target_embeddings_scr[test]

        clf = Ridge(alpha=1.0)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        y = y_test

        distances = cosine_distance(y[None], y_pred[:, None])
        acc.append(two_versus_two(distances))
    return acc


def cosine_distance(Y1, Y2):
    Y1 = Y1 / np.linalg.norm(Y1, axis=-1)[..., None]
    Y2 = Y2 / np.linalg.norm(Y2, axis=-1)[..., None]
    return 1. - np.einsum('... i, ... i -> ...', Y1, Y2)

def two_versus_two(distances, eps=1e-7):
    different = distances + distances.T

    distances_diag = np.diag(distances)
    same = distances_diag[None, :] + distances_diag[:, None] #- eps

    comparison = same < different
    return np.mean(comparison[np.triu_indices(distances.shape[0], k=1)])