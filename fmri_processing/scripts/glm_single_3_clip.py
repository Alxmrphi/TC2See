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
if 0:


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

##
beta, beta_names, r2, dm, d2s, dm_new = get_betas_from_data(img, data_list, None)
beta_filter_run2 = np.array([name in names_2_unique for name in beta_names])
beta_filter_run1 = np.array([name in names_1_unique for name in beta_names])

plt.clf()
d2s = np.asarray(d2s)
plt.plot(img[:, filter[0]])
plt.plot(d2s[:, filter[0]])
b = beta[:, filter[0]]
parts = b[:, None] * dm_new
plt.plot(parts.T)

gammas = []
all_names = []
for d in data:
    gamma, names = get_activations(d, mean_mask)
    gammas.extend(gamma)
    all_names.extend(list(names))

gamma_1, names_1 = get_activations(data[1], mean_mask)
names_1_unique = set(names_1)
gamma_2, names_2 = get_activations(data[2], mean_mask)
names_2_unique = set(names_2)

filter_run2 = np.array([name in names_2_unique for name in all_names])
filter_run1 = np.array([name in names_1_unique for name in all_names])

gammas = np.asarray(gammas)

embeddings = np.load("embeddings.npy", allow_pickle=True)[()]
target_embeddings = np.asarray([embeddings[name] for name in all_names])

print({i:all_names.count(i) for i in all_names})
def filter_counts(all_names, max_count):
    counts = {}
    indices = []
    for i, name in enumerate(all_names):
        if name not in counts:
            counts[name] = 0
        counts[name] += 1
        if counts[name] > max_count:
            continue
        indices.append(i)
    return indices

indces = filter_counts(all_names, 3)
print({i: list(np.array(all_names)[indces]).count(i) for i in np.array(all_names)[indces]})

def filter_image_types(gamma, all_names, s1, s2, roll):
    included_names = np.roll(list(set(all_names)), roll)[s1:s2]
    index = np.array([a in included_names for a in all_names])
    return gamma[index], np.array(all_names)[index]

def filter_image_repetitions(gamma, all_names, s1, s2, roll):
    valid_counts = np.roll(np.arange(0, 10), roll)[s1:s2]
    counts = {}
    indices = []
    for i, name in enumerate(all_names):
        if name not in counts:
            counts[name] = -1
        counts[name] += 1
        if counts[name] not in valid_counts:
            continue
        indices.append(i)
    #return indices
    index = np.array(indices)
    return gamma[index], np.array(all_names)[index]

def filter_im(gamma, all_names, image_number, image_repetitions, roll1=0, roll2=0):
    if image_number is not None:
        index = get_fist_n_image_types(all_names, image_number, roll1)
        all_names = np.array(all_names)[index]
        gamma = gamma[index]
    if image_repetitions is not None:
        index2 = filter_counts(all_names, image_repetitions)
        return gamma[index2], np.array(all_names)[index2]
    else:
        return gamma, all_names

xx, yy = filter_im(gammas, all_names, None, 12)
filter_image_repetitions(gammas, all_names, 1, 9, 0)
(x_train, x_test), (y_train, y_test) = split_image(xx, yy)

snr = signal_noise_ratio_names(gammas, all_names)
order = np.argsort(snr)[::-1]
filter = order[:1000]

from sklearn.linear_model import Ridge, Lasso

def getList(func):
    acc_over_list = []
    for ii in range(1, 10):
        acc_list = []
        for i in range(5):
            acc = func(i, ii)
            print(acc)
            acc_list.append(acc)
        acc_over_list.append(acc_list)
    acc_over_list = np.array(acc_over_list)
    return acc_over_list

acc_over_list = getList(lambda i, ii: one_model(gammas, all_names, splitter(2, 2+int(ii), i*2, 0, 2, i*2), embeddings, 5000, model=Ridge()))
acc_over_list2 = getList(lambda i, ii: one_model(gammas, all_names, splitter(2, 2+int(ii), i*2, 0, 2, i*2), embeddings, 5000, model=Ridge(), scramble_emb=True))
acc_over_list3 = getList(lambda i, ii: one_model(gammas, all_names, splitter(2, 2+int(ii), i*2, 0, 2, i*2), embeddings, 5000, model=Ridge(), scramble_brain=True))
print(one_model(gammas, all_names, split_image, embeddings, 10000, model=Lasso()))
plt.errorbar(range(1, 10), np.mean(acc_over_list, axis=1), np.std(acc_over_list, axis=1), label="clip")
plt.errorbar(range(1, 10), np.mean(acc_over_list2, axis=1), np.std(acc_over_list2, axis=1), label="scr clip")
plt.errorbar(range(1, 10), np.mean(acc_over_list3, axis=1), np.std(acc_over_list3, axis=1), label="scr brain")


acc_over_list_B = getList(lambda i, ii: one_model(gammas[filter_run2], np.array(all_names)[filter_run2], splitter(2, 2+int(ii), i*2, 0, 2, i*2), embeddings, 5000, model=Ridge()))
acc_over_list2_B = getList(lambda i, ii: one_model(gammas[filter_run2], np.array(all_names)[filter_run2], splitter(2, 2+int(ii), i*2, 0, 2, i*2), embeddings, 5000, model=Ridge(), scramble_emb=True))
acc_over_list3_B = getList(lambda i, ii: one_model(gammas[filter_run2], np.array(all_names)[filter_run2], splitter(2, 2+int(ii), i*2, 0, 2, i*2), embeddings, 5000, model=Ridge(), scramble_brain=True))
print(one_model(gammas, all_names, split_image, embeddings, 10000, model=Lasso()))
ax = plt.subplot(121)
plt.errorbar(range(1, 10), np.mean(acc_over_list_B, axis=1), np.std(acc_over_list_B, axis=1), label="clip")
plt.errorbar(range(1, 10), np.mean(acc_over_list2_B, axis=1), np.std(acc_over_list2_B, axis=1), label="scr clip")
plt.errorbar(range(1, 10), np.mean(acc_over_list3_B, axis=1), np.std(acc_over_list3_B, axis=1), label="scr brain")
plt.axhline(0.5, color="k", lw=0.8)

acc_over_list_C = getList(lambda i, ii: one_model(gammas[filter_run1], np.array(all_names)[filter_run1], splitter(2, 2+int(ii), i*2, 0, 2, i*2), embeddings, 5000, model=Ridge()))
acc_over_list2_C = getList(lambda i, ii: one_model(gammas[filter_run1], np.array(all_names)[filter_run1], splitter(2, 2+int(ii), i*2, 0, 2, i*2), embeddings, 5000, model=Ridge(), scramble_emb=True))
acc_over_list3_C = getList(lambda i, ii: one_model(gammas[filter_run1], np.array(all_names)[filter_run1], splitter(2, 2+int(ii), i*2, 0, 2, i*2), embeddings, 5000, model=Ridge(), scramble_brain=True))
print(one_model(gammas, all_names, split_image, embeddings, 10000, model=Lasso()))
plt.subplot(122, sharex=ax, sharey=ax)
plt.errorbar(range(1, 10), np.mean(acc_over_list_C, axis=1), np.std(acc_over_list_C, axis=1), label="clip")
plt.errorbar(range(1, 10), np.mean(acc_over_list2_C, axis=1), np.std(acc_over_list2_C, axis=1), label="scr clip")
plt.errorbar(range(1, 10), np.mean(acc_over_list3_C, axis=1), np.std(acc_over_list3_C, axis=1), label="scr brain")
plt.axhline(0.5, color="k", lw=0.8)


if 0:
    acc_over_list_beta_B = getList(lambda i, ii: one_model(beta[beta_filter_run2], np.array(beta_names)[beta_filter_run2],
                                                      splitter(2, 2 + int(ii), i * 2, 0, 2, i * 2), embeddings, 5000,
                                                      model=Ridge()))
    acc_over_list2_beta_B = getList(lambda i, ii: one_model(beta[beta_filter_run2], np.array(beta_names)[beta_filter_run2],
                                                       splitter(2, 2 + int(ii), i * 2, 0, 2, i * 2), embeddings, 5000,
                                                       model=Ridge(), scramble_emb=True))
    acc_over_list3_beta_B = getList(lambda i, ii: one_model(beta[beta_filter_run2], np.array(beta_names)[beta_filter_run2],
                                                       splitter(2, 2 + int(ii), i * 2, 0, 2, i * 2), embeddings, 5000,
                                                       model=Ridge(), scramble_brain=True))
    print(one_model(gammas, all_names, split_image, embeddings, 10000, model=Lasso()))



    acc_over_list_beta_C = getList(lambda i, ii: one_model(beta[beta_filter_run1], np.array(beta_names)[beta_filter_run1],
                                                      splitter(2, 2 + int(ii), i * 2, 0, 2, i * 2), embeddings, 5000,
                                                      model=Ridge()))
    acc_over_list2_beta_C = getList(lambda i, ii: one_model(beta[beta_filter_run1], np.array(beta_names)[beta_filter_run1],
                                                       splitter(2, 2 + int(ii), i * 2, 0, 2, i * 2), embeddings, 5000,
                                                       model=Ridge(), scramble_emb=True))
    acc_over_list3_beta_C = getList(lambda i, ii: one_model(beta[beta_filter_run1], np.array(beta_names)[beta_filter_run1],
                                                       splitter(2, 2 + int(ii), i * 2, 0, 2, i * 2), embeddings, 5000,
                                                       model=Ridge(), scramble_brain=True))
    print(one_model(gammas, all_names, split_image, embeddings, 10000, model=Lasso()))

    plt.figure(5)
    plt.clf()
    ax = plt.subplot(121)
    plt.errorbar(range(1, 10), np.mean(acc_over_list_beta_B, axis=1), np.std(acc_over_list_beta_B, axis=1), label="clip")
    plt.errorbar(range(1, 10), np.mean(acc_over_list2_beta_B, axis=1), np.std(acc_over_list2_beta_B, axis=1), label="scr clip")
    plt.errorbar(range(1, 10), np.mean(acc_over_list3_beta_B, axis=1), np.std(acc_over_list3_beta_B, axis=1), label="scr brain")
    plt.axhline(0.5, color="k", lw=0.8)
    plt.subplot(122, sharex=ax, sharey=ax)
    plt.errorbar(range(1, 10), np.mean(acc_over_list_beta_C, axis=1), np.std(acc_over_list_beta_C, axis=1), label="clip")
    plt.errorbar(range(1, 10), np.mean(acc_over_list2_beta_C, axis=1), np.std(acc_over_list2_beta_C, axis=1), label="scr clip")
    plt.errorbar(range(1, 10), np.mean(acc_over_list3_beta_C, axis=1), np.std(acc_over_list3_beta_C, axis=1), label="scr brain")
    plt.axhline(0.5, color="k", lw=0.8)


def filter(types=None, repetitions=None):
    def do_filter(x, y):
        if types is not None:
            x, y = filter_image_types(x, y, types[0], types[1], types[2])
        if repetitions is not None:
            x, y = filter_image_repetitions(x, y, repetitions[0], repetitions[1], repetitions[2])
        return x, y
    return do_filter


def getList2(func, pairs):
    acc_over_list = []
    xx = []
    for ii, jj in pairs:
        xx.append(ii)
        acc_list = []
        for i in range(5):
            acc = func(i, ii, jj)
            print(acc)
            acc_list.append(acc)
        acc_over_list.append(acc_list)
    acc_over_list = np.array(acc_over_list)
    return acc_over_list, xx
plt.figure(2)
pairs = [[1, 12], [2, 6], [3, 4], [4, 3], [6, 2], [12, 1]]
acc_over_list_new, xx = getList2(lambda i, ii, jj: one_model(gammas[filter_run2], np.array(all_names)[filter_run2],
                                                  splitter3(
                                                      filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                      filter(repetitions=[0, 2, i*2]),
                                                      filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                  ), embeddings, 5000, model=Ridge()), pairs)
plt.errorbar(xx, np.mean(acc_over_list_new, axis=1), np.std(acc_over_list_new, axis=1), label="clip")
pairs = [[2, 12], [3, 8], [4, 6], [6, 4], [8, 3], [12, 2]]
acc_over_list_new, xx = getList2(lambda i, ii, jj: one_model(gammas[filter_run2], np.array(all_names)[filter_run2],
                                                  splitter3(
                                                      filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                      filter(repetitions=[0, 2, i*2]),
                                                      filter(repetitions=[2, 2 + int(ii), i * 2], types=[0, jj, 0]),
                                                  ), embeddings, 5000, model=Ridge()), pairs)
plt.errorbar(xx, np.mean(acc_over_list_new, axis=1), np.std(acc_over_list_new, axis=1), label="clip")
pairs = [[3, 12], [4, 9], [6, 6], [9, 4]]
acc_over_list_new, xx = getList2(lambda i, ii, jj: one_model(gammas[filter_run2], np.array(all_names)[filter_run2],
                                                  splitter3(
                                                      filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                      filter(repetitions=[0, 2, i*2]),
                                                      filter(repetitions=[2, 2 + int(ii), i * 2], types=[0, jj, 0]),
                                                  ), embeddings, 5000, model=Ridge()), pairs)
plt.errorbar(xx, np.mean(acc_over_list_new, axis=1), np.std(acc_over_list_new, axis=1), label="clip")

pairs = [[4, 12], [6, 8], [8, 6]]
acc_over_list_new, xx = getList2(lambda i, ii, jj: one_model(gammas[filter_run2], np.array(all_names)[filter_run2],
                                                  splitter3(
                                                      filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                      filter(repetitions=[0, 2, i*2]),
                                                      filter(repetitions=[2, 2 + int(ii), i * 2], types=[0, jj, 0]),
                                                  ), embeddings, 5000, model=Ridge()), pairs)
plt.errorbar(xx, np.mean(acc_over_list_new, axis=1), np.std(acc_over_list_new, axis=1), label="clip")

pairs = [[5, 12], [6, 10], [10, 6]]
acc_over_list_new, xx = getList2(lambda i, ii, jj: one_model(gammas[filter_run2], np.array(all_names)[filter_run2],
                                                  splitter3(
                                                      filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                      filter(repetitions=[0, 2, i*2]),
                                                      filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                  ), embeddings, 5000, model=Ridge()), pairs)
plt.errorbar(xx, np.mean(acc_over_list_new, axis=1), np.std(acc_over_list_new, axis=1), label="clip")

pairs = [[6, 12], [8, 9], [9, 8]]
acc_over_list_new, xx = getList2(lambda i, ii, jj: one_model(gammas[filter_run2], np.array(all_names)[filter_run2],
                                                  splitter3(
                                                      filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                      filter(repetitions=[0, 2, i*2]),
                                                      filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                  ), embeddings, 5000, model=Ridge()), pairs)
plt.errorbar(xx, np.mean(acc_over_list_new, axis=1), np.std(acc_over_list_new, axis=1), label="clip")

if 1:
    plt.figure(3)
    pairs = [[1, 12], [2, 6], [3, 4], [4, 3], [6, 2], [12, 1]]
    acc_over_list_new, xx = getList2(lambda i, ii, jj: one_model(gammas[filter_run2], np.array(all_names)[filter_run2],
                                                                 splitter3(
                                                                     filter(repetitions=[2, 2 + int(ii), i * 2],
                                                                            types=[0, jj, 0]),
                                                                     filter(repetitions=[0, 2, i * 2], types=[0, jj, 0]),
                                                                     filter(repetitions=[2, 2 + int(ii), i * 2],
                                                                            types=[0, jj, 0]),
                                                                 ), embeddings, 5000, model=Ridge()), pairs)
    plt.errorbar(xx, np.mean(acc_over_list_new, axis=1), np.std(acc_over_list_new, axis=1), label="clip")
    pairs = [[2, 12], [3, 8], [4, 6], [6, 4], [8, 3], [12, 2]]
    acc_over_list_new, xx = getList2(lambda i, ii, jj: one_model(gammas[filter_run2], np.array(all_names)[filter_run2],
                                                                 splitter3(
                                                                     filter(repetitions=[2, 2 + int(ii), i * 2],
                                                                            types=[0, jj, 0]),
                                                                     filter(repetitions=[0, 2, i * 2], types=[0, jj, 0]),
                                                                     filter(repetitions=[2, 2 + int(ii), i * 2],
                                                                            types=[0, jj, 0]),
                                                                 ), embeddings, 5000, model=Ridge()), pairs)
    plt.errorbar(xx, np.mean(acc_over_list_new, axis=1), np.std(acc_over_list_new, axis=1), label="clip")
    pairs = [[3, 12], [4, 9], [6, 6], [9, 4]]
    acc_over_list_new, xx = getList2(lambda i, ii, jj: one_model(gammas[filter_run2], np.array(all_names)[filter_run2],
                                                                 splitter3(
                                                                     filter(repetitions=[2, 2 + int(ii), i * 2],
                                                                            types=[0, jj, 0]),
                                                                     filter(repetitions=[0, 2, i * 2], types=[0, jj, 0]),
                                                                     filter(repetitions=[2, 2 + int(ii), i * 2],
                                                                            types=[0, jj, 0]),
                                                                 ), embeddings, 5000, model=Ridge()), pairs)
    plt.errorbar(xx, np.mean(acc_over_list_new, axis=1), np.std(acc_over_list_new, axis=1), label="clip")

    pairs = [[4, 12], [6, 8], [8, 6]]
    acc_over_list_new, xx = getList2(lambda i, ii, jj: one_model(gammas[filter_run2], np.array(all_names)[filter_run2],
                                                                 splitter3(
                                                                     filter(repetitions=[2, 2 + int(ii), i * 2],
                                                                            types=[0, jj, 0]),
                                                                     filter(repetitions=[0, 2, i * 2], types=[0, jj, 0]),
                                                                     filter(repetitions=[2, 2 + int(ii), i * 2],
                                                                            types=[0, jj, 0]),
                                                                 ), embeddings, 5000, model=Ridge()), pairs)
    plt.errorbar(xx, np.mean(acc_over_list_new, axis=1), np.std(acc_over_list_new, axis=1), label="clip")

    pairs = [[5, 12], [6, 10], [10, 6]]
    acc_over_list_new, xx = getList2(lambda i, ii, jj: one_model(gammas[filter_run2], np.array(all_names)[filter_run2],
                                                                 splitter3(
                                                                     filter(repetitions=[2, 2 + int(ii), i * 2],
                                                                            types=[0, jj, 0]),
                                                                     filter(repetitions=[0, 2, i * 2], types=[0, jj, 0]),
                                                                     filter(repetitions=[2, 2 + int(ii), i * 2],
                                                                            types=[0, jj, 0]),
                                                                 ), embeddings, 5000, model=Ridge()), pairs)
    plt.errorbar(xx, np.mean(acc_over_list_new, axis=1), np.std(acc_over_list_new, axis=1), label="clip")

    pairs = [[6, 12], [8, 9], [9, 8]]
    acc_over_list_new, xx = getList2(lambda i, ii, jj: one_model(gammas[filter_run2], np.array(all_names)[filter_run2],
                                                                 splitter3(
                                                                     filter(repetitions=[2, 2 + int(ii), i * 2],
                                                                            types=[0, jj, 0]),
                                                                     filter(repetitions=[0, 2, i * 2], types=[0, jj, 0]),
                                                                     filter(repetitions=[2, 2 + int(ii), i * 2],
                                                                            types=[0, jj, 0]),
                                                                 ), embeddings, 5000, model=Ridge()), pairs)
    plt.errorbar(xx, np.mean(acc_over_list_new, axis=1), np.std(acc_over_list_new, axis=1), label="clip")
#plt.errorbar(np.arange(1, 10), np.mean(acc_over_list_B, axis=1), np.std(acc_over_list_B, axis=1), label="clip")

if 1:
    plt.figure(4)
    plt.clf()
    for reps in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        pairs = [[reps, 1], [reps, 2], [reps, 3], [reps, 4], [reps, 5], [reps, 6],
                 [reps, 7], [reps, 8], [reps, 9], [reps, 10], [reps, 11], [reps, 12]]
        acc_over_list_new, xxx = getList2(lambda i, ii, jj: one_model(gammas[filter_run2], np.array(all_names)[filter_run2],
                                                                      splitter3(
                                                                          filter(repetitions=[2, 8, i * 2],
                                                                                 ),
                                                                          filter(repetitions=[0, 2, i * 2],
                                                                                 ),
                                                                          filter(repetitions=[2, 2 + int(ii), i * 2],
                                                                                 types=[0, jj, 0]),
                                                                      ), embeddings, 5000, model=Ridge()), pairs)
        plt.errorbar([p[0]*p[1] for p in pairs], np.mean(acc_over_list_new, axis=1), np.std(acc_over_list_new, axis=1), label=reps)
    plt.legend(title="repetitions")
    plt.xlabel("stimuli (image types * image repetions) for SNR")
    plt.ylabel("2 vs 2 accuracy")


x, y = filter()(gammas[filter_run2], np.array(all_names)[filter_run2])
print({i: list(y).count(i) for i in y})
for ii, jj in pairs:
    print(ii, jj)
    x, y = gammas[filter_run2], np.array(all_names)[filter_run2]
    #x, y = filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0])(gammas[filter_run2], np.array(all_names)[filter_run2])
    x, y = filter_image_types(x, y, 0, jj, 0)
    print({i: list(y).count(i) for i in y})

N = gammas.shape[0]
random_ind = np.arange(N)
np.random.shuffle(random_ind)
print(one_model(gammas[random_ind], all_names, split_image, embeddings, 100000, model=Ridge()))
print(one_model(gammas[random_ind], all_names, split_image, embeddings, 100000, model=Lasso(alpha=0.25)))

print(one_ridge(gammas, all_names, split_image))


snr = signal_noise_ratio_names(all_names, gammas)
order = np.argsort(snr)[::-1]
filter = order[:100]



embeddings_scrambled = {}
names_scrambled = list(embeddings.keys())
np.random.shuffle(names_scrambled)
for name, names2 in zip(embeddings.keys(), names_scrambled):
    print(name, names2)
    embeddings_scrambled[names2] = embeddings[name]

target_embeddings_scr = np.asarray([embeddings_scrambled[name] for name in all_names])

print(gammas[:, filter].shape, target_embeddings.shape)

N = gammas.shape[0]
random_ind = np.arange(N)
np.random.shuffle(random_ind)
print(n_folder_ridge(gammas[:, filter], target_embeddings))
print(n_folder_ridge(gammas[:, filter], target_embeddings_scr))
print(n_folder_ridge(gammas[:, filter][random_ind], target_embeddings))

filter = order[:100]
acc_list = []
acc_list2 = []
acc_list3 = []
for i in range(1, 10):
    indces = filter_counts(all_names, i)
    acc_list.append(n_folder_ridge(gammas[:, filter][indces], target_embeddings[indces]))
    acc_list2.append(n_folder_ridge(gammas[:, filter][indces], target_embeddings_scr[indces]))
    N = gammas[:, filter][indces].shape[0]
    random_ind = np.arange(N)
    np.random.shuffle(random_ind)
    acc_list3.append(n_folder_ridge(gammas[:, filter][indces][random_ind], target_embeddings[indces]))

ax = plt.subplot(221)
acc_list = np.array(acc_list)
plt.errorbar(range(1, 10), np.mean(acc_list, axis=1), np.std(acc_list, axis=1), label="clip")
plt.errorbar(range(1, 10), np.mean(acc_list2, axis=1), np.std(acc_list2, axis=1), label="clip shuffled")
plt.errorbar(range(1, 10), np.mean(acc_list3, axis=1), np.std(acc_list3, axis=1), label="brain shuffled")
plt.xlabel("image repetitions")
plt.ylabel("2vs2 accuracy")
plt.legend()
plt.title("100 best snr voxels")
plt.axhline(0.5, color="k", lw=0.8)

filter = order[:500]
acc_list_B = []
acc_list2_B = []
acc_list3_B = []
for i in range(1, 10):
    indces = filter_counts(all_names, i)
    acc_list_B.append(n_folder_ridge(gammas[:, filter][indces], target_embeddings[indces]))
    acc_list2_B.append(n_folder_ridge(gammas[:, filter][indces], target_embeddings_scr[indces]))
    N = gammas[:, filter][indces].shape[0]
    random_ind = np.arange(N)
    np.random.shuffle(random_ind)
    acc_list3_B.append(n_folder_ridge(gammas[:, filter][indces][random_ind], target_embeddings[indces]))

plt.subplot(222, sharex=ax, sharey=ax)
acc_list = np.array(acc_list)
plt.errorbar(range(1, 10), np.mean(acc_list_B, axis=1), np.std(acc_list_B, axis=1), label="clip")
plt.errorbar(range(1, 10), np.mean(acc_list2_B, axis=1), np.std(acc_list2_B, axis=1), label="clip shuffled")
plt.errorbar(range(1, 10), np.mean(acc_list3_B, axis=1), np.std(acc_list3_B, axis=1), label="brain shuffled")
plt.xlabel("image repetitions")
plt.ylabel("2vs2 accuracy")
plt.legend()
plt.title("500 best snr voxels")
plt.axhline(0.5, color="k", lw=0.8)


filter = order[:5000]
acc_list_C = []
acc_list2_C = []
acc_list3_C = []
for i in range(1, 10):
    indces = filter_counts(all_names, i)
    acc_list_C.append(n_folder_ridge(gammas[:, filter][indces], target_embeddings[indces]))
    acc_list2_C.append(n_folder_ridge(gammas[:, filter][indces], target_embeddings_scr[indces]))
    N = gammas[:, filter][indces].shape[0]
    random_ind = np.arange(N)
    np.random.shuffle(random_ind)
    acc_list3_C.append(n_folder_ridge(gammas[:, filter][indces][random_ind], target_embeddings[indces]))

plt.subplot(223, sharex=ax, sharey=ax)
acc_list = np.array(acc_list)
plt.errorbar(range(1, 10), np.mean(acc_list_C, axis=1), np.std(acc_list_C, axis=1), label="clip")
plt.errorbar(range(1, 10), np.mean(acc_list2_C, axis=1), np.std(acc_list2_C, axis=1), label="clip shuffled")
plt.errorbar(range(1, 10), np.mean(acc_list3_C, axis=1), np.std(acc_list3_C, axis=1), label="brain shuffled")
plt.xlabel("image repetitions")
plt.ylabel("2vs2 accuracy")
plt.legend()
plt.title("5000 best snr voxels")
plt.axhline(0.5, color="k", lw=0.8)


filter = order[:]
acc_list_D = []
acc_list2_D = []
acc_list3_D = []
for i in range(1, 10):
    indces = filter_counts(all_names, i)
    acc_list_D.append(n_folder_ridge(gammas[:, filter][indces], target_embeddings[indces]))
    acc_list2_D.append(n_folder_ridge(gammas[:, filter][indces], target_embeddings_scr[indces]))
    N = gammas[:, filter][indces].shape[0]
    random_ind = np.arange(N)
    np.random.shuffle(random_ind)
    acc_list3_D.append(n_folder_ridge(gammas[:, filter][indces][random_ind], target_embeddings[indces]))


plt.subplot(224, sharex=ax, sharey=ax)
acc_list = np.array(acc_list)
plt.errorbar(range(1, 10), np.mean(acc_list_D, axis=1), np.std(acc_list_D, axis=1), label="clip")
plt.errorbar(range(1, 10), np.mean(acc_list2_D, axis=1), np.std(acc_list2_D, axis=1), label="clip shuffled")
plt.errorbar(range(1, 10), np.mean(acc_list3_D, axis=1), np.std(acc_list3_D, axis=1), label="brain shuffled")
plt.xlabel("image repetitions")
plt.ylabel("2vs2 accuracy")
plt.legend()
plt.title(f"all {filter.shape[0]} voxels")
plt.axhline(0.5, color="k", lw=0.8)