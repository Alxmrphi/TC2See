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
from sklearn.linear_model import Ridge, Lasso


from fmri_processing.data_loading import Data
from nilearn.glm.first_level import make_first_level_design_matrix

from glmsingle.glmsingle import getcanonicalhrflibrary
from glm_single_3_include import *

subject = "06"
run = "1"
task = "bird"
space = "T1w"
#space = "MNI152NLin2009cAsym"
tr = 2
project = "TC2See"

mask_type = "gm"
mask_type = "visual"

rsm_function = "cosine"
rsm_function = "pearson"

data = []
for i in range(6):
    data.append(Data(project, subject, task, str(i+1), space))

from fmri_processing.analysis import fit_glm, get_glm_activations, get_rsm, plot_rsm
for i, d in enumerate(data):
    d.duration = 5.5


##
data_list = [data[0], data[1], data[2], data[3], data[4], data[5]]
mean_mask = average_masks(data_list)

img = join_fmri_img(data_list, mean_mask)

##
data_runs36 = [data[2], data[5]]
data_runs25 = [data[1], data[4]]

##
embeddings = np.load("embeddings_bird.npy", allow_pickle=True)[()]

##
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
        m = re.match(r"docs/cropped/(\d*\..*).png", name)
        return m.groups()[0]

    events["stimulus"] = [get_stim(n) for n in events.stimulus]

    indices = np.array((events.onset / d.tr)).astype(np.int)

    im = ni.masking.apply_mask(d.fmri_img, mask)
    im = detrend(im)

    gamma = im[indices + 2]
    names = events.stimulus
    return gamma, names

##
gamma, names = get_activations(data_list, mean_mask)
#gamma, names = get_activations(data_list[0:1], average_masks(data_list[0:1]))
names_unique = set(names)

##
#beta, beta_names, r2, dm2, d2s, dm_new = get_betas_from_data(join_fmri_img(data_list, mean_mask), data_list, None)

##
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

def getList2(func, pairs):
    acc_over_list = []
    xx = []
    for ii, jj in pairs:
        xx.append(ii)
        acc_list = []
        for i in range(3):
            acc = func(i, ii, jj)
            print(acc)
            acc_list.append(acc)
        acc_over_list.append(acc_list)
    acc_over_list = np.array(acc_over_list)
    return acc_over_list, xx

##

acc_over_list_new, xx = getList2(lambda i, ii, jj:
one_model(gamma, names,
          splitter3(
              filter(repetitions=[1, 3, i], types=[0, 150, 0]),
              filter(repetitions=[0, 1, i], types=[0, 150, 0]),
              filter(repetitions=[1, 3, i], types=[0, 150, 0]),
          ), embeddings, 2000, model=Ridge())
                                  , [[0, 0]])

acc_over_list_new, xx = getList2(lambda i, ii, jj:
one_model(gamma, names,
          splitter3(
              filter(repetitions=[1, 3, i], types=[50, 150, 0]),
              filter(repetitions=[0, 1, i], types=[50, 150, 0]),
              filter(repetitions=[0, 3, 0], types=[0, 50, 0]),
          ), embeddings, 2000, model=Ridge())
                                  , [[0, 0]])

##
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(500, activation="relu"),
    tf.keras.layers.Dense(30, activation="softmax"),
])

def name_to_index(names):
    indices = list(set(names))
    return [indices.index(n) for n in names]


model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

##
def fit_class2(x, y, y2, split, filter_count=1000, model=None, scramble_brain=False):
    y = np.asarray(y)
    y2 = np.asarray(y2)
    (x_train, x_test, x_snr), (y_train, y_test, y_snr) = split(x, y)


    snr = signal_noise_ratio_names(x_snr, y_snr)

    (x_train, x_test, x_snr), (y_train, y_test, y_snr) = split(x, y2)
    order = np.argsort(snr)[::-1]
    filter = order[:filter_count]

    x_train = x_train[:, filter]
    x_test = x_test[:, filter]

    if scramble_brain is True:
        N = x_train.shape[0]
        random_ind = np.arange(N)
        np.random.shuffle(random_ind)
        x_train = x_train[random_ind]

    y_test = tf.keras.utils.to_categorical(np.asarray([indices.index(n) for n in y_test]), len(indices))
    y_train = tf.keras.utils.to_categorical(np.asarray([indices.index(n) for n in y_train]), len(indices))
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    #y_test = y_test[:, :10]
    #y_train = y_train[:, :10]

    clf = model
    clf.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100)

    return model


classes = np.asarray([n.split("_")[0] for n in names])
fit_class2(gamma, names, classes,
          splitter3(
              filter(repetitions=[5, 15, 0], types=[0, 30, 0]),
              filter(repetitions=[0, 5, 0], types=[0, 30, 0]),
              filter(repetitions=[0, 3, 0], types=[0, 30, 0]),
          ),
           splitter3(
               filter(repetitions=[5, 15, 0], types=[0, 30, 0]),
               filter(repetitions=[0, 5, 0], types=[0, 30, 0]),
               filter(repetitions=[0, 3, 0], types=[0, 30, 0]),
           ),
           2000, model=model)