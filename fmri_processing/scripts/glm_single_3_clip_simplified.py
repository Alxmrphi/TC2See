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
data_runs36 = [data[2], data[5]]
data_runs25 = [data[1], data[4]]

##
embeddings = np.load("embeddings.npy", allow_pickle=True)[()]

##
gamma_1, names_1 = get_activations(data_runs25, mean_mask)
names_1_unique = set(names_1)
gamma_2, names_2 = get_activations(data_runs36, mean_mask)
names_2_unique = set(names_2)

##
beta_1, beta_names_1, r2, dm2, d2s, dm_new = get_betas_from_data(join_fmri_img(data_runs25, mean_mask), data_runs25, None)
beta_2, beta_names_2, r2, dm, d2s, dm_new = get_betas_from_data(join_fmri_img(data_runs36, mean_mask), data_runs36, None)

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
        for i in range(5):
            acc = func(i, ii, jj)
            print(acc)
            acc_list.append(acc)
        acc_over_list.append(acc_list)
    acc_over_list = np.array(acc_over_list)
    return acc_over_list, xx

##
ax = plt.subplot(221)
plt.title("run 2 run 5 (0.5s presentation)")
for reps in [8]:#[2, 3, 4, 5, 6, 7, 8, 9, 10]:
    pairs = [[reps, 2], [reps, 3], [reps, 4], [reps, 5], [reps, 6],
             [reps, 7], [reps, 8], [reps, 9], [reps, 10], [reps, 11], [reps, 12]]

    acc_over_list_new, xx = getList2(lambda i, ii, jj: one_model(gamma_1, names_1,
                                                      splitter3(
                                                          filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                          filter(repetitions=[0, 2, i*2]),
                                                          filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                      ), embeddings, 5000, model=Ridge()), pairs)
    plt.errorbar([p[0]*p[1] for p in pairs], np.mean(acc_over_list_new, axis=1), np.std(acc_over_list_new, axis=1), label="gamma")

plt.subplot(222, sharex=ax, sharey=ax)
plt.title("run 3 run 6 (5.5s presentation)")
for reps in [8]:#[2, 3, 4, 5, 6, 7, 8, 9, 10]:
    pairs = [[reps, 2], [reps, 3], [reps, 4], [reps, 5], [reps, 6],
             [reps, 7], [reps, 8], [reps, 9], [reps, 10], [reps, 11], [reps, 12]]

    acc_over_list_new, xx = getList2(lambda i, ii, jj: one_model(gamma_2, names_2,
                                                      splitter3(
                                                          filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                          filter(repetitions=[0, 2, i*2]),
                                                          filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                      ), embeddings, 5000, model=Ridge()), pairs)
    plt.errorbar([p[0]*p[1] for p in pairs], np.mean(acc_over_list_new, axis=1), np.std(acc_over_list_new, axis=1), label="gamma")

##
plt.subplot(221)
for reps in [8]:#[2, 3, 4, 5, 6, 7, 8, 9, 10]:
    pairs = [[reps, 2], [reps, 3], [reps, 4], [reps, 5], [reps, 6],
             [reps, 7], [reps, 8], [reps, 9], [reps, 10], [reps, 11], [reps, 12]]

    acc_over_list_new, xx = getList2(lambda i, ii, jj: one_model(beta_1, beta_names_1,
                                                      splitter3(
                                                          filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                          filter(repetitions=[0, 2, i*2]),
                                                          filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                      ), embeddings, 5000, model=Ridge()), pairs)
    plt.errorbar([p[0]*p[1] for p in pairs], np.mean(acc_over_list_new, axis=1), np.std(acc_over_list_new, axis=1), label="beta")

plt.subplot(222, sharex=ax, sharey=ax)
for reps in [8]:#[2, 3, 4, 5, 6, 7, 8, 9, 10]:
    pairs = [[reps, 2], [reps, 3], [reps, 4], [reps, 5], [reps, 6],
             [reps, 7], [reps, 8], [reps, 9], [reps, 10], [reps, 11], [reps, 12]]

    acc_over_list_new, xx = getList2(lambda i, ii, jj: one_model(beta_2, beta_names_2,
                                                      splitter3(
                                                          filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                          filter(repetitions=[0, 2, i*2]),
                                                          filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                      ), embeddings, 5000, model=Ridge()), pairs)
    plt.errorbar([p[0]*p[1] for p in pairs], np.mean(acc_over_list_new, axis=1), np.std(acc_over_list_new, axis=1), label="beta")

plt.legend()

##
#----------------------------------------------
##
plt.subplot(223, sharex=ax, sharey=ax)
plt.title("run 2 run 5 (0.5s presentation)")
for ims in [12]:#[2, 3, 4, 5, 6, 7, 8, 9, 10]:
    pairs = [[2, ims], [3, ims], [4, ims], [5, ims], [6, ims],
             [7, ims], [8, ims]]

    acc_over_list_new, xx = getList2(lambda i, ii, jj: one_model(gamma_1, names_1,
                                                      splitter3(
                                                          filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                          filter(repetitions=[0, 2, i*2]),
                                                          filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                      ), embeddings, 5000, model=Ridge()), pairs)
    plt.errorbar([p[0]*p[1] for p in pairs], np.mean(acc_over_list_new, axis=1), np.std(acc_over_list_new, axis=1), label="gamma")

plt.subplot(224, sharex=ax, sharey=ax)
plt.title("run 3 run 6 (5.5s presentation)")
for ims in [12]:#[2, 3, 4, 5, 6, 7, 8, 9, 10]:
    pairs = [[2, ims], [3, ims], [4, ims], [5, ims], [6, ims],
             [7, ims], [8, ims]]

    acc_over_list_new, xx = getList2(lambda i, ii, jj: one_model(gamma_2, names_2,
                                                      splitter3(
                                                          filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                          filter(repetitions=[0, 2, i*2]),
                                                          filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                      ), embeddings, 5000, model=Ridge()), pairs)
    plt.errorbar([p[0]*p[1] for p in pairs], np.mean(acc_over_list_new, axis=1), np.std(acc_over_list_new, axis=1), label="gamma")

plt.subplot(223)
for ims in [12]:#[2, 3, 4, 5, 6, 7, 8, 9, 10]:
    pairs = [[2, ims], [3, ims], [4, ims], [5, ims], [6, ims],
             [7, ims], [8, ims]]

    acc_over_list_new, xx = getList2(lambda i, ii, jj: one_model(beta_1, beta_names_1,
                                                      splitter3(
                                                          filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                          filter(repetitions=[0, 2, i*2]),
                                                          filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                      ), embeddings, 5000, model=Ridge()), pairs)
    plt.errorbar([p[0]*p[1] for p in pairs], np.mean(acc_over_list_new, axis=1), np.std(acc_over_list_new, axis=1), label="beta")

plt.subplot(224)
for ims in [12]:#[2, 3, 4, 5, 6, 7, 8, 9, 10]:
    pairs = [[2, ims], [3, ims], [4, ims], [5, ims], [6, ims],
             [7, ims], [8, ims]]

    acc_over_list_new, xx = getList2(lambda i, ii, jj: one_model(beta_2, beta_names_2,
                                                      splitter3(
                                                          filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                          filter(repetitions=[0, 2, i*2]),
                                                          filter(repetitions=[2, 2+int(ii), i*2], types=[0, jj, 0]),
                                                      ), embeddings, 5000, model=Ridge()), pairs)
    plt.errorbar([p[0]*p[1] for p in pairs], np.mean(acc_over_list_new, axis=1), np.std(acc_over_list_new, axis=1), label="beta")


##
plt.errorbar(range(1, 10), np.mean(acc_over_list, axis=1), np.std(acc_over_list, axis=1), label="clip")
plt.errorbar(range(1, 10), np.mean(acc_over_list2, axis=1), np.std(acc_over_list2, axis=1), label="scr clip")
plt.errorbar(range(1, 10), np.mean(acc_over_list3, axis=1), np.std(acc_over_list3, axis=1), label="scr brain")
