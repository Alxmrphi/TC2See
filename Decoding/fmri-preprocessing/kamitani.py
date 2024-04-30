from typing import Sequence

import h5py
import numpy as np
from tqdm import tqdm
from einops import rearrange


def load_data(
        path: str,
        subject: str,
        tr_offset: int,
        run_normalize: str,
        session_normalize: bool,
):
    with h5py.File(path, 'r') as f:
        group = f[subject]
        stimulus_ids = group['stimulus_ids'][:]
        stimulus_trs = group['stimulus_trs'][:]

        num_runs = stimulus_trs.shape[0]
        num_trs = group['bold'].shape[1]

        bold = []
        ids = []
        for i in range(num_runs):
            run_trs = stimulus_trs[i] + tr_offset
            run_ids = stimulus_ids[i]

            in_range = run_trs < num_trs
            run_trs = run_trs[in_range]
            run_ids = run_ids[in_range]

            run_bold = group['bold'][i, run_trs]
            if run_normalize == 'zscore':
                run_bold = (run_bold - group['bold_mean'][i]) / group['bold_std'][i]
            elif run_normalize == 'linear_trend':
                trend_coeffs = np.stack([run_trs, np.ones_like(run_trs)], axis=1).astype(float)
                predicted_bold = trend_coeffs @ group['bold_trend'][i]
                run_bold = (run_bold - predicted_bold) / group['bold_trend_std'][i]
            bold.append(run_bold.astype(np.float32))
            ids.append(run_ids)
        bold = np.concatenate(bold)
        ids = np.concatenate(ids)

        if session_normalize:
            bold = rearrange(bold, '(s b) v -> s b v', s=15)
            bold = (bold - bold.mean(axis=1, keepdims=True)) / bold.std(axis=1, keepdims=True)
            bold = rearrange(bold, 's b v -> (s b) v', s=15)
        else:
            bold = (bold - bold.mean(axis=0)) / bold.std(axis=0)

        mask = group['fmri_mask'][:]

        return bold, ids, mask, group['affine'][:]


def repetition_shape(stimulus_ids, n):
    unique_ids, unique_counts = np.unique(stimulus_ids, return_counts=True)
    atleast_n_ids = unique_ids[unique_counts >= n]
    repetition_ids = np.stack([
        np.where(stimulus_ids == i)[0][:n]
        for i in atleast_n_ids
    ])
    return repetition_ids


def convert_ids(stimulus_ids: Sequence[str], all_stimulus_ids: Sequence[str]) -> Sequence[int]:
    stimulus_id_map = {stim_id: i for i, stim_id in enumerate(all_stimulus_ids)}
    out = []
    for stim_id in stimulus_ids:
        class_id, image_id = str(stim_id).split('.')
        while len(image_id) < 6:
            image_id = image_id + '0'
        stim_id = f'{class_id}.{image_id}'
        out.append(stimulus_id_map[stim_id])
    return out
