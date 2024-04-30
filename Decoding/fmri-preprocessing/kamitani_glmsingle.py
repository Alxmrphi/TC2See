from typing import Sequence
from pathlib import Path
from pprint import pprint

import numpy as np
from bids import BIDSLayout
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from fire import Fire
from glmsingle.glmsingle import GLM_single

from glmsingle_utils import (
    load_results,
    plot_results,
    save_results
)

# There were 15 sessions to acquire training data in this dataset
# These three groups of sessions presented the same images in the same runs
# Each image in the training set was presented exactly 5 times
# It is better to run GLMsingle on the same run from one of these groups so cross validation is possible
KAMITANI_SESSION_GROUPS = {
    'A': [f'perceptionNaturalImageTraining{i:02}' for i in (1, 4, 7, 10, 13)],
    'B': [f'perceptionNaturalImageTraining{i:02}' for i in (2, 5, 8, 11, 14)],
    'C': [f'perceptionNaturalImageTraining{i:02}' for i in (3, 6, 9, 12, 15)],
}
KAMITANI_TR = 2.


def load_data(
        dataset_layout: BIDSLayout,
        subject: str,
        sessions: Sequence[str],
        run_ids: Sequence[int],
        space='T1w',
):
    run_images = []
    events_dfs = []
    for session in sessions:
        for run_id in run_ids:
            bids_image = dataset_layout.get(
                subject=subject,
                session=session,
                space=space,
                run=run_id,
                desc='preproc',
                extension='nii.gz'
            )[0]
            events_files = dataset_layout.get(
                subject=subject,
                session=session,
                run=run_id,
                extension='tsv',
                suffix='events',
            )
            events_df = pd.read_csv(events_files[0].path, sep='\t',)
            events_dfs.append(events_df)
            run_images.append(bids_image.get_image())

    mask_image = dataset_layout.get(
        subject=subject,
        session='perceptionNaturalImageTraining01',
        space=space,
        run=1,
        desc='brain',
        extension='nii.gz'
    )[0].get_image()
    fmri_mask = mask_image.get_fdata().astype(bool)
    H, W, D = fmri_mask.shape

    # Load the fmri data and apply the mask
    fmri_batch = []
    for run_image in tqdm(run_images):
        fmri_data = run_image.get_fdata(dtype=np.float32)
        fmri_data = fmri_data[fmri_mask]
        fmri_batch.append(fmri_data)

    return fmri_batch, events_dfs, fmri_mask


def make_design_matrix(events_dfs: Sequence[DataFrame], num_trs: int):
    conditions = []
    for events_df in events_dfs:
        for i, event in events_df.iterrows():
            if int(event['event_type']) != 1:
                continue
            conditions.append(event['stimulus_id'])
    conditions = list(set(conditions))
    conditions.sort()
    conditions = {condition: i for i, condition in enumerate(conditions)}
    num_conditions = len(conditions)

    design_batch = []
    for run_id, events_df in enumerate(events_dfs):
        design_matrix = np.zeros(shape=(num_trs, num_conditions))

        for i, event in events_df.iterrows():
            if int(event['event_type']) != 1:
                continue
            condition_name = event['stimulus_id']
            c = conditions[condition_name]
            t = round(event['onset'] / KAMITANI_TR)
            design_matrix[t, c] = 1
        design_batch.append(design_matrix)
    return design_batch


def run_all(
        dataset_path: str,
        output_subfolder: str,
        stimulus_duration: float = 8,
):
    dataset_path = Path(dataset_path)
    dataset_layout = BIDSLayout(dataset_path)

    subject = '02'
    run_ids = list(range(1, 9))
    sessions = [f'perceptionNaturalImageTraining{i:02}' for i in range(1, 16)]

    fmri_batch, events_dfs, fmri_mask = load_data(dataset_layout, subject, sessions, run_ids=run_ids)
    design_batch = make_design_matrix(events_dfs, num_trs=fmri_batch[0].shape[-1])

    glmsingle_obj = GLM_single(dict(
        wantlibrary=1,
        wantglmdenoise=1,
        wantfracridge=1,
        wantfileoutputs=[1,1,1,1],
        wantmemoryoutputs=[1,1,1,1],
        sessionindicator=np.arange(1, 16).repeat(8)[None],
        xvalscheme=[list(fold) for fold in np.split(np.arange(1, 121), 5)],
        wanthdf5=1
    ))

    pprint(glmsingle_obj.params)

    output_path = dataset_path / 'glmsingle' / output_subfolder
    results_glmsingle = glmsingle_obj.fit(
        design=design_batch,
        data=fmri_batch,
        stimdur=stimulus_duration,
        tr=KAMITANI_TR,
        outputdir=str(output_path),
    )

    save_results(results_glmsingle, output_path, fmri_mask)


if __name__ == '__main__':
    Fire(run_all)
