from typing import Dict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib


def load_results(results_path: str) -> Dict:
    results_glmsingle = {}
    for file_path in Path(results_path).iterdir():
        if file_path.name == 'TYPEA_ONOFF.npy':
            results_glmsingle['typea'] = np.load(file_path, allow_pickle=True).item()
        elif file_path.name == 'TYPEB_FITHRF.npy':
            results_glmsingle['typeb'] = np.load(file_path, allow_pickle=True).item()
        elif file_path.name == 'TYPEC_FITHRF_GLMDENOISE.npy':
            results_glmsingle['typec'] = np.load(file_path, allow_pickle=True).item()
        elif file_path.name == 'TYPED_FITHRF_GLMDENOISE_RR.npy':
            results_glmsingle['typed'] = np.load(file_path, allow_pickle=True).item()
    return results_glmsingle


def plot_results(results_glmsingle: Dict, fmri_mask: np.ndarray, d_layer: int):
    plt.figure(figsize=(12, 8))
    final_results = results_glmsingle['typed']

    def plot_result(i, field, cmap, clim):
        plt.subplot(2, 2, i)
        x = final_results[field]

        if len(x.shape) == 2:
            x = np.nanmean(x, axis=-1)

        volume = np.zeros_like(fmri_mask, dtype=np.float32)
        volume[fmri_mask] = x

        plt.imshow(volume[:, :, d_layer], cmap=cmap, clim=clim)

        plt.colorbar()
        plt.title(field)
        plt.axis(False)

    plot_result(1, 'betasmd', 'RdBu_r', [-5, 5])
    plot_result(2, 'R2', 'hot', [0, 55])
    plot_result(3, 'HRFindex', 'jet', [0, 20])
    plot_result(4, 'FRACvalue', 'copper', [0, 1])


def save_results(
        results_glmsingle: Dict,
        output_path: str,
        fmri_mask: np.ndarray,
):
    """
    Saves glmsingle results as nifti files to the output path (typically where you saved the run)
    This function isn't totally complete and doesn't save all results, and some are saved incorrectly.
    """
    type_names = {
        'typea': 'onoff',
        'typeb': 'fithrf',
        'typec': 'fithrf_GLMdenoise',
        'typed': 'fithrf_GLMdenoise_RR'
    }

    num_voxels = fmri_mask.sum()
    for type_name, results in results_glmsingle.items():
        result_name = type_names[type_name]
        result_path = Path(output_path) / result_name
        result_path.mkdir(exist_ok=True, parents=True)

        for k, v in results.items():
            if isinstance(v, np.ndarray):
                print(k, v.dtype, v.shape)
                if v.shape[0] == num_voxels:
                    if len(v.shape) == 1:
                        out_img = np.zeros_like(fmri_mask, dtype=v.dtype if v.dtype != bool else int)
                        out_img[fmri_mask] = v
                        img = nib.Nifti1Image(out_img, np.eye(4))
                        nib.save(img, result_path / f'{result_name}_{k}.nii.gz')
            else:
                print(k, type(v))
