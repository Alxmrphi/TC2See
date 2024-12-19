import os
import sys

dir2 = os.path.abspath('../../')
dir3 = os.path.abspath('../')

if not dir2 in sys.path or not dir3 in sys.path: 
    sys.path.append(dir2)
    sys.path.append(dir3)

import glmsingle
import numpy as np 
import nibabel as nib
import nilearn as nil 
import nilearn.surface as surf
import pandas as pd
import pickle
import sklearn

from glmsingle.glmsingle import GLM_single
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-subj', type=str, help='Subject number', required=True)
args = parser.parse_args()

subj = args.subj
print(f"Processing Subject {subj}'s GLM Data...")

output_path = Path(f"../../results/glm_single/sub_{subj}")
if not os.path.exists(output_path):
    os.makedirs(output_path)

data_path = Path("../../data")
surfaces_path = Path("../../data/processed/fmriprep_surfs")
csv_folder_path = Path("../../data/raw_data/csv_files")

idx_to_fname = pickle.load(open(data_path / 'idx_to_fname.pkl', 'rb'))
fname_to_idx = pickle.load(open(data_path / 'fname_to_idx.pkl', 'rb'))

tr = 1.97
stimdur = 2.0

print(f"\n Subject: {subj}\n")

subj_csv_files = csv_folder_path / Path(f"sub_{subj}/.csv")
csv_file = subj_csv_files / Path(f'TC2See_{subj}_1_result_store.csv') 

results_df = pd.read_csv(csv_file, sep='\t')
events_df = results_df[results_df['stimulus'].str.endswith('png')].copy()

func_folder = surfaces_path / Path(f'sub-{subj}/func')
func_file_lh = Path(f'sub-{subj}_task-bird_run-1_hemi-L_space-fsaverage_bold.func.gii')
func_file_rh = Path(f'sub-{subj}_task-bird_run-1_hemi-R_space-fsaverage_bold.func.gii')
func_path_lh = func_folder / func_file_lh
func_path_rh = func_folder / func_file_rh

def process_stim_column(stimulus):
    assert type(stimulus) == str
    return stimulus.split('.')[1:-1][0]
    
def stimulus_to_class(stimulus2):
    return fname_to_idx[stimulus2]

events_df.loc[:, 'filename'] = events_df['stimulus'].apply(process_stim_column)
events_df.loc[:, 'file_id'] = events_df['filename'].apply(stimulus_to_class)

fmri_lh = surf.load_surf_data(func_path_lh)
fmri_rh = surf.load_surf_data(func_path_lh)
n_conds = len(fname_to_idx) # number of keys in the mapping dictionary
n_vertices_lh, n_timepoints_lh = fmri_lh.shape
n_vertices_rh, n_timepoints_rh = fmri_rh.shape

assert n_vertices_lh == n_vertices_rh, 'Number of vertices in left and right hemispheres do not match'
assert n_timepoints_lh == n_timepoints_rh, 'Number of timepoints in left and right hemispheres do not match'

n_timepoints = n_timepoints_lh
n_vertices = n_vertices_lh

print(f'n_vertices_lh: \n{n_vertices_lh}\n')
print(f'n_timepoints_lh: \n{n_timepoints_lh}\n')
print(f'n_conds: \n{n_conds}\n')

def get_data_and_design_matrices(sub_no, hemi, run_no, n_timepoints):
    """Docstring: TBD """

    hemi = 'L' if hemi == 'lh' else 'R'
    func_file = Path(f'sub-{sub_no}_task-bird_run-{run_no}_hemi-{hemi}_space-fsaverage_bold.func.gii')
    data = surf.load_surf_data(func_folder / func_file)
    file = Path(f'TC2See_{sub_no}_{run_no}_result_store.csv')
    path = subj_csv_files / file
    df = pd.read_csv(path, sep='\t')
    events_df = df[df['stimulus'].str.endswith('png')].copy() 
    events_df.loc[:, 'filename'] = events_df['stimulus'].apply(process_stim_column)
    events_df.loc[:, 'file_id'] = events_df['filename'].apply(stimulus_to_class)
    design = np.zeros((n_timepoints, n_conds))

    for t in range(len(events_df)):
        tr, idx = events_df.iloc[t][['tr', 'file_id']]
        tr = int(tr)
        design[tr, idx] = 1

    return design, data

hemi = 'lh'

design1, data1 = get_data_and_design_matrices(subj, hemi, 1, n_timepoints)
design2, data2 = get_data_and_design_matrices(subj, hemi, 2, n_timepoints)
design3, data3 = get_data_and_design_matrices(subj, hemi, 3, n_timepoints)
design4, data4 = get_data_and_design_matrices(subj, hemi, 4, n_timepoints)
design5, data5 = get_data_and_design_matrices(subj, hemi, 5, n_timepoints)
design6, data6 = get_data_and_design_matrices(subj, hemi, 6, n_timepoints)

data1_ = StandardScaler().fit_transform(data1)
data2_ = StandardScaler().fit_transform(data2)
data3_ = StandardScaler().fit_transform(data3)
data4_ = StandardScaler().fit_transform(data4)
data5_ = StandardScaler().fit_transform(data5)
data6_ = StandardScaler().fit_transform(data6)

data_list = [data1_, data2_, data3_, data4_, data5_, data6_]
design_list = [design1, design2, design3, design4, design5, design6]

split_val = 118

design_list2 = [design_list[0][:split_val,:], design_list[0][split_val:,:], design_list[1][:split_val,:],
                design_list[1][split_val:,:], design_list[2][:split_val,:], design_list[2][split_val:,:],
                design_list[3][:split_val,:], design_list[3][split_val:,:], design_list[4][:split_val,:],
                design_list[4][split_val:,:], design_list[5][:split_val,:], design_list[5][split_val:,:]]

data_list2 = [data_list[0][:,:split_val], data_list[0][:,split_val:], data_list[1][:,:split_val],
                data_list[1][:,split_val:], data_list[2][:,:split_val], data_list[2][:,split_val:],
                data_list[3][:,:split_val], data_list[3][:,split_val:], data_list[4][:,:split_val],
                data_list[4][:,split_val:], data_list[5][:,:split_val], data_list[5][:,split_val:]]

opt = dict()
opt['wantlibrary'] = 1
opt['wantglmdenoise'] = 1
opt['wantfracridge'] = 1

# for the purpose of this example we will keep the relevant outputs in memory
# and also save them to the disk
opt['wantfileoutputs'] = [1,1,1,1]
opt['wantmemoryoutputs'] = [1,1,1,1]

# running python GLMsingle involves creating a GLM_single object
# and then running the procedure using the .fit() routine
glmsingle_obj = GLM_single(opt)
# visualize all the hyperparameters
print(glmsingle_obj.params)

results_glmsingle = glmsingle_obj.fit(
    design_list2,
    data_list2,
    stimdur,
    tr,
    outputdir=str(output_path  / 'left_hem')
)

pickle.dump(results_glmsingle, open(output_path / f'left_hem/sub-{subj}_hemi-{hemi}_glmsingle_results.pkl', 'wb'))

betas = np.squeeze(results_glmsingle['typed']['betasmd']).T
betas = np.clip(betas, a_min=-5, a_max=5)
betas.shape
pickle.dump(betas, open(output_path / f'left_hem/sub-{subj}_hemi-{hemi}_all_betas.pkl', 'wb'))

# consolidate design matrices
designALL = np.concatenate(design_list2,axis=0)
n_trs, n_conds = designALL.shape

print(f"{n_trs = }, {n_conds = }")

# construct a vector containing 0-indexed condition numbers in chronological order
cond_order = []
for p in range(n_trs):
    # some TRs are acquired but have no stimulus presentation in them and are all zero, ignore these
    if np.any(designALL[p]):
        tmp = np.argwhere(designALL[p])[0,0]
        #print(tmp)
        cond_order.append(tmp)
        
cond_order = np.array(cond_order)
pickle.dump(cond_order, open(output_path / f'sub-{subj}_condition_order.pkl', 'wb'))

repTRs = [] # 2 x images containing stimulus trial indices.
conds_seen = []
cond_trs = []

# the first row refers to the first presentation; the second row refers to
# the second presentation.
for cond in range(n_conds): # loop over every condition
    
    TRs = np.argwhere(cond_order==cond)[:,0] # find TRs where this condition was shown

    if len(TRs) < 3:
        continue
    
    # note that for conditions with 3 presentations, we are simply ignoring the third trial
    assert len(TRs) == 3, f"Investigate potential problem here. Number of presentatinos for this condition != 3, TRs = {TRs}" 
    if len(TRs) >= 3:
        selected_TRs = [int(TRs[0]), int(TRs[1]), int(TRs[2])]
        repTRs.append(selected_TRs)
        conds_seen.append(cond)
        cond_trs.append((cond, selected_TRs))
        print(f"Condition {cond}. File = {idx_to_fname[cond]}, at TRs {selected_TRs}")

repTRs = np.array(repTRs)
print(f"{repTRs.shape = }")
repTRs = np.vstack(repTRs).T   
print(f"{repTRs.shape = }")
print(len(conds_seen))

pickle.dump(cond_trs, open(output_path / f'left_hem/sub-{subj}_condition_tr_mapping.pkl', 'wb'))

n_repeated_conds = len(repTRs[0])
n_vertices = betas.shape[1]

print(f"{n_repeated_conds = }, {n_vertices = }")

betas_averaged = np.zeros((n_repeated_conds, n_vertices)) # i.e. (150, 163842) 

for i, (cond, trs) in enumerate(cond_trs):
    betas_averaged[i,:] = np.mean(betas[trs,:], axis=0)
    
pickle.dump(betas_averaged, open(output_path / f'left_hem/sub-{subj}_hemi-{hemi}_betas_averaged_.pkl', 'wb'))

hemi = 'rh'

design1, data1 = get_data_and_design_matrices(subj, hemi, 1, n_timepoints)
design2, data2 = get_data_and_design_matrices(subj, hemi, 2, n_timepoints)
design3, data3 = get_data_and_design_matrices(subj, hemi, 3, n_timepoints)
design4, data4 = get_data_and_design_matrices(subj, hemi, 4, n_timepoints)
design5, data5 = get_data_and_design_matrices(subj, hemi, 5, n_timepoints)
design6, data6 = get_data_and_design_matrices(subj, hemi, 6, n_timepoints)

data1_ = StandardScaler().fit_transform(data1)
data2_ = StandardScaler().fit_transform(data2)
data3_ = StandardScaler().fit_transform(data3)
data4_ = StandardScaler().fit_transform(data4)
data5_ = StandardScaler().fit_transform(data5)
data6_ = StandardScaler().fit_transform(data6)

data_list = [data1_, data2_, data3_, data4_, data5_, data6_]
design_list = [design1, design2, design3, design4, design5, design6]

split_val = 118

design_list2 = [design_list[0][:split_val,:], design_list[0][split_val:,:], design_list[1][:split_val,:],
                design_list[1][split_val:,:], design_list[2][:split_val,:], design_list[2][split_val:,:],
                design_list[3][:split_val,:], design_list[3][split_val:,:], design_list[4][:split_val,:],
                design_list[4][split_val:,:], design_list[5][:split_val,:], design_list[5][split_val:,:]]

data_list2 = [data_list[0][:,:split_val], data_list[0][:,split_val:], data_list[1][:,:split_val],
                data_list[1][:,split_val:], data_list[2][:,:split_val], data_list[2][:,split_val:],
                data_list[3][:,:split_val], data_list[3][:,split_val:], data_list[4][:,:split_val],
                data_list[4][:,split_val:], data_list[5][:,:split_val], data_list[5][:,split_val:]]

opt = dict()
opt['wantlibrary'] = 1
opt['wantglmdenoise'] = 1
opt['wantfracridge'] = 1

# for the purpose of this example we will keep the relevant outputs in memory
# and also save them to the disk
opt['wantfileoutputs'] = [1,1,1,1]
opt['wantmemoryoutputs'] = [1,1,1,1]

# running python GLMsingle involves creating a GLM_single object
# and then running the procedure using the .fit() routine
glmsingle_obj = GLM_single(opt)

# visualize all the hyperparameters
print(glmsingle_obj.params)

results_glmsingle = glmsingle_obj.fit(
    design_list2,
    data_list2,
    stimdur,
    tr,
    outputdir=str(output_path / 'right_hem')
)

pickle.dump(results_glmsingle, open(output_path / f'right_hem/sub-{subj}_hemi-{hemi}_glmsingle_results.pkl', 'wb'))

betas = np.squeeze(results_glmsingle['typed']['betasmd']).T
betas = np.clip(betas, a_min=-5, a_max=5)
betas.shape
pickle.dump(betas, open(output_path / f'right_hem/sub-{subj}_hemi-{hemi}_all_betas.pkl', 'wb'))

n_repeated_conds = len(repTRs[0])
n_vertices = betas.shape[1]

betas_averaged = np.zeros((n_repeated_conds, n_vertices)) # i.e. (150, 163842) 

for i, (cond, trs) in enumerate(cond_trs):
    betas_averaged[i,:] = np.mean(betas[trs,:], axis=0)
    
pickle.dump(betas_averaged, open(output_path / f'right_hem/sub-{subj}_hemi-{hemi}_betas_averaged_.pkl', 'wb'))