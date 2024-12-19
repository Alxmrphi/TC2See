import numpy as np
import pandas as pd
import rsatoolbox
from rsatoolbox import vis
from rsatoolbox.rdm import RDMs
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu
from scipy.stats import wasserstein_distance 
from scipy.spatial.distance import squareform, pdist
from statsmodels.stats.multitest import multipletests
import h5py

dataset_root = Path('../../data')
results_dir = Path('../../results')

expertise_groups   = ['low', 'high']
avg_duplicates = True
rdm_dist = "correlation"

subject_strs_dict = {
    'low':      ['low', ['06', '15', '20', '21', '23', '28', '29', '32', '39']],
    'moderate': ['moderate', ['05', '07', '10', '14', '17', '18', '19', '22', '25', '30', '31', '33', '34', '37']],
    'high':     ['high', ['08', '09', '11', '12', '16', '24', '26', '27', '35', '36', '38']]
}

ROIs = ["FFC", "VVC", "PHA1", "PHA2", "PHA3", "LO1", "LO2", "LO3", "IPS1", "MT"]
RDM_dict = {}

for group in expertise_groups:
    RDM_dict[group] = {}
    expertise_level_str = subject_strs_dict[group][0]
    subject_strs        = subject_strs_dict[group][1]

    for ROI in ROIs:
        RDM_dict[group][ROI] = {}

        for subject_str in subject_strs:
            roi_path = dataset_root / f"processed/roi_representations/sub_{subject_str}" / ROI
            rdm_path = dataset_root / f"processed/RDMs/{rdm_dist}/sub_{subject_str}" / ROI
            rdm_file_path = rdm_path / f'rdm_for_{ROI}.hdf5'

            if not rdm_path.exists():
                rdm_path.mkdir(parents=True, exist_ok=True)

                sub_roi_data = pd.read_parquet(roi_path / f'reps_for_{ROI}.parquet')
                
                if avg_duplicates:
                    sub_roi_data['stimulus_category'] = sub_roi_data['stimulus_category'].apply(lambda x: 1 if x == "Sparrow" else 2)
                    sub_roi_data = sub_roi_data.groupby('stimulus_ids').mean().reset_index()
                    sub_roi_data['stimulus_category'] = sub_roi_data['stimulus_category'].apply(lambda x: "Sparrow" if x == 1 else "Warbler")

                stim_category = list(sub_roi_data['stimulus_category'])
                stim_ids = list(sub_roi_data['stimulus_ids'])
                sub_roi_data = sub_roi_data.drop(columns=['stimulus_ids', 'stimulus_category'])
                rdm = None

                if rdm_dist == "wasserstein":
                    distances = pdist(sub_roi_data.to_numpy(), metric=wasserstein_distance) # Convert pairwise distances to a square matrix rdm

                    rdm = RDMs( 
                        dissimilarities=distances, 
                        rdm_descriptors={'name': ['wasserstein_rdm']}, 
                        pattern_descriptors={'stim_ids': stim_ids} 
                    )

                    rdm_np = squareform(distances)
                    rdm_df = pd.DataFrame(rdm_np, index=stim_ids, columns=stim_ids)
                    vmin = rdm_np.min() 
                    vmax = rdm_np.max()
                    
                
                elif rdm_dist == "correlation":
                    data = rsatoolbox.data.Dataset(
                        sub_roi_data.to_numpy(),
                        obs_descriptors={'stim_category': stim_category, 'stim_ids': stim_ids}
                    )

                    rdm = rsatoolbox.rdm.calc_rdm(data, method='correlation')
                    rdm_matrix = rdm.get_matrices()[0] 
                    rdm_df = pd.DataFrame(rdm_matrix, index=data.obs_descriptors["stim_ids"], columns=data.obs_descriptors["stim_ids"])
                    
                    vmin = 0.4
                    vmax = 1.6

                print(f"Saving RDM for subject {subject_str}, ROI: {ROI}")
                RDM_dict[group][ROI][subject_str] = rdm
                rdm.save(rdm_file_path, file_type='hdf5', overwrite=True)

                plt.figure(figsize=(10, 8))
                sns.heatmap(rdm_df, cmap='coolwarm', center=1, vmin=vmin, vmax=vmax, square=True, cbar_kws={'label': 'Dissimilarity'})
                plt.title(f'RDM for {ROI}')
                plt.xlabel('Stimulus ID')
                plt.ylabel('Stimulus ID')
                plt.savefig(results_dir / f'rdm_plots/{rdm_dist}/{rdm_dist}_rdm_{subject_str}_{ROI}.png', dpi=300)
                plt.close()

            else:
                rdm = rsatoolbox.rdm.rdms.load_rdm(rdm_file_path, file_type='hdf5')
                RDM_dict[group][ROI][subject_str] = rdm


# Means of all RDM correlations for each ROI and each expertise level
group_correlation_means = {} 
group_correlation_stds = {}
group_correlation_arrays = {}


for group in expertise_groups:
    group_correlation_arrays[group] = {}
    group_correlation_means[group] = {}
    group_correlation_stds[group] = {}
    subject_strs = subject_strs_dict[group][1]
    
    for ROI in ROIs:
        # RDMs for all subjects in the current group and ROI
        rdms_list = [RDM_dict[group][ROI][subject_str] for subject_str in subject_strs]
        correlations = []

        # Compute pairwise correlations between RDMs
        for i in range(len(rdms_list)):
            for j in range(i + 1, len(rdms_list)):
                rdm1 = rdms_list[i]
                rdm2 = rdms_list[j]
                
                # Compare the two RDMs using correlation
                corr_value = rsatoolbox.rdm.compare(rdm1, rdm2, method='corr')
                correlations.append(corr_value)
        
        # Correlation vector containing the correlation of each RDM to each other RDM for this ROI and expertise level
        group_correlation_arrays[group][ROI] = correlations

        # Calculate the mean of all RDM correlations for this ROI and expertise level
        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        
        group_correlation_means[group][ROI] = mean_corr
        group_correlation_stds[group][ROI] = std_corr


data = []
for (group, rois), (group2, rois2) in zip(group_correlation_means.items(), group_correlation_stds.items()):
    for (roi, mean_corr), (roi2, std_corr) in zip(rois.items(), rois2.items()):
        data.append({
            'Group': group,
            'ROI': roi,
            'Mean_Correlation': mean_corr,
            'STD_Correlation': std_corr
        })

df = pd.DataFrame(data)

p_values_results = {}

for ROI in ROIs:
    # Get the correlation values for high and low groups
    high_group_correlations = group_correlation_arrays['high'][ROI]
    low_group_correlations = group_correlation_arrays['low'][ROI]

    # Mann-Whitney U test
    u_statistic, p_value = mannwhitneyu(low_group_correlations, high_group_correlations, alternative='two-sided')
    p_values_results[ROI] = p_value.item()


p_values_df = pd.DataFrame(list(p_values_results.items()), columns=['ROI', 'p_value'])
print(f"\n{p_values_df}\n")

p_values = list(p_values_results.values())

adjusted_p_values = multipletests(p_values, method='fdr_bh')[1] # fdr_bh = Benjamini–Hochberg procedure
adjusted_p_values_results = {roi: adjusted_p_value for roi, adjusted_p_value in zip(p_values_results.keys(), adjusted_p_values)}
adjusted_p_values_df = pd.DataFrame(list(adjusted_p_values_results.items()), columns=['ROI', 'Adjusted_p_value'])

print(f"\n{adjusted_p_values_df}\n")

mean_pivot = df.pivot(index='ROI', columns='Group', values='Mean_Correlation').reindex(ROIs)
std_pivot = df.pivot(index='ROI', columns='Group', values='STD_Correlation').reindex(ROIs)

# Create the bar plot without error bars
ax = mean_pivot.plot(kind='bar', figsize=(12, 6))

# Add labels and titles
plt.ylabel('Mean RDM Correlation')
plt.title('Mean RDM Correlations for Each ROI by Expertise Group')
plt.xticks(rotation=90)
ax.set_ylim(-0.01, 0.025)

# Loop through each bar and add text annotations for the standard deviation
for i, bar in enumerate(ax.patches): 
    height = bar.get_height()  
    
    # Calculate the corresponding ROI and Group from the bar's position
    row = i % len(mean_pivot.index)  
    col = i // len(mean_pivot.index) 
    
    roi = mean_pivot.index[row]  
    group = mean_pivot.columns[col] 
    std_value = std_pivot.iloc[row, col] 

    # Annotation with standard deviation value at the top of each bar
    ax.annotate(f'{std_value:.2f}', 
                (bar.get_x() + bar.get_width() / 2, height),  # Place in the middle of the bar
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(results_dir / f'{rdm_dist}_barplot_correlations_with_std_annotations.png')


# Bar plot of adjusted p-values for each ROI
plt.figure(figsize=(12, 6))
sns.barplot(x='ROI', y='Adjusted_p_value', data=adjusted_p_values_df)
plt.axhline(y=0.05, color='red', linestyle='--', label='Significance Threshold (0.05)')
plt.title('P-values (Adjusted) for Each ROI')
plt.xticks(rotation=90)
plt.ylabel('Adjusted P-value')
plt.xlabel('ROI')
plt.legend()
plt.tight_layout()
plt.savefig(results_dir / f'{rdm_dist}_roi_mann_whitney_test_adjusted_pvalues.png')

# Bar plot of adjusted p-values for each ROI
plt.figure(figsize=(12, 6))
sns.barplot(x='ROI', y='p_value', data=p_values_df)
plt.axhline(y=0.05, color='red', linestyle='--', label='Significance Threshold (0.05)')
plt.title('P-values for Each ROI')
plt.xticks(rotation=90)
plt.ylabel('P-value')
plt.xlabel('ROI')
plt.legend()
plt.tight_layout()
plt.savefig(results_dir / f'{rdm_dist}_roi_mann_whitney_test_pvalues.png')


# Function to plot histograms and boxplots for a given ROI
def plot_distribution_and_ranks(group_correlation_arrays, ROI):
    high_group_correlations = np.array(group_correlation_arrays['high'][ROI]).flatten()
    low_group_correlations = np.array(group_correlation_arrays['low'][ROI]).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram of RDM correlations for high and low groups
    sns.histplot(high_group_correlations, ax=axes[0], color='blue', label='High Expertise', bins=10)
    sns.histplot(low_group_correlations, ax=axes[0], color='green', label='Low Expertise', bins=10)
    axes[0].set_title(f'Distribution of RDM Correlations - {ROI}')
    axes[0].set_xlabel('RDM Correlation')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].set_xlim(0, 0.5)

    # Boxplot to show the rank distribution and potential overlap
    combined_data = [high_group_correlations, low_group_correlations]
    sns.boxplot(data=combined_data, ax=axes[1], palette=['blue', 'green'])

    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['High Expertise', 'Low Expertise'])
    axes[1].set_title(f'Boxplot of RDM Correlations - {ROI}')
    axes[1].set_ylabel('RDM Correlation')
    axes[1].set_ylim(-0.05, 0.5)

    plt.tight_layout()
    plt.savefig(results_dir / f'{rdm_dist}_box_and_hist_of_{ROI}_high_v_low.png')
