import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import h5py
from scipy.stats import pearsonr
import seaborn as sns
import json


ROI_set = "HL"
values_type = "BOLD"
other_info = "bold-occ"
# with open(f"../../results/roi_decoding_accuracies/combined_accuracy_dicts/bold_decoding_accs_occ.json", "r") as file:
#     data = json.load(file)
with open(f"../../results/roi_decoding_accuracies/combined_accuracy_dicts/ROI_accuracies_surfs_Bold.json", "r") as file:
    data = json.load(file)
# with open(f"../../results/roi_decoding_accuracies/combined_accuracy_dicts/glm_decoding_accs_avg_R2.json", "r") as file:
#     data = json.load(file)

subject_strs_dict = {
    'low':      ['06', '15', '20', '21', '23', '28', '29', '32', '39'],
    'moderate': ['05', '07', '10', '14', '17', '18', '19', '22', '25', '30', '31', '33', '34', '37'],
    'high':     ['08', '09', '11', '12', '16', '24', '26', '27', '35', '36', '38']
}
roi_groups = {
    "HL": ["all_rois", "FFC", "VVC", "LO1", "LO2", "LO3", "PHA1", "PHA2", "PHA3", "IPS1", "MT", "LOC1_to_LOC3", "PHA1_to_PHA3"],
    "LL": ["all_rois", "V1", "V2", "V3_V3A_V3B_V3CD", "V4", "V6", "V7", "V8"]
}

bird_quiz_scores = { "05": 0.64, "06": 0.49, "07": 0.61, "08": 0.88, "09": 0.68, "10": 0.58, "11": 0.68, "12": 0.71, "14": 0.63, "15": 0.44, "16": 0.68, "17": 0.63, "18": 0.61, "19": 0.54, "20": 0.47, "21": 0.47, "22": 0.51, "23": 0.47, "24": 0.69, "25": 0.58, "26": 0.73, "27": 0.64, "28": 0.41, "29": 0.37, "30": 0.64, "31": 0.64, "32": 0.37, "33": 0.53, "34": 0.54, "35": 0.68, "36": 0.63, "37": 0.58, "38": 0.75, "39": 0.51, "40": 0.59}
all_roi_decoding_accuracies = []
accuracies_per_expertise = {
    'low': [],
    'moderate': [],
    'high': []
}
roi_regions = None

# loop through subject accuracy dicts
for subject, inner_dict in data.items():
    inner_dict = {roi: val for roi, val in inner_dict.items() if roi in roi_groups[ROI_set]}
    rois = list(inner_dict.keys())
    roi_regions = rois

    y = list(inner_dict.values())
    all_roi_decoding_accuracies.append(y)
    expertise_level = ''
    for expertise_group in subject_strs_dict:
        if subject in subject_strs_dict[expertise_group]:
            expertise_level = expertise_group

    if expertise_level != '':
        accuracies_per_expertise[expertise_level].append(y)
        



with open(f"../../results/roi_decoding_accuracies/combined_accuracy_dicts/bold_decoding_accs_occ.json", "r") as file:
    data2 = json.load(file)

all_roi_decoding_accuracies2 = []
accuracies_per_expertise2 = {
    'low': [],
    'moderate': [],
    'high': []
}
roi_regions = None

# loop through subject accuracy dicts
for subject, inner_dict in data2.items():
    inner_dict = {roi: val for roi, val in inner_dict.items() if roi in roi_groups[ROI_set]}
    rois = list(inner_dict.keys())
    roi_regions = rois

    y = list(inner_dict.values())
    all_roi_decoding_accuracies2.append(y)
    expertise_level = ''
    for expertise_group in subject_strs_dict:
        if subject in subject_strs_dict[expertise_group]:
            expertise_level = expertise_group

    if expertise_level != '':
        accuracies_per_expertise2[expertise_level].append(y)




y_min = -2.5
y_max = 1.5

roi_regions = list(roi_groups[ROI_set])
expertise_levels = ["low", "moderate", "high"]
group_colors = ["coral", "skyblue", "limegreen"]

# Collect average accuracies for each expertise level
grouped_accuracies = {}
for expertise_level in expertise_levels:
    accuracies_array = np.array(accuracies_per_expertise[expertise_level])
    grouped_accuracies[expertise_level] = np.mean(accuracies_array, axis=0)


grouped_accuracies2 = {}
for expertise_level in expertise_levels:
    accuracies_array = np.array(accuracies_per_expertise2[expertise_level])
    grouped_accuracies2[expertise_level] = np.mean(accuracies_array, axis=0)

# Calculate the differences between the two grouped_accuracies
grouped_accuracy_differences = {}
for expertise_level in expertise_levels:
    grouped_accuracy_differences[expertise_level] = (
        grouped_accuracies2[expertise_level] - grouped_accuracies[expertise_level]
    )

# Create grouped bar plot
x = np.arange(len(roi_regions))  # the label locations
width = 0.25  # width of the bars

plt.figure(figsize=(14, 7))
for i, expertise_level in enumerate(expertise_levels):
    plt.bar(x + i * width, grouped_accuracy_differences[expertise_level], width, label=f'{expertise_level.capitalize()} Expertise', color=group_colors[i])

# Customize plot
plt.xlabel('Regions of Interest')
plt.ylabel('Average Accuracy')
plt.title(f'({values_type}) Overall ROI Decoding Accuracy Difference (original - occluded) by Expertise Level')
plt.xticks(x + width, roi_regions, rotation=45, ha='right')  # Center group labels
plt.ylim(bottom=y_min, top=y_max)
plt.legend(title="Expertise Level")
plt.tight_layout()

# Save plot
plt.savefig(f"../../results/roi_decoding_accuracies/{other_info}_{values_type}_grouped_accuracy_differences_{ROI_set}.png")




accuracies_array = np.array(all_roi_decoding_accuracies)
average_accuracies = np.mean(accuracies_array, axis=0)

# # Calculate the minimum accuracy and set the y-axis lower limit slightly below it
# y_min = min(average_accuracies) - 1  # Adjust the offset as needed for better visualization

# # Plot the overall accuracies for ROIs
# plt.figure(figsize=(12, 6))
# plt.bar(roi_regions, average_accuracies, color='coral')
# plt.xlabel('Regions of Interest')
# plt.ylabel('Average Accuracy')
# plt.title('Overall Decoding Accuracies for ROIs')
# plt.xticks(rotation=45, ha='right')
# plt.ylim(bottom=y_min)  # Set the y-axis to start just below the lowest accuracy
# plt.tight_layout()
# plt.savefig(f"/home/jamesmck/projects/def-afyshe-ab/jamesmck/bird_data_analysis/results/roi_decoding_accuracies/all_subs_accuracies_{ROI_set}_{values_type}_.png")


# y_min = 48.5
# y_max = 56.5

# # for expertise_level in accuracies_per_expertise:
# accuracies_array = np.array(accuracies_per_expertise["low"])
# average_accuracies = np.mean(accuracies_array, axis=0)

# plt.figure(figsize=(12, 6))
# plt.bar(roi_regions, average_accuracies, color='coral')
# plt.xlabel('Regions of Interest')
# plt.ylabel('Average Accuracy')
# plt.title(f'({values_type}) Overall ROI Decoding Accuracies for participants with low bird quiz scores')
# plt.xticks(rotation=45, ha='right')
# plt.ylim(bottom=y_min)  # Set the y-axis to start just below the lowest accuracy
# plt.ylim(top=y_max)  # Set the y-axis to start just below the lowest accuracy
# plt.tight_layout()
# plt.savefig(f"../../results/roi_decoding_accuracies/{other_info}_{values_type}_low_score_accuracies_{ROI_set}.png")


# accuracies_array = np.array(accuracies_per_expertise["moderate"])
# average_accuracies = np.mean(accuracies_array, axis=0)

# plt.figure(figsize=(12, 6))
# plt.bar(roi_regions, average_accuracies, color='coral')
# plt.xlabel('Regions of Interest')
# plt.ylabel('Average Accuracy')
# plt.title(f'({values_type}) Overall ROI Decoding Accuracies for participants with moderate bird quiz scores')
# plt.xticks(rotation=45, ha='right')
# plt.ylim(bottom=y_min)  # Set the y-axis to start just below the lowest accuracy
# plt.ylim(top=y_max)  # Set the y-axis to start just below the lowest accuracy
# plt.tight_layout()
# plt.savefig(f"../../results/roi_decoding_accuracies/{other_info}_{values_type}_moderate_score_accuracies_{ROI_set}.png")


# accuracies_array = np.array(accuracies_per_expertise["high"])
# average_accuracies = np.mean(accuracies_array, axis=0)

# plt.figure(figsize=(12, 6))
# plt.bar(roi_regions, average_accuracies, color='coral')
# plt.xlabel('Regions of Interest')
# plt.ylabel('Average Accuracy')
# plt.title(f'({values_type}) Overall ROI Decoding Accuracies for participants with high bird quiz scores')
# plt.xticks(rotation=45, ha='right')
# plt.ylim(bottom=y_min)  # Set the y-axis to start just below the lowest accuracy
# plt.ylim(top=y_max)  # Set the y-axis to start just below the lowest accuracy
# plt.tight_layout()
# plt.savefig(f"../../results/roi_decoding_accuracies/{other_info}_{values_type}_high_score_accuracies_{ROI_set}.png")