import numpy as np
import json 
import matplotlib.pyplot as plt
import pandas as pd

 
bird_quiz_scores = { "05": 0.64, "06": 0.49, "07": 0.61, "08": 0.88, "09": 0.68, "10": 0.58, "11": 0.68, "12": 0.71, "14": 0.63, "15": 0.44, "16": 0.68, "17": 0.63, "18": 0.61, "19": 0.54, "20": 0.47, "21": 0.47, "22": 0.51, "23": 0.47, "24": 0.69, "25": 0.58, "26": 0.73, "27": 0.64, "28": 0.41, "29": 0.37, "30": 0.64, "31": 0.64, "32": 0.37, "33": 0.53, "34": 0.54, "35": 0.68, "36": 0.63, "37": 0.58, "38": 0.75, "39": 0.51, "40": 0.59}
subject_strs_dict = {
    'low':      ['low', ['06', '15', '20', '21', '23', '28', '29', '32', '39']],
    'moderate': ['moderate', ['05', '07', '10', '14', '17', '18', '19', '22', '25', '30', '31', '33', '34', '37']],
    'high':     ['high', ['08', '09', '11', '12', '16', '24', '26', '27', '35', '36', '38']]
}
roi_groups = {
    "hl": ["all_rois", "FFC", "VVC", "LO1", "LO2", "LO3", "PHA1", "PHA2", "PHA3", "IPS1", "MT", "LOC1_to_LOC3", "PHA1_to_PHA3"],
    "ll": ["all_rois", "V1", "V2", "V3_V3A_V3B_V3CD", "V4", "V6", "V7", "V8"]
}

ROI_set = "hl"
values_type = "BOLD"
# other_info = "occ"

with open(f"../../results/roi_decoding_accuracies/combined_accuracy_dicts/ROI_accuracies_surfs_Bold.json", "r") as file:
    data = json.load(file)
with open(f"../../results/roi_decoding_accuracies/combined_accuracy_dicts/bold_decoding_accs_occ.json", "r") as file:
    data2 = json.load(file)
with open(f"../../results/roi_decoding_accuracies/combined_accuracy_dicts/glm_decoding_accs_avg_R2.json", "r") as file:
    data3 = json.load(file)

correlation_dict = {}
for roi in data["05"].keys():
    if roi in roi_groups[ROI_set]:
        correlation_dict[roi] = {}
        correlation_dict[roi]["bird_scores"] = [value for key, value in bird_quiz_scores.items() if key in data.keys()]
        correlation_dict[roi]["ROI_accs"] = []
        
        for key, roi_dict in data.items():
            if key != "13":
                correlation_dict[roi]["ROI_accs"].append(roi_dict[roi])

correlations = {}
for roi, values in correlation_dict.items():
    bird_scores = np.array(values["bird_scores"])
    roi_accs = np.array(values["ROI_accs"])
    corr = np.corrcoef(bird_scores, roi_accs)[0, 1]
    correlations[roi] = corr


correlation_dict2 = {}
for roi in data2["05"].keys():
    if roi in roi_groups[ROI_set]:
        correlation_dict2[roi] = {}
        correlation_dict2[roi]["bird_scores"] = [value for key, value in bird_quiz_scores.items() if key in data2.keys()]
        correlation_dict2[roi]["ROI_accs"] = []
        
        for key, roi_dict in data2.items():
            if key != "13":
                correlation_dict2[roi]["ROI_accs"].append(roi_dict[roi])

correlations2 = {}
for roi, values in correlation_dict2.items():
    bird_scores = np.array(values["bird_scores"])
    roi_accs = np.array(values["ROI_accs"])
    corr = np.corrcoef(bird_scores, roi_accs)[0, 1]
    correlations2[roi] = corr


correlation_diff = {}
for roi, corr in correlations.items():
    correlation_diff[roi] = corr - correlations2[roi]

y_max = 0.25
y_min = -0.3
plt.figure(figsize=(8, 5))
plt.bar(correlation_diff.keys(), correlation_diff.values(), color="skyblue")
plt.title(f"({values_type}) Correlation Difference (original - occ) for Each ROI")
plt.ylabel("Correlation Coefficient")
plt.xlabel("ROI")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.xticks(rotation=90)  
plt.ylim(bottom=y_min, top=y_max)
plt.tight_layout()
plt.savefig(f"/home/jamesmck/projects/def-afyshe-ab/jamesmck/bird_data_analysis/results/roi_accuracy_corrs/diff_{ROI_set}_{values_type}_roi_correlation_bar_plot_dec19.png")
