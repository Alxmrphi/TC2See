import json 
import os
import json
from pathlib import Path

def combine_json_files(folder_path):
    combined_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'): 
            file_path = folder_path / filename
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                combined_data.update(data)

    return combined_data

individual_acs_dir = Path('../../results/roi_decoding_accuracies/individual_accuracy_dicts/bold_surfs_occ')


combined_avg = combine_json_files(individual_acs_dir)
combined_avg = {key: combined_avg[key] for key in sorted(combined_avg.keys())}

output_dir = Path('../../results/roi_decoding_accuracies/combined_accuracy_dicts')

with open(output_dir / 'bold_decoding_accs_occ.json', 'w') as sorted_avg_file:
    json.dump(combined_avg, sorted_avg_file, indent=4)
