import os
import glob
import sys
from pathlib import Path
import shutil
import re
import pandas as pd
import numpy as np
import shutil

sys.path.append("E:/fmri_processing")

from fmri_processing.preprocessing2 import convert_dicom, copy_nii_to_project, process_tsv_files

tmp_folder = "E:/fmri_processing/results/__tmp3__"
output_folder = "E:/fmri_processing/results"

project = "TC2See"
participants = ["35"]

for participant in participants:
    sub = participant

    base_folder = f"E:/fmri_processing/data/{project}_{sub}"
    recordings = base_folder + "/study"
    regex_runs = r".*Run(\d)_(.*)_\d*"
    regex_anat = r"t1_mprage_sag_p2_iso1.0(?:_\d*)?"

    # regex_tsv = r".*/sub-(?:.*\D)?(\d+)(?:.*\D)?_task-(.*)_run-(\d)_events.[ct]sv"
    # regex_tsv = rf".*/(?:.*\D)?(\d+)(?:.*\D)?_()(\d)_result_store.csv"
    regex_tsv = rf".*{re.escape(os.sep)}(?:.*\D)?(\d+)(?:.*\D)?_()(\d)_result_store.csv"

    TR = 2
    throw_away_trs = 5

    # convert the dicom to niifty
    convert_dicom(base_folder, tmp_folder, regex_runs, regex_anat, throw_away_trs=throw_away_trs)
    # copy and rename the files
    copy_nii_to_project(tmp_folder, project, sub, output_folder)
    # copy and strip throw-away trs from the tsv files
    process_tsv_files(base_folder, project, regex_tsv, output_folder, TR, throw_away_trs=throw_away_trs)
