from __future__ import print_function 
import subprocess
from subprocess import Popen, PIPE

excluded_subs = [13]
subs = [num for num in range(33, 35) if num not in excluded_subs]
participants = [ '0' + str(num) if num < 10 else str(num) for num in subs]

def execute(cmd):
    print(subprocess.PIPE)
    print(cmd)
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

for i in participants:
    command = f'docker run --rm -v E:/fmri_processing/results/TC2See:/data:ro -v E:/fmri_processing/results:/out -v E:/fmri_processing/scripts:/license -e FS_LICENSE=/license/license.txt -e FREESURFER_HOME=/opt/freesurfer nipreps/fmriprep /data /out/derivatives_TC2See_new participant --participant-label {i} -w /out/derivatives_TC2See_new_work --output-spaces fsaverage'
    for path in execute(command.split(' ')):
        print(path, end="")
