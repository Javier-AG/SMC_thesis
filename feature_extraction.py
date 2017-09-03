import os
import sys
import numpy as np
import subprocess

# FEATURE EXTRACTION using Essentia's Out-of-box Freesound Extractor

# Selection of samples. Change folder and run again!
folder = 'FOLDER_NAME'
pardir = 'PARENT_DIRECTORY_PATH' + folder + '/'
filenames = os.listdir(pardir)
samples = np.array([track for track in filenames if (track[-3:] == 'wav')])

# Create yaml file per each sample. It can be slow for many samples...
for track in samples:
    subprocess.Popen(['./streaming_extractor_freesound ' + os.path.join(pardir,track) + ' ' + track[:-4]], shell=True)
print "++++++++++++ EXTRACTION FINISHED ++++++++++++++"
