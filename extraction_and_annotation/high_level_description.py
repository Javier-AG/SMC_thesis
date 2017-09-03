
# coding: utf-8

# In[2]:

import Timbral_Brightness as bright
import Timbral_Depth as depth
import Timbral_Hardness as hard
import Timbral_Roughness as rough
import os
import numpy as np
import pandas as pd


# High-level descriptors calculation

# In[29]:

# Set folders: change source directory 
pardir = 'DATASET_PATH'
folder = 'FOLDER_NAME'
# Initialize arrays
b = []
d = []
h = []
r = []
# Timbral models
tracks = np.array([track for track in os.listdir(os.path.join(pardir,folder)) if track[-3:] == 'wav'])
for track in tracks: 
    b.append(bright.timbral_brightness(os.path.join(pardir,folder,track)))
    d.append(depth.timbral_depth(os.path.join(pardir,folder,track)))
    h.append(hard.timbral_hardness(os.path.join(pardir,folder,track)))    
    r.append(rough.timbral_roughness(os.path.join(pardir,folder,track)))
# Normalization and rearrange (0 to 100)
# Brightness
b1 = b - min(b)
b_norm = (b1 / max(b1))
# Depth
d1 = d - min(d)
d_norm = (d1 / max(d1))
# Hardness
h1 = h - min(h)
h_norm = (h1 / max(h1))
# Roughness
r1 = r - min(r)
r_norm = (r1 / max(r1))
    
#print "Brightness: \n", b, "\n", b_norm, "\n", "Depth: \n", d, "\n", d_norm, "\n", "Hard: \n", h, "\n", h_norm, "\n", "Roughness: \n", r, "\n", r_norm, "\n"


# Final CSV (with low and high-level features) creation

# In[30]:

pardir_csv = 'DATASET_PATH'
path_csv = os.path.join(pardir_csv,folder+"_features.csv")
df = pd.read_csv(path_csv,index_col=0)
df['brightness'] = b_norm
df['depth'] = d_norm
df['hardness'] = h_norm
df['roughness'] = r_norm
df.to_csv('OUT_DATASET_PATH'+folder+'_descriptors.csv')

