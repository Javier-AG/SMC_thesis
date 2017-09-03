import os
import sys
import random
import pandas as pd
import numpy as np
import yaml
import subprocess

folder = 'FOLDER_NAME'
pardir = 'PARENT_DIRECTORY_PATH' + folder + '/'
category = 'acoustic' # or 'digital'
filenames = os.listdir(pardir)
outdir = os.getcwd() + "/"  
samples = np.array([track for track in filenames if (track[-3:] == 'wav')])

# DICTIONARY CREATION from yaml to csv

feat_df = pd.DataFrame()
names = np.array([])
elementsTotal = 0
# Create a dictionary
for track in samples:
    elementsTotal+=1
    yamlFeaturesFile = os.path.join(outdir + track[:-4] + '_statistics.yaml')
    if os.path.isfile(yamlFeaturesFile):
        #print "+++++++++++++ " + yamlFeaturesFile + " ++++++++++++++++"
        names = np.append(names,track)        
        df = pd.DataFrame()
        f = open(yamlFeaturesFile)
        feat_dict = yaml.safe_load(f)
        f.close()

        # disacrded as irrelevant or because it has inf values
        feat_dict.pop('tonal',None)
        feat_dict.pop('rhythm',None)
        feat_dict.pop('metadata',None)
        feat_dict['sfx'].pop('oddtoevenharmonicenergyratio',None)
        feat_dict['lowlevel'].pop('startFrame',None)
        feat_dict['lowlevel'].pop('stopFrame',None)

        for family in feat_dict.keys():
            for desc in feat_dict[family]:
                if type(feat_dict[family][desc]) == dict:
                    if type(feat_dict[family][desc]['mean']) != list:                
                        df[desc + '_' + 'mean'] = [feat_dict[family][desc]['mean']]
                    else:
                        for i in range(len(feat_dict[family][desc]['mean'])):
                            df[desc + '_' + 'mean' + '_' + str(i)] = feat_dict[family][desc]['mean'][i]
                            df[desc + '_' + 'var' + '_' + str(i)] = feat_dict[family][desc]['var'][i]            
                else:
                    df[desc] = feat_dict[family][desc]
        feat_df = pd.concat([feat_df,df],ignore_index=True)
    #print "+++++++++++++++++++"
    #print feat_df
    #print "+++++++++++++++++++"
    
# Add instrument class        
feat_df['instrument'] = folder
# Add category class        
feat_df['category'] = category
# Convert dictionary to CSV
newIndex = feat_df.set_index(names)
newIndex.to_csv(outdir + '_features.csv',)
# Show how many samples there are in the folder
print 'Num of Samples (Total): ', elementsTotal