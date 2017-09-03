from sklearn import svm, neighbors, tree
from sklearn.model_selection import cross_val_score
import pandas as pd
import csv
import numpy as np

# Read trainning and evaluation datasets. Feature selection.

train_path = 'commercial_dataset_features.csv'
train_data = pd.read_csv(train_path, index_col=0)
closedhh_data = train_data.loc[train_data['instrument'] == 'closedhh']
openhh_data = train_data.loc[train_data['instrument'] == 'openhh']
crash_data = train_data.loc[train_data['instrument'] == 'crash']
ride_data = train_data.loc[train_data['instrument'] == 'ride']
kick_data = train_data.loc[train_data['instrument'] == 'kick']
snare_data = train_data.loc[train_data['instrument'] == 'snare']
tom_data = train_data.loc[train_data['instrument'] == 'tom']
eval_path = 'free_dataset_features.csv'
eval_data = pd.read_csv(eval_path, index_col=0)
closedhh_data_eval = eval_data.loc[eval_data['instrument'] == 'closedhh']
openhh_data_eval = eval_data.loc[eval_data['instrument'] == 'openhh']
crash_data_eval = eval_data.loc[eval_data['instrument'] == 'crash']
ride_data_eval = eval_data.loc[eval_data['instrument'] == 'ride']
kick_data_eval = eval_data.loc[eval_data['instrument'] == 'kick']
snare_data_eval = eval_data.loc[eval_data['instrument'] == 'snare']
tom_data_eval = eval_data.loc[eval_data['instrument'] == 'tom']

# INSTRUMENT feature selection
data_instr_train = train_data[['spectral_contrast_var_0','spectral_contrast_mean_1','spectral_contrast_mean_2','spectral_contrast_mean_3','spectral_contrast_mean_4','spectral_contrast_mean_5','spectral_entropy_mean','pitch_instantaneous_confidence_mean','effective_duration_mean','logattacktime_mean','instrument']].copy()
X_instr = data_instr_train.iloc[:,:-1].values
y_instr = data_instr_train.instrument.values
data_instr_eval = eval_data[['spectral_contrast_var_0','spectral_contrast_mean_1','spectral_contrast_mean_2','spectral_contrast_mean_3','spectral_contrast_mean_4','spectral_contrast_mean_5','spectral_entropy_mean','pitch_instantaneous_confidence_mean','effective_duration_mean','logattacktime_mean','instrument']].copy()
X_instr_eval = data_instr_eval.iloc[:,:-1].values
y_instr_eval = data_instr_eval.instrument.values

# CLOSEDHH CATEGORY feature selection
data_closed_train = closedhh_data[['dissonance_mean','barkbands_spread_mean','mfcc_mean_1','tristimulus_var_0','category']].copy()
X_closed = data_closed_train.iloc[:,:-1].values
y_closed = data_closed_train.category.values
data_closed_eval = eval_data[['dissonance_mean','barkbands_spread_mean','mfcc_mean_1','tristimulus_var_0','category']].copy()
X_closed_eval = data_closed_eval.iloc[:,:-1].values
y_closed_eval = data_closed_eval.category.values

# OPENHH CATEGORY feature selection
data_open_train = openhh_data[['pitch_mean','pitch_salience_mean','spectral_strongpeak_mean','mfcc_mean_1','mfcc_mean_7','mfcc_var_9','tristimulus_mean_2','category']].copy()
X_open = data_open_train.iloc[:,:-1].values
y_open = data_open_train.category.values
data_open_eval = eval_data[['pitch_mean','pitch_salience_mean','spectral_strongpeak_mean','mfcc_mean_1','mfcc_mean_7','mfcc_var_9','tristimulus_mean_2','category']].copy()
X_open_eval = data_open_eval.iloc[:,:-1].values
y_open_eval = data_open_eval.category.values

# CRASH CATEGORY feature selection
data_crash_train = crash_data[['spectral_spread_mean','spectral_entropy_mean','flatness_mean','category']].copy()
X_crash = data_crash_train.iloc[:,:-1].values
y_crash = data_crash_train.category.values
data_crash_eval = eval_data[['spectral_spread_mean','spectral_entropy_mean','flatness_mean','category']].copy()
X_crash_eval = data_crash_eval.iloc[:,:-1].values
y_crash_eval = data_crash_eval.category.values

# RIDE CATEGORY feature selection
data_ride_train = ride_data[['spectral_spread_mean','spectral_entropy_mean','silence_rate_60dB_mean','category']].copy()
X_ride = data_ride_train.iloc[:,:-1].values
y_ride = data_ride_train.category.values
data_ride_eval = eval_data[['spectral_spread_mean','spectral_entropy_mean','silence_rate_60dB_mean','category']].copy()
X_ride_eval = data_ride_eval.iloc[:,:-1].values
y_ride_eval = data_ride_eval.category.values

# KICK CATEGORY feature selection
data_kick_train = kick_data[['mfcc_var_4','mfcc_var_7','spectral_energy_mean','category']].copy()
X_kick = data_kick_train.iloc[:,:-1].values
y_kick = data_kick_train.category.values
data_kick_eval = eval_data[['mfcc_var_4','mfcc_var_7','spectral_energy_mean','category']].copy()
X_kick_eval = data_kick_eval.iloc[:,:-1].values
y_kick_eval = data_kick_eval.category.values

# TOM CATEGORY feature selection
data_tom_train = tom_data[['gfcc_var_1','barkbands_mean_1','category']].copy()
X_tom = data_tom_train.iloc[:,:-1].values
y_tom = data_tom_train.category.values
data_tom_eval = eval_data[['gfcc_var_1','barkbands_mean_1','category']].copy()
X_tom_eval = data_tom_eval.iloc[:,:-1].values
y_tom_eval = data_tom_eval.category.values

# SNARE CATEGORY feature selection
data_snare_train = snare_data[['spectral_entropy_mean','gfcc_mean_1','barkbands_var_17','barkbands_var_22','barkbands_var_23','erb_bands_var_17','category']].copy()
X_snare = data_snare_train.iloc[:,:-1].values
y_snare = data_snare_train.category.values
data_snare_eval = eval_data[['spectral_entropy_mean','gfcc_mean_1','barkbands_var_17','barkbands_var_22','barkbands_var_23','erb_bands_var_17','category']].copy()
X_snare_eval = data_snare_eval.iloc[:,:-1].values
y_snare_eval = data_snare_eval.category.values


# Fit classification models with SVM.

instrument_model = svm.SVC(C=1.0, kernel='poly')
inst = instrument_model.fit(X_instr, y_instr) 
closed_model = svm.SVC(C=1.0, kernel='linear')
cl = closed_model.fit(X_closed, y_closed) 
open_model = svm.SVC(C=1.0, kernel='linear')
op = open_model.fit(X_open, y_open)
crash_model = svm.SVC(C=1.0, kernel='poly')
cr = crash_model.fit(X_crash, y_crash)
ride_model = svm.SVC(C=1.0, kernel='poly')
ri = ride_model.fit(X_ride, y_ride)
kick_model = svm.SVC(C=1.0, kernel='linear')
ki = kick_model.fit(X_kick, y_kick)
snare_model = svm.SVC(C=1.0, kernel='linear')
sn = snare_model.fit(X_snare, y_snare)
tom_model = svm.SVC(C=1.0, kernel='linear')
to = tom_model.fit(X_tom, y_tom) 


# Predict instrument and category classes.

instrument = []
category = []
for i in np.arange(len(X_instr_eval)):
    a = inst.predict([X_instr_eval[i]])
    instrument = np.append(instrument, a)
    if instrument[i] == 'closedhh':
        category = np.append(category, cl.predict([X_closed_eval[i]]))
    elif instrument[i] == 'openhh':
        category = np.append(category, op.predict([X_open_eval[i]]))
    elif instrument[i] == 'crash':
        category = np.append(category, cr.predict([X_crash_eval[i]]))
    elif instrument[i] == 'ride': 
        category = np.append(category, ri.predict([X_ride_eval[i]]))
    elif instrument[i] == 'kick':
        category = np.append(category, ki.predict([X_kick_eval[i]]))
    elif instrument[i] == 'snare': 
        category = np.append(category, sn.predict([X_snare_eval[i]]))
    elif instrument[i] == 'tom': 
        category = np.append(category, to.predict([X_tom_eval[i]]))
    else:
        "Error."
print "+++++++++++++++++"
print "Real instrument: \n", eval_data['instrument'].as_matrix(), "\n", len(eval_data['instrument'].as_matrix())
print "Predicted instrument: \n", instrument, "\n", len(instrument)
print "+++++++++++++++++"
print "CLOSEDHH"
print "Real category: \n", closedhh_data_eval['category'].as_matrix(), "\n", len(closedhh_data_eval['category'].as_matrix())
print "Predicted category: \n", category[:len(closedhh_data_eval['category'].as_matrix())], "\n", len(category[:len(closedhh_data_eval['category'].as_matrix())])
print "OPENHH"
print "Real category: \n", openhh_data_eval['category'].as_matrix(), "\n", len(openhh_data_eval['category'].as_matrix())
print "Predicted category: \n", category[:len(openhh_data_eval['category'].as_matrix())], "\n", len(category[:len(openhh_data_eval['category'].as_matrix())])
print "CRASH"
print "Real category: \n", crash_data_eval['category'].as_matrix(), "\n", len(crash_data_eval['category'].as_matrix())
print "Predicted category: \n", category[:len(crash_data_eval['category'].as_matrix())], "\n", len(category[:len(crash_data_eval['category'].as_matrix())])
print "RIDE"
print "Real category: \n", ride_data_eval['category'].as_matrix(), "\n", len(ride_data_eval['category'].as_matrix())
print "Predicted category: \n", category[:len(ride_data_eval['category'].as_matrix())], "\n", len(category[:len(ride_data_eval['category'].as_matrix())])
print "KICK"
print "Real category: \n", kick_data_eval['category'].as_matrix(), "\n", len(kick_data_eval['category'].as_matrix())
print "Predicted category: \n", category[:len(kick_data_eval['category'].as_matrix())], "\n", len(category[:len(kick_data_eval['category'].as_matrix())])
print "SNARE"
print "Real category: \n", snare_data_eval['category'].as_matrix(), "\n", len(snare_data_eval['category'].as_matrix())
print "Predicted category: \n", category[:len(snare_data_eval['category'].as_matrix())], "\n", len(category[:len(snare_data_eval['category'].as_matrix())])
print "TOM"
print "Real category: \n", tom_data_eval['category'].as_matrix(), "\n", len(tom_data_eval['category'].as_matrix())
print "Predicted category: \n", category[:len(tom_data_eval['category'].as_matrix())], "\n", len(category[:len(tom_data_eval['category'].as_matrix())])


# Models' and predictions' accuracies.

accuracy_NI = np.mean(cross_val_score(instrument_model,X_instr,y_instr, cv=10))
accuracy_FS = np.mean(cross_val_score(instruent_model,X_instr_eval,y_instr_eval, cv=10))

accuracy_NI_closed = np.mean(cross_val_score(closed_model,X_closed,y_closed, cv=10))
accuracy_FS_closed = np.mean(cross_val_score(closed_model,X_closed_eval,y_closed_eval, cv=10))

accuracy_NI_open = np.mean(cross_val_score(open_model,X_open,y_open, cv=10))
accuracy_FS_open = np.mean(cross_val_score(open_model,X_open_eval,y_open_eval, cv=10))

accuracy_NI_crash = np.mean(cross_val_score(crash_model,X_crash,y_crash, cv=10))
accuracy_FS_crash = np.mean(cross_val_score(crash_model,X_crash_eval,y_crash_eval, cv=10))

accuracy_NI_ride = np.mean(cross_val_score(ride_model,X_ride,y_ride, cv=10))
accuracy_FS_ride = np.mean(cross_val_score(ride_model,X_ride_eval,y_ride_eval, cv=10))

accuracy_NI_kick = np.mean(cross_val_score(kick_model,X_kick,y_kick, cv=10))
accuracy_FS_kick = np.mean(cross_val_score(kick_model,X_kick_eval,y_kick_eval, cv=10))

accuracy_NI_snare = np.mean(cross_val_score(snare_model,X_snare,y_snare, cv=10))
accuracy_FS_snare = np.mean(cross_val_score(snare_model,X_snare_eval,y_snare_eval, cv=10))

accuracy_NI_tom = np.mean(cross_val_score(tom_model,X_tom,y_tom, cv=10))
accuracy_FS_tom = np.mean(cross_val_score(tom_model,X_tom_eval,y_tom_eval, cv=10))

closedhh_a,openhh_a,crash_a,ride_a,kick_a,snare_a,tom_a = 0, 0, 0, 0, 0, 0, 0 
closedhh_d,openhh_d,crash_d,ride_d,kick_d,snare_d,tom_d = 0, 0, 0, 0, 0, 0, 0 
for i in np.arange(len(instrument)):
    if instrument[i] == 'closedhh' and category[i] == 'acoustic':
        closedhh_a+=1
    elif instrument[i] == 'closedhh' and category[i] == 'digital':
        closedhh_d+=1
    elif instrument[i] == 'openhh' and category[i] == 'acoustic':
        openhh_a+=1
    elif instrument[i] == 'openhh' and category[i] == 'digital':
        openhh_d+=1
    elif instrument[i] == 'crash' and category[i] == 'acoustic':
        crash_a+=1
    elif instrument[i] == 'crash' and category[i] == 'digital':
        crash_d+=1
    elif instrument[i] == 'ride' and category[i] == 'acoustic':
        ride_a+=1
    elif instrument[i] == 'ride' and category[i] == 'digital':
        ride_d+=1
    elif instrument[i] == 'kick' and category[i] == 'acoustic':
        kick_a+=1
    elif instrument[i] == 'kick' and category[i] == 'digital':
        kick_d+=1
    elif instrument[i] == 'snare' and category[i] == 'acoustic':
        snare_a+=1
    elif instrument[i] == 'snare' and category[i] == 'digital':
        snare_d+=1
    elif instrument[i] == 'tom' and category[i] == 'acoustic':
        tom_a+=1
    elif instrument[i] == 'tom' and category[i] == 'digital':
        tom_d+=1

closed_num = eval_data.loc[eval_data['instrument'] == 'closedhh']
cl_a,cl_d = 0, 0
for i in closed_num['category']:
    if i == 'acoustic':
        cl_a+=1
    elif i == 'digital':
        cl_d+=1
open_num = eval_data.loc[eval_data['instrument'] == 'openhh']
op_a,op_d = 0, 0
for i in open_num['category']:
    if i == 'acoustic':
        op_a+=1
    elif i == 'digital':
        op_d+=1
crash_num = eval_data.loc[eval_data['instrument'] == 'crash']
cr_a,cr_d = 0, 0
for i in crash_num['category']:
    if i == 'acoustic':
        cr_a+=1
    elif i == 'digital':
        cr_d+=1       
ride_num = eval_data.loc[eval_data['instrument'] == 'ride']
ri_a,ri_d = 0, 0
for i in ride_num['category']:
    if i == 'acoustic':
        ri_a+=1
    elif i == 'digital':
        ri_d+=1 
kick_num = eval_data.loc[eval_data['instrument'] == 'kick']
ki_a,ki_d = 0, 0
for i in kick_num['category']:
    if i == 'acoustic':
        ki_a+=1
    elif i == 'digital':
        ki_d+=1 
snare_num = eval_data.loc[eval_data['instrument'] == 'snare']
sn_a,sn_d = 0, 0
for i in snare_num['category']:
    if i == 'acoustic':
        sn_a+=1
    elif i == 'digital':
        sn_d+=1 
tom_num = eval_data.loc[eval_data['instrument'] == 'tom']
to_a,to_d = 0, 0
for i in tom_num['category']:
    if i == 'acoustic':
        to_a+=1
    elif i == 'digital':
        to_d+=1 
        
print "closedhh acoustic: ", closedhh_a, "predicted samples, ", cl_a, "real samples"
print "closedhh digital ", closedhh_d, "predicted samples, ", cl_d, "real samples"
print "openhh acoustic ", openhh_a, "predicted samples, ", op_a, "real samples"
print "openhh digital ", openhh_d, "predicted samples, ", op_d, "real samples"
print "ride acoustic ", ride_a, "predicted samples, ", ri_a, "real samples"
print "ride digital ", ride_d, "predicted samples, ", ri_d, "real samples"
print "crash acoustic ", crash_a, "predicted samples, ", cr_a, "real samples"
print "crash digital ", crash_d, "predicted samples, ", cr_d, "real samples"
print "kick acoustic ", kick_a, "predicted samples, ", ki_a, "real samples"
print "kick digital ", kick_d, "predicted samples, ", ki_d, "real samples"
print "snare acoustic ", snare_a, "predicted samples, ", sn_a, "real samples"
print "snare digital ", snare_d, "predicted samples, ", sn_d, "real samples"
print "tom acoustic ", tom_a, "predicted samples, ", to_a, "real samples"
print "tom digital ", tom_d, "predicted samples, ", to_d, "real samples"

flag_ins,flag,f_closed,f_open,f_crash,f_ride,f_kick,f_snare,f_tom = 0, 0, 0, 0, 0, 0, 0, 0, 0

for i in np.arange(len(instrument)):
    if eval_data['instrument'].as_matrix()[i] == instrument[i]:
        flag_ins+=1
        if eval_data['category'].as_matrix()[i] == category[i]:
            flag+=1
            print i, instrument[i], category[i]
            if category[i] == 'closedhh':
                f_closed+=1
            elif category[i] == 'openhh':
                f_open+=1
            elif category[i] == 'crash':
                f_crash+=1
            elif category[i] == 'ride':
                f_ride+=1
            elif category[i] == 'kick':
                f_kick+=1
            elif category[i] == 'snare':
                f_snare+=1
            elif category[i] == 'tom':
                f_tom+=1
#print flag_ins, flag                
prediction = flag_ins / float(len(instrument))
prediction_closed = f_closed / float(cl_a + cl_d)
prediction_open = f_open / float(op_a + op_d)
prediction_crash = f_crash / float(cr_a + cr_d)
prediction_ride = f_ride / float(ri_a + ri_d)
prediction_kick = f_kick / float(ki_a + ki_d)
prediction_snare = f_snare / float(sn_a + sn_d)
prediction_tom = f_tom / float(to_a + to_d)
prediction_total = flag / float(len(category))

print "++++++++++"
print "INSTRUMENT MODEL \n"
print "Commercial (Training) Dataset accuracy: ", accuracy_NI*100,"% \n", "Free (Evaluation) Dataset accuracy: ", accuracy_FS*100,"% \n"
print "Prediction accuracy: ", prediction*100,"% \n"           
print "++++++++++"
print "CATEGORY MODELS \n"
print "CLOSEDHH MODEL"
print "Commercial (Training) Dataset accuracy: ", accuracy_NI_closed*100,"% \n", "Free (Evaluation) Dataset accuracy: ", accuracy_FS_closed*100,"%"
print "Prediction accuracy: ", prediction_closed*100,"% \n"
print "OPENHH MODEL"
print "Commercial (Training) Dataset accuracy: ", accuracy_NI_open*100,"% \n", "Free (Evaluation) Dataset accuracy: ", accuracy_FS_open*100,"%"
print "Prediction accuracy: ", prediction_open*100,"% \n"
print "CRASH MODEL"
print "Commercial (Training) Dataset accuracy: ", accuracy_NI_crash*100,"% \n", "Free (Evaluation) Dataset accuracy: ", accuracy_FS_crash*100,"%"
print "Prediction accuracy: ", prediction_crash*100,"% \n"
print "RIDE MODEL"
print "Commercial (Training) Dataset accuracy: ", accuracy_NI_ride*100,"% \n", "Free (Evaluation) Dataset accuracy: ", accuracy_FS_ride*100,"%"
print "Prediction accuracy: ", prediction_ride*100,"% \n"
print "KICK MODEL"
print "Commercial (Training) Dataset accuracy: ", accuracy_NI_kick*100,"% \n", "Free (Evaluation) Dataset accuracy: ", accuracy_FS_kick*100,"%"
print "Prediction accuracy: ", prediction_kick*100,"% \n"
print "SNARE MODEL"
print "Commercial (Training) Dataset accuracy: ", accuracy_NI_snare*100,"% \n", "Free (Evaluation) Dataset accuracy: ", accuracy_FS_snare*100,"%"
print "Prediction accuracy: ", prediction_snare*100,"% \n"
print "TOM MODEL"
print "Commercial (Training) Dataset accuracy: ", accuracy_NI_tom*100,"% \n", "Free (Evaluation) Dataset accuracy: ", accuracy_FS_tom*100,"%"
print "Prediction accuracy: ", prediction_tom*100,"% \n"
print "TOTAL PREDICTION"
print "Prediction accuracy: ", prediction_total*100,"%"


# Export new CSV with predicted values

descriptors_data = pd.read_csv('free_dataset_descriptors.csv', index_col=0)
descriptors_data = descriptors_data.drop('instrument', 1)
descriptors_data = descriptors_data.drop('category', 1)
descriptors_data['instrument'] = instrument
descriptors_data['category'] = category
descriptors_data.to_csv('free_dataset_predicted.csv')
