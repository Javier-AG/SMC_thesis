import pandas as pd
import numpy as np

data = pd.read_csv('EVALUATION_FILE_PATH')
# Testing global answers
response1 = data['response1']
no, yes, ns = 0, 0, 0
for i in response1:
	if i == 'NO':
		no+=1
	elif i == 'NOT SURE':
		ns+=1
	elif i == 'YES':
		yes+=1
	else:
		print "Error."
response2 = data['response2']
no_2, yes_2, ns_2 = 0, 0, 0
for i in response2:
	if i == 'NO':
		no_2+=1
	elif i == 'NOT SURE':
		ns_2+=1
	elif i == 'YES':
		yes_2+=1
	else:
		print "Error."
response3 = data['response3']
bad, good, ns_3 = 0, 0, 0
for i in response3:
	if i == 'BAD':
		bad+=1
	elif i == 'NOT SURE':
		ns_3+=1
	elif i == 'GOOD':
		good+=1
	else:
		print "Error."
# Testing category models
instrument = data['instrument']
no_closed, yes_closed, ns_closed = 0, 0, 0
no_open, yes_open, ns_open = 0, 0, 0
no_crash, yes_crash, ns_crash = 0, 0, 0
no_ride, yes_ride, ns_ride = 0, 0, 0
no_kick, yes_kick, ns_kick = 0, 0, 0
no_snare, yes_snare, ns_snare = 0, 0, 0
no_tom, yes_tom, ns_tom = 0, 0, 0
total_closed, total_open, total_crash, total_ride, total_kick, total_snare, total_tom = 0, 0, 0, 0, 0, 0, 0
for i in np.arange(len(instrument)):
    if instrument[i] == 'closedhh':
    	total_closed +=1
        if response2[i] == 'NO':
            no_closed+=1
        elif response2[i] == 'NOT SURE':
            ns_closed+=1
        elif response2[i] == 'YES':
            yes_closed+=1
    elif instrument[i] == 'openhh':
    	total_open+=1
        if response2[i] == 'NO':
            no_open+=1
        elif response2[i] == 'NOT SURE':
            ns_open+=1
        elif response2[i] == 'YES':
            yes_open+=1
    elif instrument[i] == 'crash':
    	total_crash+=1
        if response2[i] == 'NO':
            no_crash+=1
        elif response2[i] == 'NOT SURE':
            ns_crash+=1
        elif response2[i] == 'YES':
            yes_crash+=1
    elif instrument[i] == 'ride':
    	total_ride+=1
        if response2[i] == 'NO':
            no_ride+=1
        elif response2[i] == 'NOT SURE':
            ns_ride+=1
        elif response2[i] == 'YES':
            yes_ride+=1
    elif instrument[i] == 'kick':
    	total_kick+=1
        if response2[i] == 'NO':
            no_kick+=1
        elif response2[i] == 'NOT SURE':
            ns_kick+=1
        elif response2[i] == 'YES':
            yes_kick+=1
    elif instrument[i] == 'snare':
    	total_snare+=1
        if response2[i] == 'NO':
            no_snare+=1
        elif response2[i] == 'NOT SURE':
            ns_snare+=1
        elif response2[i] == 'YES':
            yes_snare+=1
    elif instrument[i] == 'tom':
    	total_tom+=1
        if response2[i] == 'NO':
            no_tom+=1
        elif response2[i] == 'NOT SURE':
            ns_tom+=1
        elif response2[i] == 'YES':
            yes_tom+=1

# Accuracy by users
print "\nRESULTS: \n"
print "INSTRUMENT MODEL: "
print "NO: ", (no/float(len(response1)))*100, "%	", "NOT SURE: ", (ns/float(len(response1)))*100, "%	", "YES: ", (yes/float(len(response1)))*100, "% \n"
print "CATEGORY MODELS: "
print "NO: ", (no_2/float(len(response2)))*100, "%	", "NOT SURE: ", (ns_2/float(len(response2)))*100, "%	", "YES: ", (yes_2/float(len(response2)))*100, "% \n"
print "CLOSED HIHAT MODEL"
print "NO: ", (no_closed/float(total_closed))*100, "%	", "NOT SURE: ", (ns_closed/float(total_closed))*100, "%	", "YES: ", (yes_closed/float(total_closed))*100, "%\n"
print "OPEN HIHAT MODEL"
print "NO: ", (no_open/float(total_open))*100, "%	", "NOT SURE: ", (ns_open/float(total_open))*100, "%	", "YES: ", (yes_open/float(total_open))*100, "%\n"
print "CRASH MODEL"
print "NO: ", (no_crash/float(total_crash))*100, "%	", "NOT SURE: ", (ns_crash/float(total_crash))*100, "%	", "YES: ", (yes_crash/float(total_crash))*100, "%\n"
print "RIDE MODEL"
print "NO: ", (no_ride/float(total_ride))*100, "%	", "NOT SURE: ", (ns_ride/float(total_ride))*100, "%	", "YES: ", (yes_ride/float(total_ride))*100, "%\n"
print "KICK MODEL"
print "NO: ", (no_kick/float(total_kick))*100, "%	", "NOT SURE: ", (ns_kick/float(total_kick))*100, "%	", "YES: ", (yes_kick/float(total_kick))*100, "%\n"
print "SNARE MODEL"
print "NO: ", (no_closed/float(total_snare))*100, "%	", "NOT SURE: ", (ns_snare/float(total_snare))*100, "%	", "YES: ", (yes_snare/float(total_snare))*100, "%\n"
print "TOM MODEL"
print "NO: ", (no_tom/float(total_tom))*100, "%	", "NOT SURE: ", (ns_tom/float(total_tom))*100, "%	", "YES: ", (yes_tom/float(total_tom))*100, "%\n"
print "HIGH-LEVEL DESCRIPTORS: "
print "NO: ", (bad/float(len(response2)))*100, "%	", "NOT SURE: ", (ns_3/float(len(response2)))*100, "%	", "YES: ", (good/float(len(response2)))*100, "% \n"

# Number of repeated selections
data_1 = data.groupby(['instrument', 'category']).size().reset_index().rename(columns={0:'count'})
print "REPEATED SELECTIONS"
print data_1, "\n"
