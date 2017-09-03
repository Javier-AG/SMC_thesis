from ipywidgets import widgets
from IPython.display import display, Audio
from IPython.core.display import display, HTML
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import re
import random

# LOAD BUTTONS, CHECKS & SLIDERS.

def load_interface1():
    instrument = widgets.RadioButtons(
        options=['kick', 'snare', 'tom', 'openhh', 'closedhh', 'ride', 'crash'],
        description='Select instrument:',
        disabled=False
    )
    category = widgets.RadioButtons(
        options=['acoustic', 'digital'],
        description='Select category:',
        disabled=False
    )
    # button = widgets.Button(
    #     description='Done',
    #     disabled=False,
    #     button_style='success', # 'success', 'info', 'warning', 'danger' or ''
    #     tooltip='Click me',
    #     icon=''
    # )
    accordion = widgets.Accordion(children=[instrument, category])
    accordion.set_title(0, 'Drum sample instrument')
    accordion.set_title(1, 'Drum sample category')
    
    return instrument, category, accordion

def load_interface2():

    check1 = widgets.Checkbox(
        value=False,
        description='Brigthness',
        disabled=False
    )
    slider1 = widgets.FloatSlider(
        value=0.5,
        min=0,
        max=1.0,
        step=0.1,
        readout=True,
        readout_format='.1f',
    )
    check2 = widgets.Checkbox(
        value=False,
        description='Depth',
        disabled=False
    )
    slider2 = widgets.FloatSlider(
        value=0.5,
        min=0,
        max=1.0,
        step=0.1,
        readout=True,
        readout_format='.1f',
    )
    check3 = widgets.Checkbox(
        value=False,
        description='Hardness',
        disabled=False
    )
    slider3 = widgets.FloatSlider(
        value=0.5,
        min=0,
        max=1.0,
        step=0.1,
        readout=True,
        readout_format='.1f',
    )
    check4 = widgets.Checkbox(
        value=False,
        description='Roughness',
        disabled=False
    )
    slider4 = widgets.FloatSlider(
        value=0.5,
        min=0,
        max=1.0,
        step=0.1,
        readout=True,
        readout_format='.1f',
    )
    
    return check1, slider1, check2, slider2, check3, slider3, check4, slider4

# DEFINITIONS TO MAKE SAMPLE RETRIEVAL.

def display_fs_embed(fs_id):
    
    # Display sound from freesound.org 
    html_code = '<iframe frameborder="0" scrolling="no" src="https://www.freesound.org/embed/sound/iframe/%i/simple/medium_no_info/" width="124" height="78"></iframe>' % fs_id
    whtml = widgets.HTML(
        value=html_code,
        placeholder='Some HTML',
        description='Your retrieved sample: \n',
        #disabled=False
    )
    display(whtml)

def search_sounds_and_show_results(instrument, category, check1, slider1, check2, slider2, check3, slider3, check4, slider4):
    
    # Read csv file (created by system analysis)
    df = pd.read_csv('free_dataset_predicted.csv',index_col=0)
    
    # Select only sounds that correspond to the user selection (instrument + category)
    instr = ['kick', 'snare', 'tom', 'openhh', 'closedhh', 'ride', 'crash']
    cat = ['acoustic', 'digital']
    if instrument.value == instr[0] and category.value == cat[0]:
        df1 = df[df['instrument'] == instr[0]]
        df2 = df1[df1['category'] == cat[0]]          
    elif instrument.value == instr[0] and category.value == cat[1]:
        df1 = df[df['instrument'] == instr[0]]
        df2 = df1[df1['category'] == cat[1]]
    elif instrument.value == instr[1] and category.value == cat[0]:
        df1 = df[df['instrument'] == instr[1]]
        df2 = df1[df1['category'] == cat[0]]
    elif instrument.value == instr[1] and category.value == cat[1]:
        df1 = df[df['instrument'] == instr[1]]
        df2 = df1[df1['category'] == cat[1]]
    elif instrument.value == instr[2] and category.value == cat[0]:
        df1 = df[df['instrument'] == instr[2]]
        df2 = df1[df1['category'] == cat[0]]
    elif instrument.value == instr[2] and category.value == cat[1]:
        df1 = df[df['instrument'] == instr[2]]
        df2 = df1[df1['category'] == cat[1]]
    elif instrument.value == instr[3] and category.value == cat[0]:
        df1 = df[df['instrument'] == instr[3]]
        df2 = df1[df1['category'] == cat[0]]
    elif instrument.value == instr[3] and category.value == cat[1]:
        df1 = df[df['instrument'] == instr[3]]
        df2 = df1[df1['category'] == cat[1]]
    elif instrument.value == instr[4] and category.value == cat[0]:
        df1 = df[df['instrument'] == instr[4]]
        df2 = df1[df1['category'] == cat[0]]
    elif instrument.value == instr[4] and category.value == cat[1]:
        df1 = df[df['instrument'] == instr[4]]
        df2 = df1[df1['category'] == cat[1]]
    elif instrument.value == instr[5] and category.value == cat[0]:
        df1 = df[df['instrument'] == instr[5]]
        df2 = df1[df1['category'] == cat[0]]
    elif instrument.value == instr[5] and category.value == cat[1]:
        df1 = df[df['instrument'] == instr[5]]
        df2 = df1[df1['category'] == cat[1]]
    elif instrument.value == instr[6] and category.value == cat[0]:
        df1 = df[df['instrument'] == instr[6]]
        df2 = df1[df1['category'] == cat[0]]
    elif instrument.value == instr[6] and category.value == cat[1]:
        df1 = df[df['instrument'] == instr[6]]
        df2 = df1[df1['category'] == cat[1]]
    else:
        print "Error during instrument or category selection process..."
    
    # Select sounds corresponding to high-level descriptors values from user selection
    # 4 descriptors selected
    if check1.value is True and check2.value is True and check3.value is True and check4.value is True:
        df3 = df2.loc[:,['brightness','depth','hardness','roughness']]
        user_selection = [slider1.value,slider2.value,slider3.value,slider4.value]
    # 3 descriptors selected    
    elif check1.value is True and check2.value is True and check3.value is True and check4.value is False:
        df3 = df2.loc[:,['brightness','depth','hardness']]
        user_selection = [slider1.value,slider2.value,slider3.value]
    elif check1.value is False and check2.value is True and check3.value is True and check4.value is True:
        df3 = df2.loc[:,['depth','hardness','roughness']] 
        user_selection = [slider2.value,slider3.value,slider4.value]
    elif check1.value is True and check2.value is False and check3.value is True and check4.value is True:
        df3 = df2.loc[:,['brightness','hardness','roughness']] 
        user_selection = [slider1.value,slider3.value,slider4.value]
    elif check1.value is True and check2.value is True and check3.value is False and check4.value is True:
        df3 = df2.loc[:,['brightness','depth','roughness']]
        user_selection = [slider1.value,slider2.value,slider4.value]
    # 2 descriptors selected    
    elif check1.value is True and check2.value is True and check3.value is False and check4.value is False:
        df3 = df2.loc[:,['brightness','depth']]
        user_selection = [slider1.value,slider2.value]
    elif check1.value is True and check2.value is False and check3.value is True and check4.value is False:
        df3 = df2.loc[:,['brightness','hardness']]
        user_selection = [slider1.value,slider3.value]
    elif check1.value is True and check2.value is False and check3.value is False and check4.value is True:
        df3 = df2.loc[:,['brightness','roughness']]
        user_selection = [slider1.value,slider4.value]
    elif check1.value is False and check2.value is True and check3.value is True and check4.value is False:
        df3 = df2.loc[:,['depth','hardness']] 
        user_selection = [slider2.value,slider3.value]
    elif check1.value is False and check2.value is True and check3.value is False and check4.value is True:
        df3 = df2.loc[:,['depth','roughness']]
        user_selection = [slider2.value,slider4.value]
    elif check1.value is False and check2.value is False and check3.value is True and check4.value is True:
        df3 = df2.loc[:,['hardness','roughness']]
        user_selection = [slider3.value,slider4.value]
    # 1 descriptor selected
    elif check1.value is True and check2.value is False and check3.value is False and check4.value is False:
        df3 = df2.loc[:,['brightness']]
        user_selection = [slider1.value]
    elif check1.value is False and check2.value is True and check3.value is False and check4.value is False:
        df3 = df2.loc[:,['depth']] 
        user_selection = [slider2.value]
    elif check1.value is False and check2.value is False and check3.value is True and check4.value is False:
        df3 = df2.loc[:,['hardness']] 
        user_selection = [slider3.value]
    elif check1.value is False and check2.value is False and check3.value is False and check4.value is True:
        df3 = df2.loc[:,['roughness']]
        user_selection = [slider4.value]
    # No descriptor selected
    elif check1.value is False and check2.value is False and check3.value is False and check4.value is False:
        df3 = df2.loc[:,['roughness']] # we only need same structure, random choice at the end
        user_selection = 0
    else:
        print "Error during descriptors selection process..."
        
    # Gather FREESOUND ID from retrieved sound name
    list_IDs = [] # list of Freesound IDs
    for i in df3.index:
        list_IDs = np.append(list_IDs, re.split(r"_",i)[0]) # get IDs from sounds names
    if user_selection == 0: # no high-level descriptor selected
        ID = int(random.choice(list_IDs))
    else: # one or more high-level descriptors selected
        candidates = df3.as_matrix()
        neigh = NearestNeighbors(n_neighbors=1).fit(candidates)
        dist, i = neigh.kneighbors([user_selection])
        ID = int(list_IDs[i][0][0])
    print "Retrieved sound ID: ", ID
    display_fs_embed(ID)

# WRITE YOUR PERSONAL DATA

def load_personal():
    name = widgets.Text(
        value='Name and Surname',
        placeholder='Type something',
        description='Name: ',
        disabled=False
    )
    exp = widgets.SelectMultiple(
        options=['Drummer', 'Music Producer', 'None'],
        value=['None'],
        rows=3,
        description='Musical experience: ',
        disabled=False
    )
    exp2 = widgets.SelectMultiple(
        options=['Less than 5 years', 'More than 5 years', 'None'],
        value=['None'],
        rows=3,    
        description='Years of experience: ',
        disabled=False
    )
    return name, exp, exp2

# QUESTIONS

def load_questions():
    response1 = widgets.ToggleButtons(
            options=['NO', 'NOT SURE', 'YES'],
            description='',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
        )
    response2 = widgets.ToggleButtons(
            options=['NO', 'NOT SURE', 'YES'],
            description='',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
        )
    response3 = widgets.ToggleButtons(
            options=['BAD', 'NOT SURE', 'GOOD'],
            description='',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
        )
    return response1, response2, response3
