
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import altair as alt
from collections import OrderedDict
from sklearn.inspection import permutation_importance
from layout import get_wide_container

import streamlit as st
import altair as alt
import pandas as pd
import pprint
import numpy as np

from sklearn.inspection import permutation_importance

# In[2]:


cols = ['age',
'workclass',
'fnlwgt',
'education',
'education-num',
'marital-status',
'occupation',
'relationship',
'race',
'sex',
'capital-gain',
'capital-loss',
'hours-per-week',
'native-country', 'income']


# In[3]:


data = pd.read_csv('census_data.csv', names = cols, index_col = False)


# In[4]:




# In[5]:


data.drop(columns=['fnlwgt','relationship'])


# In[9]:


uni = ((data['occupation'].unique()))
feature2_data = ((data['race'].unique()))
feature3_data = ((data['marital-status'].unique()))
feature4_data = ((data['education'].unique()))


# In[39]:


jobs = ((data['occupation']))
race = ((data['race']))
marital = ((data['marital-status']))
education = ((data['education']))


# In[17]:


jobs_types = pd.DataFrame(uni, columns = ['occupation'])
race_types = pd.DataFrame(feature2_data, columns = ['race'])
marital_types = pd.DataFrame(feature3_data, columns = ['marital-status'])
education_types = pd.DataFrame(feature4_data, columns = ['education'])


# In[88]:



jobs_index = []
for i in range(len(uni)):
    jobs_index.append(i)
    
jobs_types['occupation_number'] = jobs_index

def convert_to_job_num(job,jobs_types):
    b = 0
    job_num = 0
    for types in jobs_types['occupation']:
        if (job == types):
            job_num = jobs_types['occupation_number'][b]
        b = b+1
    return job_num

occupy = []
for values in jobs:
    occupy.append(convert_to_job_num(values,jobs_types))
job_class = pd.DataFrame(occupy, columns = ['job_numbers'])
job_class['income'] = data['income']
jc = job_class.values.tolist()


# In[94]:



race_index = []
for i in range(len(feature2_data)):
    race_index.append(i)
    
race_types['race_number'] = race_index

def convert_to_race_num(race,race_types):
    b = 0
    job_num = 0
    for types in race_types['race']:
        if (race == types):
            job_num = race_types['race_number'][b]
        b = b+1
    return job_num

race_holder = []
for values in race:
    race_holder.append(convert_to_race_num(values,race_types))
race_class = pd.DataFrame(race_holder, columns = ['race_numbers'])
race_class['income'] = data['income']
rc = race_class.values.tolist()


# In[126]:



marital_index = []
for i in range(len(feature3_data)):
    marital_index.append(i)
    
marital_types['marital_number'] = marital_index

def convert_to_marital_num(marital,marital_types):
    b = 0
    job_num = 0
    for types in marital_types['marital-status']:
        if (marital == types):
            job_num = marital_types['marital_number'][b]
        b = b+1
    return job_num

marital_holder = []
for values in marital:
    marital_holder.append(convert_to_marital_num(values,marital_types))
marital_class = pd.DataFrame(marital_holder, columns = ['marital_numbers'])
marital_class['income'] = data['income']
mc = marital_class.values.tolist()


# In[134]:



education_index = []
for i in range(len(feature4_data)):
    education_index.append(i)
    
education_types['education_numbers'] = education_index

def convert_to_education_num(education,education_types):
    b = 0
    job_num = 0
    for types in education_types['education']:
        if (education == types):
            job_num = education_types['education_numbers'][b]
        b = b+1
    return job_num

education_holder = []
for values in education:
    education_holder.append(convert_to_education_num(values,education_types))
education_class = pd.DataFrame(education_holder, columns = ['education_numbers'])
education_class['income'] = data['income']
ec = education_class.values.tolist()


# In[ ]:


less = job_class['income'].unique()[0]
greater = job_class['income'].unique()[1]


# In[136]:


count4 = [0]*len(feature4_data)

for i in range(len(count4)):
    counter4 = 0
    for item in ec:
        if((item[1] == greater) and (item[0]==i)):
            counter4 = counter4 + 1
    count4[i] = counter4
  


# In[137]:


education_frame = pd.DataFrame(feature4_data, columns = ['education-status'])
education_frame['above_50K'] = count4


# In[128]:


count3 = [0]*len(feature3_data)

for i in range(len(count3)):
    counter3 = 0
    for item in mc:
        if((item[1] == greater) and (item[0]==i)):
            counter3 = counter3 + 1
    count3[i] = counter3
  


# In[131]:


marital_frame = pd.DataFrame(feature3_data, columns = ['marital-status'])
marital_frame['above_50K'] = count3


# In[108]:


count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

for i in range(len(count)):
    counter = 0
    for item in jc:
        if((item[1] == greater) and (item[0]==i)):
            counter = counter + 1
    count[i] = counter
    


# In[109]:


count2 = [0]*len(feature2_data)

for i in range(len(count2)):
    counter2 = 0
    for item in rc:
        if((item[1] == greater) and (item[0]==i)):
            counter2 = counter2 + 1
    count2[i] = counter2
    


# In[142]:


jobbo = pd.DataFrame(uni, columns = ['job_types'])
jobbo['above_50K'] = count


# In[147]:


race_frame = pd.DataFrame(feature2_data, columns = ['race_types'])
race_frame['above_50K'] = count2


# In[149]:





# In[123]:


new_frame = pd.concat([jobbo, race_frame])


# In[144]:




def add_labels(angles, values, labels, offset, ax):
    
    # This is the space between the end of the bar and the label
    padding = 4
    
    # Iterate over angles, values, and labels, to add all of them.
    for angle, value, label, in zip(angles, values, labels):
        angle = angle
        
        # Obtain text rotation and alignment
        rotation, alignment = get_label_rotation(angle, offset)

        # And finally add the text
        ax.text(
            x=angle, 
            y=value + padding, 
            s=label, 
            ha=alignment, 
            va="center", 
            rotation=rotation, 
            rotation_mode="anchor"
        ) 


# In[145]:


def get_label_rotation(angle, offset):
    # Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"
    return rotation, alignment


# In[150]:


ANGLES = np.linspace(0, 2 * np.pi, len(jobbo), endpoint=False)
VALUES = jobbo["above_50K"].values
LABELS = jobbo["job_types"].values

# Determine the width of each bar. 
# The circumference is '2 * pi', so we divide that total width over the number of bars.
WIDTH = 2 * np.pi / len(VALUES)

# Determines where to place the first bar. 
# By default, matplotlib starts at 0 (the first bar is horizontal)
# but here we say we want to start at pi/2 (90 deg)
OFFSET = np.pi / 2

# Initialize Figure and Axis
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(20, 10), subplot_kw={"projection": "polar"})

# Specify offset
ax.set_theta_offset(OFFSET)

# Set limits for radial (y) axis. The negative lower bound creates the whole in the middle.
ax.set_ylim(-100, 2000)

# Remove all spines
ax.set_frame_on(False)

# Remove grid and tick marks
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_xticks([])
ax.set_yticks([])

# Add bars
ax.bar(
    ANGLES, VALUES, width=WIDTH, linewidth=2,
    color="#61a4b2", edgecolor="black"
)

# Add labels
add_labels(ANGLES, VALUES, LABELS, OFFSET, ax)


# In[153]:


ANGLES = np.linspace(0, 2 * np.pi, len(race_frame), endpoint=False)
VALUES = race_frame["above_50K"].values
LABELS = race_frame["race_types"].values

# Determine the width of each bar. 
# The circumference is '2 * pi', so we divide that total width over the number of bars.
WIDTH = 2 * np.pi / len(VALUES)

# Determines where to place the first bar. 
# By default, matplotlib starts at 0 (the first bar is horizontal)
# but here we say we want to start at pi/2 (90 deg)
OFFSET = np.pi / 2

# Initialize Figure and Axis
plt.style.use('dark_background')
fig2, ax = plt.subplots(figsize=(20, 10), subplot_kw={"projection": "polar"})

# Specify offset
ax.set_theta_offset(OFFSET)

# Set limits for radial (y) axis. The negative lower bound creates the whole in the middle.
ax.set_ylim(-100, 10000)

# Remove all spines
ax.set_frame_on(False)

# Remove grid and tick marks
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_xticks([])
ax.set_yticks([])

# Add bars
ax.bar(
    ANGLES, VALUES, width=WIDTH, linewidth=2,
    color="#61a4b2", edgecolor="black"
)

# Add labels
add_labels(ANGLES, VALUES, LABELS, OFFSET, ax)


# In[155]:


ANGLES = np.linspace(0, 2 * np.pi, len(marital_frame), endpoint=False)
VALUES = marital_frame["above_50K"].values
LABELS = marital_frame["marital-status"].values

# Determine the width of each bar. 
# The circumference is '2 * pi', so we divide that total width over the number of bars.
WIDTH = 2 * np.pi / len(VALUES)

# Determines where to place the first bar. 
# By default, matplotlib starts at 0 (the first bar is horizontal)
# but here we say we want to start at pi/2 (90 deg)
OFFSET = np.pi / 2

# Initialize Figure and Axis
plt.style.use('dark_background')
fig3, ax = plt.subplots(figsize=(20, 10), subplot_kw={"projection": "polar"})

# Specify offset
ax.set_theta_offset(OFFSET)

# Set limits for radial (y) axis. The negative lower bound creates the whole in the middle.
ax.set_ylim(-100, 10000)

# Remove all spines
ax.set_frame_on(False)

# Remove grid and tick marks
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_xticks([])
ax.set_yticks([])

# Add bars
ax.bar(
    ANGLES, VALUES, width=WIDTH, linewidth=2,
    color="#61a4b2", edgecolor="black"
)

# Add labels
add_labels(ANGLES, VALUES, LABELS, OFFSET, ax)


# In[ ]:


ANGLES = np.linspace(0, 2 * np.pi, len(education_frame), endpoint=False)
VALUES = education_frame["above_50K"].values
LABELS = education_frame["education-status"].values

# Determine the width of each bar. 
# The circumference is '2 * pi', so we divide that total width over the number of bars.
WIDTH = 2 * np.pi / len(VALUES)

# Determines where to place the first bar. 
# By default, matplotlib starts at 0 (the first bar is horizontal)
# but here we say we want to start at pi/2 (90 deg)
OFFSET = np.pi / 2

# Initialize Figure and Axis
plt.style.use('dark_background')
fig4, ax = plt.subplots(figsize=(20, 10), subplot_kw={"projection": "polar"})

# Specify offset
ax.set_theta_offset(OFFSET)

# Set limits for radial (y) axis. The negative lower bound creates the whole in the middle.
ax.set_ylim(-100, 2000)

# Remove all spines
ax.set_frame_on(False)

# Remove grid and tick marks
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_xticks([])
ax.set_yticks([])

# Add bars
ax.bar(
    ANGLES, VALUES, width=WIDTH, linewidth=2,
    color="#61a4b2", edgecolor="black"
)

# Add labels
add_labels(ANGLES, VALUES, LABELS, OFFSET, ax)


# In[ ]:





# In[ ]:

def make_data_circle_plot():
    with get_wide_container():
        st.subheader("Attribute Plot")
        st.markdown('''For our data, individuals with certain attributes tend to be more likely to have higher income.Here we attempt to visualize which of those attributes have an outsized impact on income.\n''')
        selector = st.selectbox(
            "Select Data Attribute type",
            ('Occupation', 'Race', 'Marital-Status', 'Education'), key = "circle_plot")
        if selector == 'Occupation':
            st.dataframe(jobbo)
            st.pyplot(fig)

        elif selector == 'Race':
            st.dataframe(race_frame)
            st.pyplot(fig2)

        elif selector == 'Marital-Status':
            st.dataframe(marital_frame)
            st.pyplot(fig3)

        elif selector == 'Education':
            st.dataframe(education_frame)
            st.pyplot(fig4)




