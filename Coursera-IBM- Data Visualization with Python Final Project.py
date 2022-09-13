#!/usr/bin/env python
# coding: utf-8

# # Data Visualization with Python:IBM Coursera- Final Project

# ### Question 1)

# A survey was conducted to gauge an audience interest in different data science topics, namely:
# 
# Big Data (Spark / Hadoop)
# 
# Data Analysis / Statistics
# 
# Data Journalism
# 
# Data Visualization
# 
# Deep Learning
# 
# Machine Learning
# 
# The participants had three options for each topic: Very Interested, Somewhat interested, and Not interested. 2,233 respondents completed the survey.
# 
# The survey results have been saved in a csv file and can be accessed through this link: https://cocl.us/datascience_survey_data.
# 
# If you examine the csv file, you will find that the first column represents the data science topics and the first row represents the choices for each topic.
# 
# Use the pandas read_csv method to read the csv file into a pandas dataframe.

# In order to read the data into a dataframe like the above, one way to do that is to use the index_col parameter in order to load the first column as the index of the dataframe. Here is the documentation on the pandas read_csv method: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
# 
# Once you have succeeded in creating the above dataframe, please upload a screenshot of your dataframe with the actual numbers. (5 marks)

# In[58]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[59]:


df_ds = pd.read_csv(r"C:\Users\das90\OneDrive\Coursera courses\IBM Python\Data Visualization\Matplotlib\Topic_Survey_Assignment.csv", index_col=0)
df_ds.head()


# Next, we calculate a column of total respondendts for each topic:

# In[60]:


df_ds["Total"] = df_ds.sum(axis=1)
df_ds.head()


# ## Question 2)

# Use the artist layer of Matplotlib to replicate the bar chart below to visualize the percentage of the respondents' interest in the different data science topics surveyed.

# So we first need to compute the percentages for each type of data science course:

# In[61]:


df_ds["Very interested"] = np.round(df_ds["Very interested"]/df_ds["Total"],2)
df_ds["Somewhat interested"] = np.round(df_ds["Somewhat interested"]/df_ds["Total"],2)
df_ds["Not interested"] = np.round(df_ds["Not interested"]/df_ds["Total"],2)
df_ds.head()


# Next, we sort the data by very interested:

# In[62]:


df_ds.sort_values(by="Very interested", ascending=False).iloc[0].drop("Total")


# Now we plot the bar charts:

# In[63]:


import matplotlib as mpl
import matplotlib.pyplot as plt


# Setting the matplotlib style:

# In[64]:


mpl.style.use("ggplot")


# Using the artist layer to plot the bars:

# In[65]:


columns=['Very interested', 'Somewhat interested', 'Not interested']

ax = df_ds[columns].sort_values(by='Very interested', ascending=False).plot(kind="bar", 
                                                                            figsize=(20,8), 
                                                                            width=0.8,
                                                                            color = ['#5cb85c', '#5bc0de', '#d9534f'])
                                                                                                                                                                              
ax.set_ylim(0,1)
ax.set_xlim(-0.6,5.7)
ax.set_title('Percentage of Respondents\' Interest in Data Science Areas', fontdict = {'fontsize' : 16}) # change the font size of the title
ax.legend(fontsize=14, facecolor='white') # change the font size of the legend
ax.tick_params(axis="x", labelsize=14) # change the font size of the x axix label

# remove the left, top, and right borders; make sure the color of x axix labels is black

ax.set_facecolor('white') # change background to white
ax.set_yticklabels([]) # turn off y ticks
ax.axhline(0, color='black') # draw an x axix line
ax.tick_params(axis='x', colors='black') # make sure the color of x axix labels is black

# Adds the percentages numbers over the bars

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.0%}'.format(height), (x, y + height + 0.01))


# ## Question 2)
# 

# In the final lab, we created a map with markers to explore crime rate in San Francisco, California. In this question, you are required to create a Choropleth map to visualize crime in San Francisco.
# 
# Before you are ready to start building the map, let's restructure the data so that it is in the right format for the Choropleth map. Essentially, you will need to create a dataframe that lists each neighborhood in San Francisco along with the corresponding total number of crimes.
# 
# Based on the San Francisco crime dataset, you will find that San Francisco consists of 10 main neighborhoods, namely:
# 
# Central,
# 
# Southern,
# 
# Bayview,
# 
# Mission,
# 
# Park,
# 
# Richmond,
# 
# Ingleside,
# 
# Taraval,
# 
# Northern, and,
# 
# Tenderloin.
# 
# Convert the San Francisco dataset, which you can also find here, https://cocl.us/sanfran_crime_dataset, into a pandas dataframe, like the one shown below, that represents the total number of crimes in each neighborhood.

# In[79]:


import pandas as pd
import numpy as np


# First we load the data into python:

# In[80]:


df_sf = pd.read_csv(r"C:\Users\das90\OneDrive\Coursera courses\IBM Python\Data Visualization\Matplotlib\Police_Department_Incidents_-_Previous_Year__2016_.csv")
df_sf.head()


# Next, we count the crime for each district:

# In[81]:


sf=df_sf.groupby(["PdDistrict"], sort=0).apply(lambda g: pd.Series({                                                                 'Count': g.IncidntNum.count(),                                                                 }))


# Resetting the index:

# In[82]:


sf.reset_index(level=0,inplace=True)


# In[83]:


sf


# Renaming the columns:

# In[88]:


sf=sf.rename(columns={"PdDistrict":"Neighborhood"})


# In[89]:


sf


# Now you should be ready to proceed with creating the Choropleth map.
# 
# As you learned in the Choropleth maps lab, you will need a GeoJSON file that marks the boundaries of the different neighborhoods in San Francisco. In order to save you the hassle of looking for the right file, I already downloaded it for you and I am making it available via this link: https://cocl.us/sanfran_geojson.
# 
# For the map, make sure that:
# 
# it is centred around San Francisco,
# 
# you use a zoom level of 12,
# 
# you use fill_color = 'YlOrRd',
# 
# you define fill_opacity = 0.7,
# 
# you define line_opacity=0.2, and,
# 
# you define a legend and use the default threshold scale

# First we import folium:

# In[91]:


import folium


# Importing the Json file:

# In[94]:


sf_geo = r"C:\Users\das90\OneDrive\Coursera courses\IBM Python\Data Visualization\Matplotlib\san-francisco.geojson"


# Defining the latitude and longitude of San Francisco

# In[95]:


latitude = 37.77
longitude = -122.42


# Defining the map of San Francisco using folium:

# In[96]:


sanfran_map = folium.Map(location=[latitude,longitude], zoom_start=12)
sanfran_map


# Generating the chloropeth map using the total number of crimes of each neighborhood of San Francisco:

# In[98]:


sanfran_map.choropleth(
    geo_data=sf_geo,
    data=sf,
    columns=['Neighborhood', 'Count'],
    key_on='feature.properties.DISTRICT',
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Crime Numbers of Each Neighborhood in San Francisco'
)
sanfran_map


# In[ ]:




