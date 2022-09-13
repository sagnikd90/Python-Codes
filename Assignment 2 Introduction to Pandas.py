#!/usr/bin/env python
# coding: utf-8

# 
# Assignment 2: Pnadas Introduction
# 
# The following codes loads the Olympics Gold Medalist dataset collected from Wikipedia.

# In[16]:


import pandas as pd
import numpy as np


# In[17]:


df = pd.read_csv(r"C:\Users\das90\OneDrive\Coursera courses\Michigan Learning Python\Course_1\course1_downloads\course1_downloads\olympics.csv", index_col = 0, skiprows = 1)

for col in df.columns:
  if col[:2]=='01':
      df.rename(columns={col:'Gold'+col[4:]}, inplace=True)
  if col[:2]=='02':
      df.rename(columns={col:'Silver'+col[4:]}, inplace=True)
  if col[:2]=='03':
      df.rename(columns={col:'Bronze'+col[4:]}, inplace=True)
  if col[:1]=='№':
      df.rename(columns={col:'#'+col[1:]}, inplace=True)

names_ids = df.index.str.split('\s\(') 

df.index = names_ids.str[0]  
df['ID'] = names_ids.str[1].str[:3] 

df = df.drop('Totals')
df.head()  


# Question 0) Example
# 
# The following code returns the medal columns for Afghanistan.

# In[18]:


def answer_zero():
    return df.iloc[0]

answer_zero()


# Question 1) Which country won the most gold medals in the Summer Games?

# In[20]:


def answer_one():
    return df['Gold'].idxmax()
answer_one()


# Question 2) Which country had the biggest difference between their summer and winter gold medal counts?

# In[21]:


def answer_two():
    return ((df['Gold'] - df['Gold.1']).idxmax())

answer_two()


# Which country has the biggest difference between their summer gold medal counts and winter gold medal counts relative to their total gold medal count?

# In[22]:


def answer_three():
    only_gold = df.where((df['Gold'] > 0) & (df['Gold.1'] > 0))
    only_gold = only_gold.dropna()
    return (abs((only_gold['Gold'] - only_gold['Gold.1']) / only_gold['Gold.2'])).idxmax()

answer_three()


# Question 4) Write a function that creates a Series called "Points" which is a weighted value where each gold medal (`Gold.2`) counts for 3 points, silver medals (`Silver.2`) for 2 points, and bronze medals (`Bronze.2`) for 1 point. The function should return only the column (a Series object) which you created.

# In[23]:


def answer_four():
    df['Points'] = (df['Gold.2'] * 3 + df['Silver.2'] * 2 + df['Bronze.2'] * 1)
    return df['Points']
                       
answer_four()


# For the next set of questions, we will be using the US census data. 
# 
# We will load the census data as census_df into python.

# In[24]:


censusdf = pd.read_csv(r"C:\Users\das90\OneDrive\Coursera courses\Michigan Learning Python\Course_1\course1_downloads\course1_downloads\census.csv")


# In[25]:


censusdf.head()


# Question 5) Which state has the most counties in it? (hint: consider the sumlevel key carefully! You'll need this for future questions too...)

# In[27]:


def answer_five():
    newdf = censusdf[censusdf['SUMLEV'] == 50]
    return newdf.groupby('STNAME').count()['SUMLEV'].idxmax()

answer_five()


# Question 6) Only looking at the three most populous counties for each state, what are the three most populous states (in order of highest population to lowest population)? Use `CENSUS2010POP`.
# 

# In[28]:


def answer_six():
    newdf = censusdf[censusdf['SUMLEV'] == 50]
    most_populous_counties = newdf.sort_values('CENSUS2010POP', ascending=False).groupby('STNAME').head(3)
    return most_populous_counties.groupby('STNAME').sum().sort_values('CENSUS2010POP', ascending=False).head(3).index.tolist()

answer_six()


# Question 7) Which county has had the largest absolute change in population within the period 2010-2015? (Hint: population values are stored in columns POPESTIMATE2010 through POPESTIMATE2015, you need to consider all six columns.)

# In[32]:


def answer_seven():
    df=censusdf[censusdf['SUMLEV'] == 50]
    df['STDEV'] = df[['POPESTIMATE2010','POPESTIMATE2011','POPESTIMATE2012',
                      'POPESTIMATE2013','POPESTIMATE2014','POPESTIMATE2015']].std(axis=1)
    return df.loc[df['STDEV'].idxmax()]['CTYNAME']
answer_seven()


# Question 8) Create a query that finds the counties that belong to regions 1 or 2, whose name starts with 'Washington', and whose POPESTIMATE2015 was greater than their POPESTIMATE 2014.

# In[33]:


def answer_eight():
    counties = censusdf[censusdf['SUMLEV'] == 50]
    region = counties[(counties['REGION'] == 1) | (counties['REGION'] == 2)]
    washington = region[region['CTYNAME'].str.startswith("Washington")]
    grew = washington[washington['POPESTIMATE2015'] > washington['POPESTIMATE2014']]
    return grew[['STNAME', 'CTYNAME']]

answer_eight()


# In[ ]:




