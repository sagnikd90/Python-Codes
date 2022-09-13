#!/usr/bin/env python
# coding: utf-8

# ### Introduction to Pandas:Generating distributions and Hypothesis Testing

# In this module, we will learn how to generate different types of probability distributions and also look at how to conduct hypothesis testing.

# ### Generating probability distributions

# Let us first look at how to generate binomial distributions. In the first example, we will generate a binomial distribution, which has only one trial and the probability of getting a zero is 1. The code to generate this distribution is as follows:

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


np.random.binomial(1,0.5)


# The previous output gives us a 0 which basically means that with half probability, if we run this trial only once, we will get a 0. We could have got 1 as the output, also with half probability.
# 
# Now we will run the same trial a 1000 times with the same probability of getting a 0 as half. The code to do so is as follows:

# In[4]:


np.random.binomial(1000,0.5)


# We see that out of 1000 trials, in 509 trials we will get a 0. If we now divide the outcome by 1000, we will get the probability of getting a 0 in each trial out of the 1000 trials. The code to do that is as follows:

# In[5]:


np.random.binomial(1000,0.5)/1000


# So we see that the probability of getting a 0 in each trial when we run 1000 trials is very close to 0.5. If we increase the number of trials further, we get:

# In[6]:


np.random.binomial(100000,0.5)/100000


# Thus it is clear that as we increase the number of trials, the probability of getting a 0 in each trials gets closer and closer to 0.5.

# We can generate weighted binomial trials too. Let us consider the following example, where we generate the number of times a tornado happens over 100000 days when the probability of a tornado happening is 0.0001/day. The code to generate this is as follows:

# In[7]:


prob_torn = 0.01/100

np.random.binomial(100000,prob_torn)


# So we see from the output that within 100000 days there will be tornadoes on 16 days.

# Now let us see, how many tornadoes we will get in a 1000000 days, when the tornadoes happen in back to back days. The code to compute this is as follows:

# In[8]:


chance_of_tornado = 0.01

tornado_events = np.random.binomial(1,chance_of_tornado,1000000)

two_days_in_a_row = 0
for j in range(1,len(tornado_events)-1):
    if tornado_events[j]==1 and tornado_events[j-1]==1:
        two_days_in_a_row+=1

print('{} tornadoes back to back in {} years'.format(two_days_in_a_row,1000000/365))        


# Now lets look at generating random normal distributions.
# 
# The following is the code to generate a random uniform distribution with mean 0 and standrad deviation 1:

# In[10]:


np.random.uniform(0,1)


# The next line of codes generates a random normal distribution:

# In[19]:


distribution=np.random.uniform(0,size=10000)


# Now we can compute the standard deviation of distrbution using the std command.

# In[20]:


np.std(distribution)


# We can also compute the Skewness,Kurtosis of any distribution. To do that, we need the scipy python package.

# In[23]:


import scipy.stats as stats


# In[24]:


stats.kurtosis(distribution)


# Similarly, we can compute the skewness of a distribution also.

# In[26]:


stats.skew(distribution)


# Using pandas, we cna generate Chi-Square distributions also.WHen generating a Chi-Square distribution, we need to specify the size and the degrees of freedom for the distribution. To generate a Chi-Square distribution, the following are the codes:

# In[36]:


chi_squared_df3 = np.random.chisquare(3,10000)


# In[37]:


chi_squared_df5 = np.random.chisquare(5,10000)


# In[38]:


chi_squared_df10 = np.random.chisquare(10,10000)


# Now we can use the matplotlib package of python to generate the distrbutions graphically and compare between them. To generate the distributions graphically, the following are the codes: 

# In[39]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
import matplotlib.pyplot as plt

output = plt.hist([chi_squared_df3,chi_squared_df5,chi_squared_df10],bins=100,histtype='step',label = ['3 degrees of freedom','5 degrees of freedom','10 degrees of freedom'])
plt.legend(loc='upper right')


# ### Hypothesis Testing
# 

# Now we will look at how to do simple hypothesis testing using Pandas. To do that, we first need to import a dataset.

# In[41]:


df=pd.read_csv(r"C:\Users\das90\OneDrive\Coursera courses\Michigan Learning Python\Course_1\course1_downloads\course1_downloads\grades.csv")

df.head()


# Now we create two variables which indicates whether a student submitted an assignment1 early or late.

# In[42]:


early = df[df['assignment1_submission']<='2015-31-12']
late = df[df['assignment1_submission']> '2015-31-12']

early.mean()


# In[43]:


late.mean()


# Now we can try and see if there is any statistically significant difference between the means of the grades for the early and the late submissions.

# In[45]:


from scipy import stats


# Let us first check if there is any significant difference between the means for the assignments.

# In[47]:


stats.ttest_ind(early['assignment1_grade'],late['assignment1_grade'])


# In[48]:


stats.ttest_ind(early['assignment2_grade'],late['assignment2_grade'])


# In[49]:


stats.ttest_ind(early['assignment3_grade'],late['assignment3_grade'])


# In[50]:


stats.ttest_ind(early['assignment4_grade'],late['assignment4_grade'])


# In[ ]:




