#!/usr/bin/env python
# coding: utf-8

# # Merging Data frames in Python

# In this module, we will learn, how to merge dataframes in python using pandas. But first, let us start with an easy example where we want to add a new variable to a small dataframe such that each row of the dataframe is assigned a different value for the variable. 
# 
# Since this is a small dataframe, we can simply define the new variable using the correct idnexing and a new variable with an unique value for each row of data will be generated.
# 
# In order to see how this is done, let us create a simple dataset. The following codes creates a simple dataset.

# In[4]:


import pandas as pd

df = pd.DataFrame([{'Name': 'Lionel Messi', 'Country': 'Argentina', 'Club': 'Barcelona', 'Goals': 756}, 
                   {'Name': 'Cristiano Ronaldo', 'Country': 'Portugal', 'Club': 'Juventus', 'Goals': 730}, 
                   {'Name': 'Neymar', 'Country': 'Brazil', 'Club': 'PSG', 'Goals': 347}],
                   index = ['Player 1', 'Player 2', 'Player 3'])


df


# Now suppose we want to add another column to this dataframe mentioning the old clubs of the players they used to play in before the current club. We would simple create a new variable called old club and assign values to the variable corresponding to each player.

# In[5]:


df['Old Club'] = ['Newell Old Boys', 'Real Madrid', 'Barcelona']
df


# When we are assigning values to each observation through a new variable, if we don't know the value of the variable for an observation, we can simply use None argument as done in the following example. 

# In[7]:


df['Balon D` Or'] = ['6','5',None]
df


# We can reset the index and use the series index to assign values to each observation using a new variable. The following codes does just that:

# In[9]:


adf = df.reset_index()
adf['UCL'] = pd.Series({0:'Four', 1:'Four'})
adf


# When we are dealing with bigger datasets with a lot more observations, it is not possible to create new variables and assign values to each variable individually. Especially, when we are trying to merge two datasets, we are merging multiple observations together with the help of an index variable, which may or may not have the same number of observations. Let us build a more complex dataset. The following commands helps us to build a more complex dataset:

# In[11]:


country_df = pd.DataFrame([{'Name': 'Lionel Messi', 'Country':'Argentina'}, 
                        {'Name': 'Andres Iniesta', 'Country':'Spain'},
                        {'Name': 'Xavi Hernandez', 'Country':'Spain'},
                       {'Name': 'Paulo Dybala', 'Country':'Argentina'}])
country_df = country_df.set_index('Name')

club_df = pd.DataFrame([{'Name': 'Lionel Messi', 'Club': 'Barcelona'},
                       {'Name': 'Andres Iniesta', 'Club':'Barcelona'},
                       {'Name': 'Neymar Jr.', 'Club':'PSG'},
                       {'Name':'Sergio Busquets', 'Club': 'Barcelona'}])
club_df = club_df.set_index('Name')

print(country_df.head())
print()
print(club_df.head())


# Now there are multiple ways in which we can merge the two datasets. Both the datasets have the common index called 'Name'. We can merge the datasets using that common index 'Name'. 
# 
# One way to merge is to merge both the datasets so that it includes all the variables and all the observations. This is analogous to finding the union of two sets. The following codes does just that, merge the whole datasets irrespective of the commanlity between the datasets.
# 

# In[13]:


pd.merge(country_df,club_df, how='outer', left_index = True, right_index = True)


# As we can see from the above output, variables across the datasets which does not have commonality or are missing from any one of the datasets, are assigned a NaN value by python automatically. For instance, Neymar Jr. does not feature in the first dataset and hence is assigned a value of NaN for the variable country. Similarly, Paulo Dybala does not feature in the club dataset and is hence assigned a value of NaN for the variable club.
# 
# Now suppose we want to merge the datasets in such a way that only the common observations to both the datasets remain and all the other observations are excluded. This is analogous to taking an intersection of two sets in mathematics. The following command does just that:

# In[14]:


pd.merge(country_df,club_df,how = 'inner', left_index=True,right_index=True)


# So the output correctly prints the details of the two players, Messi and Iniesta, who are the only two common observations across the two datasets. 

# Now we want to merge the country_df dataset with club_df irrespective of the fact whether those who feature on the country_df dataset, feature in the club_df dataset. The following commands does just that:

# In[15]:


pd.merge(country_df,club_df,how='left',left_index=True,right_index=True)


# Similarly, if we want to merge the country_df dataset and the club_df dataset irrespective of the fact whether those who feature on the club_df dataset feature on the country_df dataset. The following commands does just that:

# In[16]:


pd.merge(country_df,club_df,how='right',left_index=True,right_index=True)


# Now we want to see what happens when we try to merge two datasets which have a discrepancy for one variable over the two datasets.Let us rebuild the previous dataset by introducing this discrepancy.

# In[17]:


country_df = pd.DataFrame([{'Name': 'Lionel Messi', 'Country':'Argentina', 'Location':'Rosario'}, 
                        {'Name': 'Andres Iniesta', 'Country':'Spain', 'Location':'Seville'},
                        {'Name': 'Xavi Hernandez', 'Country':'Spain', 'Location':'Barcelona'},
                       {'Name': 'Paulo Dybala', 'Country':'Argentina', 'Location':'Buenos Aires'}])
country_df = country_df.set_index('Name')

club_df = pd.DataFrame([{'Name': 'Lionel Messi', 'Club': 'Barcelona','Location':'Barcelona'},
                       {'Name': 'Andres Iniesta', 'Club':'Barcelona','Location':'Barcelona'},
                       {'Name': 'Neymar Jr.', 'Club':'PSG', 'Location':'Paris'},
                       {'Name':'Paolo Dybala', 'Club': 'Juventus', 'Location':'Turin'}])
club_df = club_df.set_index('Name')


# In[18]:


pd.merge(country_df,club_df,how='left', left_on='Name', right_on='Name')


# In the previous command, we used the column name for indexing using "left_on" and "right_on", rather than using a normal index.
# 
# Lastly, let us build a dataset where the names of the players are divided into two columns, First name and Last name. The following commands do just that:

# In[19]:


country_df = pd.DataFrame([{'First Name': 'Lionel','Last Name':'Messi', 'Country':'Argentina', 'Location':'Rosario'}, 
                        {'First Name': 'Andres','Last Name':'Iniesta', 'Country':'Spain', 'Location':'Seville'},
                        {'First Name': 'Xavi','Last Name':'Hernandez', 'Country':'Spain', 'Location':'Barcelona'},
                       {'First Name': 'Paolo','Last Name':'Dybala', 'Country':'Argentina', 'Location':'Buenos Aires'}])

club_df = pd.DataFrame([{'First Name': 'Lionel','Last Name':'Messi', 'Club': 'Barcelona','Location':'Barcelona'},
                       {'First Name': 'Andres','Last Name':'Iniesta', 'Club':'Barcelona','Location':'Barcelona'},
                       {'First Name': 'Neymar','Last Name':'Junior', 'Club':'PSG', 'Location':'Paris'},
                       {'First Name': 'Paolo','Last Name':'Dybala', 'Club': 'Juventus', 'Location':'Turin'}])


# Now we want to merge both by first name and last name. The following commands helps us to do so:

# In[20]:


pd.merge(country_df,club_df,how='inner', left_on=['First Name', 'Last Name'], right_on=['First Name', 'Last Name'])


# #Making codes more pandorable:Using panda idioms

# In[21]:


import pandas as pd


# We want to import the American Census data which is a CSV file. The next line of code does just that

# In[22]:


df = pd.read_csv(r"C:\Users\das90\OneDrive\Coursera courses\Michigan Learning Python\Course_1\course1_downloads\course1_downloads\census.csv")
df.head()


# In the above dataset, we can see that the dataset has an id variable called SUMLEV which takes the value 40 when referring to a State itself and takes the value 50 for the states when it includes the counties of the state also. We want to keep the counties only and exclude the observations for which SUMLEV==40. We can do that using the following line of code.

# In[26]:


df=df[df['SUMLEV']==50]


# Next up, we want to index the dataset by the State name and the county name. That is achieved by running the following codes:
# 

# In[27]:


df.set_index(['STNAME', 'CTYNAME'])


# Next up, we want to rename some of the column variables. Suppose we want to rename the variable ESTIMATESBASE2010 as Estimates Base 2010. The following line of code achieves this goal:

# In[28]:


df.rename(columns={'ESTIMATESBASE2010': 'Estimates Base 2010'})


# Now we can run the whole line of codes written above in a much easier way and thus making our codes much pandorable. To make the above written codes more pandorable, we will write the following line of codes.

# In[32]:


df = pd.read_csv(r"C:\Users\das90\OneDrive\Coursera courses\Michigan Learning Python\Course_1\course1_downloads\course1_downloads\census.csv")

(df.where(df['SUMLEV']==50)
       .dropna()
       .set_index(['STNAME','CTYNAME'])
       .rename(columns={'ESTIMATESBASE2010':'Estimates Base 2010'}))


# Thus from the above line of pandorable codes, we can clearly see that we have achieved the same result as we had done while writing down each line of code separetely. 

# Suppose we want to summarize or iterate a function over all the rows of a column or multiple column i.e. iterate over all the rows for a particular variable. This can be done in multiple ways. At first we will look at how to do it in general and then we will look at how to make the codes more pandorable using the lambda function of python. 
# 
# Let us consider the census dataset and for each columns of the population estimate, we want to find the min and max for each county of each state for the population estimates.
# 
# Firstly, we want to create a new series which will have the min and max across the population estimates of all the counties for all the states. Codes whih helps to do this is as follows:

# In[34]:


import numpy as np

def min_max(row):
    data = row[['POPESTIMATE2010',
                'POPESTIMATE2011',
                'POPESTIMATE2012',
                'POPESTIMATE2013',
                'POPESTIMATE2014',
                'POPESTIMATE2015']]
    return pd.Series({'min': np.min(data), 'max': np.max(data)})

df.apply(min_max, axis=1)


# The above set of codes return a new series with the newly constructed variables min and max. However, if we want to add the newly computed columns to the original dataset, we use the following set of codes:

# In[35]:


def min_max(row):
    data = row[['POPESTIMATE2010',
               'POPESTIMATE2011',
               'POPESTIMATE2012',
               'POPESTIMATE2013',
               'POPESTIMATE2014',
               'POPESTIMATE2015']]
    row['max'] = np.max(data)
    row['min'] = np.min(data)
    
    return row

df.apply(min_max,axis=1)


# Now suppose we want to perform the same analysis. but now we would like to use pandorable code so that the readability of the code is better. The following set of codes performs the same operation but the codes are made pandorable using the pandas lambda function.

# In[36]:


rows = ['POPESTIMATE2010',
       'POPESTIMATE2011',
       'POPESTIMATE2012',
       'POPESTIMATE2013',
       'POPESTIMATE2014',
       'POPESTIMATE2015']
df.apply(lambda x:np.max(x[rows]), axis=1)


# ### Using the group_by() function in Pandas

# Now suppose that we want to find aggregate measures a column or multiple columns. This also can be done in multiple ways. We can loop over every observation of the column or loop over a chosen index or we can simply use the groupby function of pandas andfind the aggregate measures.
# 
# Let us first start by looping over a chosen index to find the aggregate measure of a variable. In order to do so, let us reload the American Census data.

# In[38]:


import pandas as pd

df = pd.read_csv(r"C:\Users\das90\OneDrive\Coursera courses\Michigan Learning Python\Course_1\course1_downloads\course1_downloads\census.csv")

df = df[df['SUMLEV']==50]

df.head()


# Let us consider the variable CENSUS2010POP. We want to find the aggreate of the variable CENSUS2010POP for each state, by iterating value of CENSUS2010POP over the counties for each state.
# 
# The code which performs this operation is as follows:

# In[39]:


for state in df['STNAME'].unique():
    avg = np.average(df.where(df['STNAME']==state).dropna()['CENSUS2010POP'])
    print('Counties in the state' + state + 'have an average population of' + str(avg))
    
    


# In the above set of codes, what we basically do is, find the average of the variable CENSUS2010POP by looping over all the states. First, we create a for loop, which loops over the variable 'STNAME' uniquely and implements the function np.average on the population variable CENSUS2010POP. 
# 
# The above result can also be achieved by using the groupby() function. The following are the codes when we apply the groupby() function:

# In[40]:


for group, frame in df.groupby('STNAME'):
    avg = np.average(frame['CENSUS2010POP'])
    print('Counties in the state of' + group + 'have an average population of' + str(avg))

    


# In the above two examples we have created a new series of data. It is not necessary to do so and we can get a new dataframe containing the average of the population variable for each state.
# 
# What we can do in this case is that, we can create a loop in a way such that it splits the dataset according to the mentioned index and then take the average for each of the states.
# 
# We first build a function that splits the dataset according to the statenames. The code to do so is as follows:

# In[44]:


df = df.set_index('STNAME')

def fun(item):
    if item[0]<'M':
        return 0
    if item[0]<'Q':
        return 1
    return 2
    
for group, frame in df.groupby(fun):
    print('There are ' + str(len(frame)) + ' records in group ' + str(group) + ' for processing.')


# In the abose set of codes, we first split the data by running a loop over the names of the States and divide the state by the initials of the state names. Then we use the group by function to count the length of the observations across the splitted groups. The code returns the number of observations across the 3 splitted groups.
# 
# Now we want to perform a split,apply and combine method. In this method, we want to first split the data, then apply a function to the split data and finally combine the results. 
# 
# Let us start simple. We first want to groupby the census data by the statename and then use the aggregate function find the mean of CENSUS2010POP variable. The following are the codes to perform this function.

# In[46]:


import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\das90\OneDrive\Coursera courses\Michigan Learning Python\Course_1\course1_downloads\course1_downloads\census.csv")

df = df[df['SUMLEV']==50]

df.groupby('STNAME').agg({'CENSUS2010POP':np.average})


# The groupby and agg method behaves differently when we are using it fro dataframe and when we are using it for a series.
# 
# For instance using the following codes, we take the census data and turn it into a series which gives us the aggregates average and sum of the single column CENSUS2010POP. We groupby the statename and use the level parameter. The following are the codes to perform this operation:

# In[61]:


import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\das90\OneDrive\Coursera courses\Michigan Learning Python\Course_1\course1_downloads\course1_downloads\census.csv")

df = df[df['SUMLEV']==50]


# In[ ]:


(df.set_index('STNAME').groupby(level=0)['CENSUS2010POP']
    .agg({'avg': np.average, 'sum': np.sum}))


# Running the above set of code is giving a specification error. Why I am not so sure.

# Suppose we want to find the average and the sum of the two variables, POPESTIMATE2010 and POPESTIMATE2011 and get a hierarchal dataframe in return. The following codes will do the desired operation:

# In[63]:


(df.set_index('STNAME').groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']
    .agg({'avg': np.average, 'sum': np.sum}))


# The agg is not working for some reason. It is giving a specification error.

# ### Scales in python

# In[77]:


import pandas as pd
import numpy as np

df = pd.DataFrame(['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D'],
                  index=['excellent', 'excellent', 'excellent', 'good', 'good', 'good', 'ok', 'ok', 'ok', 'poor', 'poor'])
df.rename(columns={0: 'Grades'}, inplace=True)
df


# In[78]:


df['Grades'].astype('category').head()


# As we can see from the output above that the Grades have been saved as a category variable. Now if we want to order the grades in a specific order, we will use the following set of codes:

# In[84]:


grades = df['Grades']



# ### Pivot Tables
# 

# In order to summarize data, pivot tables are very useful. While computing a pivot table, we can use a variable as an index, a variable we want to summarize and a variable for column.
# 
# Let us load the cars data into python and then construct a pivot table which gives the mean of the battery capacity of each car type over the years. For this we first need to load the data into python:

# In[85]:


import pandas as pd

df = pd.read_csv(r"C:\Users\das90\OneDrive\Coursera courses\Michigan Learning Python\Course_1\course1_downloads\course1_downloads\cars.csv")
df.head()


# As we can see from the table above, we have the year the first electric car was made by the company, the make of the car and the attributes of the car. Among these we want to make the YEAR as the index, the Make as the column and the battery capacity (KW) as the variable for which we want a summary statistics. The code for the operation is as follows:

# In[88]:


df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=np.mean)


# In order to get rid of the NaN values, we can use the fill_value=0.

# In[89]:


df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=np.mean, fill_value=0)


# Now suppose we want to aggregate multiple variables with the same index and columns. The following code does just that:

# In[91]:


df.pivot_table(values=['(kW)','(km)'], index='YEAR', columns='Make', aggfunc=np.mean, fill_value=0)


# We can also input more aggregate measures in the aggfunc attribute. In the following code, we do just that:

# In[92]:


df.pivot_table(values=['(kW)'], index='YEAR', columns='Make', aggfunc=[np.mean,np.min], fill_value=0, margins=True)


# ### Date functionality in pandas

# Pandas have basically 4 types of date and time classes:Timestamp, Datetime index, Period and Period Index.
# 
# If we are more concrened about a specific point in time, we use Timestamp. The following is an example of Timestamp:
# 

# #### Timestamp index

# In[93]:


import pandas as pd


# In[94]:


import numpy as np


# In[95]:


pd.Timestamp('26/05/2020 7:22PM')


# Time stamp deals with a specific point in time. When we are concerned about a span of time, we use the Period time class. The following is an example of period class.
# 

# #### Period

# In[98]:


pd.Period('2020')


# In[99]:


pd.Period('05/2020')


# In[100]:


pd.Period('26/05/2020')


# In the above three codes, we can see that when we use the Period class to specify time, mentioning only the month and year gives us a Month Year index. Mentioning the full date gives us a Day index.

# #### DatetimeIndex

# The index of a Timestamp is a DatetimeIndex. 
# 
# Let us create a data series using a timestamp. The following are the required codes:

# In[103]:


t1 = pd.Series(list('abcd'),[pd.Timestamp('26/05/2020'), pd.Timestamp('27/05/2020'), pd.Timestamp('28/10/2020'), pd.Timestamp('29/05/2020')])
t1


# In[104]:


type(t1.index)


# #### Period Index

# The index of Period is a PeriodIndex.
# 
# Let us create a data series using a PeriodIndex. The following are the required codes.

# In[106]:


t2 = pd.Series(list('efgh'), [pd.Period('26/05/2020'), pd.Period('27/05/2020'), pd.Period('28/05/2020'), pd.Period('29/05/2020')])
t2


# In[107]:


type(t2.index)


# Now let us look at how to convert any date to Datetime. 
# 
# Let us build a dataframe using a datetime as the index.

# In[110]:


d1 = ['26 June 2020', '30 July 2020', '17/06/2020', '2020-05-25']

ts3 = pd.DataFrame(np.random.randint(10,100,(4,2)), index=d1, columns=list('ab'))

ts3


# As we can see clearly from the table, the dates look very haphazard. We can use the to_datetime command to convert the dates to one specific class. The codes to so are as follows: 

# In[114]:


ts3.index = pd.to_datetime(ts3.index, dayfirst=True)
ts3


# #### Timedeltas

# Sometimes, when we are working with time series data, we are often concerned with finding the difference of a variable between two specific points of time or between two different period using the Timedeltas.

# In[116]:


pd.Timestamp('5/26/2020')-pd.Timestamp('5/30/2020')


# In[117]:


pd.Timestamp('5/26/2020 07:56AM') + pd.Timedelta('12D 3H')


# ### Working with dataframe

# When we are working with dataframe which has a DatetimeIndex, there are a few tricks that we can use. Let us see what those tricks are:
# 
# First let us create a range of datetimeIndex using the date_range() function.The code is below:

# In[136]:


dates = pd.date_range('05-26-2020', periods=100, freq='2W-SUN')


# Now we can create a dataframe using this datetimeIndex. The codes are the following:

# In[137]:


df = pd.DataFrame({'Count 1':100 + np.random.randint(-10,10,100).cumsum(), 'Count2':-200+np.random.randint(-20,20,100).cumsum()}, index=dates)
df


# We can check what day of the week a specific date is using the following command:

# In[138]:


df.index.day_name


# We can see the difference between the two dates using the df.diff() command:

# In[139]:


df.diff()


# Suppose we want to find the mean of the variables over the time period. To do this, we can use the df.resample command:

# In[140]:


df.resample('M').mean()


# We can even use partial string indexing for a particular year or a month. To partially index by a particular year we use the following command:

# In[141]:


df['2022']


# To partially index by a particular year and month, we can use the following command:

# In[143]:


df['2022-08']


# We can also index partially over a range of date. Suppose we want to index the data in such a way that we see all observations since 2025-08-03. The following code achieves this objective:

# In[144]:


df['2022-08':]


# Now suppose we want to change the frequency of the dates from bi weekly to daily. We can do that using the asfreq() command:

# In[145]:


df.asfreq('W', method='ffill')


# Now we want to plot the time series data. To plot the time series data, we need to import the matplotlib.pylot panda first.

# In[147]:


import matplotlib.pyplot as plt


# In[148]:


get_ipython().run_line_magic('matplotlib', 'inline')

df.plot()


# In[ ]:




