#!/usr/bin/env python
# coding: utf-8

# ## Data Visualization in Python using Matplotlib

# Before we start with the visualizations, we need to download the data. We will be using the immigration data to USA from United Nations website. 
# 
# Let us first download the data:

# In[1]:


import pandas as pd
import numpy as np


# Since the data is in excel format, we need to download the module of pandas which allows us to read excel data files. The following line of codes helps us to do that:

# In[2]:


get_ipython().system('conda install -c anaconda xlrd --yes')


# Now we load the data into Python:

# In[4]:


df_can = pd.read_excel(r"C:\Users\das90\OneDrive\Coursera courses\IBM Python\Data Visualization\Matplotlib\UN_MigFlow_All_CountryFiles\Canada.xlsx",
                      skiprows=range(20),
                      skipfooter=2,
                      sheet_name="Canada by Citizenship")
df_can.head()


# Now we explore the dataset a bit:

# In[5]:


df_can.info()


# To check the column names of the dataset and the indices as a list, we use the following code:

# In[6]:


df_can.columns.tolist()


# In[7]:


df_can.index.tolist()


# To check the shape of the dataset, we use the following code:

# In[9]:


df_can.shape


# Now lets rename some of the columns so that they become more understandable:

# In[13]:


df_can.rename(columns={"OdName":"Country","AreaName":"Continent","RegName":"Region"}, inplace=True)
df_can.head(2)


# Now we remove some of the variables that we don't need like Coverage, AREA, REG. We use the following line of commands to do so:

# In[14]:


df_can.drop(["Coverage", "AREA", "REG", "DEV"], axis=1, inplace=True)
df_can.head()


# We create a new variable which represents the total immigration for each country for the specified period and add that variable to the dataset:

# In[15]:


df_can["Total"] = df_can.sum(axis=1,numeric_only=None)


# In[16]:


df_can["Total"]


# We can look at the descriptive statistics of the data, by simply using the describe command:

# In[17]:


df_can.describe()


# Let us now look at the immigrations for some countries separately. We need to do slicing in order to do that. For instance, if we want to look at the immigration to Canda from some of the European Countries, we do the following slicing:
# 
# First we need to set the variable country as the index for the dataset:

# In[20]:


df_can.set_index("Country", inplace=True)
df_can.head()


# Now using the index, we can do different sorts of slicing:

# In[21]:


df_can.loc["Germany", :]


# Column names that are integers (such as the years) might introduce some confusion. For example, when we are referencing the year 2013, one might confuse that when the 2013th positional index.
# 
# To avoid this ambuigity, let's convert the column names into strings: '1980' to '2013'.

# In[22]:


df_can.columns=list(map(str,df_can.columns))


# In[23]:


years = list(map(str,range(1980,2014)))


# We can use the filtering creiteria to filter the dataset according to specific values of the variables. An example of filtering in given below:

# In[24]:


df_can[(df_can['Continent']=='Asia') & (df_can['Region']=='Southern Asia')]


# In[19]:


df_can[(df_can["Continent"]=="Europe") & (df_can["Region"]=="Western Europe")]


# ## Visualizing the Data with Matplotlib

# First we import the backend for matplotlib:

# In[25]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[26]:


import matplotlib as mlb
import matplotlib.pyplot as plt


# ### Line Plots

# Pandas has an inbuilt implementation of matplotlib that we can use. It is given by the .plot command. We can plot histograms and line plots using the .plot command. 
# 
# To see how line plots work, let us consider the country Haiti and see how the immigration to Canada from Haiti has changed over the years. The following line of codes helps us to plot the immigration from Haiti to Canada over the years:
# 
# First, we slice the data and select the country Haiti and the relevant number of years:

# In[27]:


Haiti = df_can.loc["Haiti", years]
Haiti.head()


# The simpler way to make this plot is by writing the following code:

# But first, let us choose a plotting style. We can choose a number of plotting style. For this particular example, we choose a ggplot like plotting style.
# 
# We set the plotting style using the following commands:

# In[28]:


print(plt.style.available)
mlb.style.use(["ggplot"])


# In[29]:


Haiti.plot()


# As we can see from the line plot, there was a very sudden spike in the number of immigrants from Haiti in the year 2011 due to the massive earthquake.

# The above plot is missing the axis titles and the graph itself does not have a title. To add the titles to the plot, we use the following set of commands:
# 
# First we change the index values of Haiti to type int:

# In[30]:


Haiti.index = Haiti.index.map(int)

Haiti.plot(kind="line")
plt.title("Immigration to Canada")
plt.ylabel("Total Number of Immigrants")
plt.xlabel("Years")


# As mentioned earlier, there was a spike in immigration due to the earthquake. Now we want to mark that point in the graph itself. We could write the following set of commands:

# In[31]:


Haiti.index = Haiti.index.map(int)

Haiti.plot(kind="line")
plt.title("Immigration to Canada")
plt.ylabel("Total Number of Immigrants")
plt.xlabel("Years")
plt.text(2000,6000,"2010 Earthquake")


# Now we want to plot the immigration from some of the Western European countries to Canada. To do that, we first need to slice the data according to the countries we are concerned with:

# In[32]:


df_we = df_can.loc[["Germany", "France", "Italy"], years]
df_we.head()


# Now if we use the simple plotting commands, we get the following graph:

# In[33]:


df_we.plot(kind="line")


# This does not look right. This is because the dataset df_we is in wide format. We need to convert it to a long format so that we can have the years on the x-axis and the number of immigrants on the y-axis. To convert to a long format, we simply transpose df_we. We use the following commands:

# In[34]:


df_we = df_we.transpose()
df_we.head()


# Now we have the data in the long format with the variable years as the index. We can now use the plot commands to plot the graph:

# In[35]:


df_we.index = df_we.index.map(int)
df_we.plot(kind="line")
plt.title("Immigration to Canada from Western Europe")
plt.xlabel("Years")
plt.ylabel("Total Number of Immigrants")


# Now we will compare the trend of the top 5 countries which had the highest number of immigration to Canada. 
# 
# To do this, we first need to arrange the dataset in such a way that only the top 5 countries are left. In order to achieve this, we first sort the dataset according to the total number of immigrants. The command to sort the dataset is given by:

# In[36]:


df_can.sort_values(by = "Total", ascending=False, axis=0, inplace=True)
df_top_5 = df_can.head(5)
df_top_5.head()


# Now we only require the countries and the years from the dataset. We slice and take out the countires and store it in a new dataframe:

# In[37]:


df_top_5 = df_top_5.loc[["India", "China", "United Kingdom of Great Britain and Northern Ireland", "Philippines", "Pakistan"], years]
df_top_5.head()


# Now since the data is in the wide format, to plot, we need to convert it to a long format. The following command does so:

# In[38]:


df_top_5 = df_top_5.transpose()
df_top_5


# Now we can plot the data:

# In[39]:


df_top_5.index = df_top_5.index.map(int)
df_top_5.plot(kind="line", figsize=(14,8))
plt.title("Immigration trend from the top 5 countries")
plt.xlabel("Years")
plt.ylabel("Total Number of Immigrants")
plt.legend(framealpha=1,frameon=True)


# ## Plotting Area Plots

# In the previous plot, we plotted the trends in immigration for the top 5 countries. We used a line plot to plot the trend. In place of line plots, we can also use area plots, which are a nice way to represent trends over time. We use the df_top_5 dataset.

# In[40]:


df_top_5.head()


# Let us first plot an unstacked area plot:

# In[41]:


df_top_5.index=df_top_5.index.map(int)
df_top_5.plot(kind="area",
             stacked=False,
             figsize=(20,10)
             )
plt.title("Immigration trend of top 5 countries")
plt.xlabel("Years")
plt.ylabel("Number of Immigrants")


# The above unstacked plot has a default transperency level set at 0.5. We can control the transparency level using the alpha parameter. Suppose we want the transparency level to be 0.3. Then we write the following code:

# In[42]:


df_top_5.index=df_top_5.index.map(int)
df_top_5.plot(kind="area",
             stacked=False,
             alpha=0.3, 
             figsize=(20,10)
             )
plt.title("Immigration trend of top 5 countries")
plt.xlabel("Years")
plt.ylabel("Number of Immigrants")


# Now if we want a stacked area plot, we run the following command:

# In[43]:


df_top_5.index=df_top_5.index.map(int)
df_top_5.plot(kind="area",
             stacked=True,
             figsize=(20,10)
             )
plt.title("Immigration trend of top 5 countries")
plt.xlabel("Years")
plt.ylabel("Number of Immigrants")


# There are two ways to plot in Python:
# 
# (1) Using the scripting layer
# 
# (2) Using the artist layer
# 
# Till now, we have been using the scripting layer to plot the graphs. Let us now look at how to plot using the artist layer. In the following three blocks of commands, we will plot the same plots plotted above using the artist layer.

# ### Area Plot using Artist Layer

# #### Plot 1: Simple Unstacked Area plot of top 5 countries with default transparency

# In[44]:


df_top_5.index = df_top_5.index.map(int)

ax = df_top_5.plot(kind="area",
                  stacked=False,
                  figsize=(20,10))
ax.set_title("Immigration trend from top 5 countries")
ax.set_xlabel("Years")
ax.set_ylabel("Number of Immigrants")


# #### Plot 2: Simple Unstacked Area plot of top 5 countries with 0.3 transparency

# In[45]:


df_top_5.index = df_top_5.index.map(int)

ax = df_top_5.plot(kind="area",
                  stacked=False,
                  alpha = 0.3, 
                  figsize=(20,10))
ax.set_title("Immigration trend from top 5 countries")
ax.set_xlabel("Years")
ax.set_ylabel("Number of Immigrants")


# #### Plot 3: Simple stacked Area plot of top 5 countries
# 

# In[46]:


df_top_5.index = df_top_5.index.map(int)

ax = df_top_5.plot(kind="area",
                  stacked=True, 
                  figsize=(20,10))
ax.set_title("Immigration trend from top 5 countries")
ax.set_xlabel("Years")
ax.set_ylabel("Number of Immigrants")


# Now using the artist layer, we are going to plot the immigration trend in the countries with the least number of immigrations.
# 
# For that we first need to slice out those respective countries from the dataste:

# In[47]:


df_bot_5=df_can.tail(5)
df_bot_5.head()


# In the next step, we take on the rows and the years and store it in the dataset. Next, we store the data in the long format.

# In[48]:


df_bot_5 = df_bot_5.loc[["San Marino", "New Caledonia", "Marshall Islands", "Western Sahara", "Palau"], years]
df_bot_5 = df_bot_5.transpose()
df_bot_5.head()


# Now we will plot the area plots, using the artist layer:
# 
# #### Plot 1: Simple Unstacked Area Plot with Default Transparency

# In[49]:


df_bot_5.index = df_bot_5.index.map(int)
ax = df_bot_5.plot(kind="area",
                  stacked=False,
                  figsize=(20,10))
ax.set_title("Immigration from the bottom 5 countries")
ax.set_xlabel("Years")
ax.set_ylabel("Total Number of Immigrants")


# #### Plot 2: Simple Unstacked Area Plot with 0.3 Transparency

# In[50]:


df_bot_5.index = df_bot_5.index.map(int)
ax = df_bot_5.plot(kind="area",
                  stacked=False,
                  alpha=0.3, 
                  figsize=(20,10))
ax.set_title("Immigration from the bottom 5 countries")
ax.set_xlabel("Years")
ax.set_ylabel("Total Number of Immigrants")


# #### Plot 3: Simple Stacked Area Plot

# In[51]:


df_bot_5.index = df_bot_5.index.map(int)
ax = df_bot_5.plot(kind="area",
                  stacked=True, 
                  figsize=(20,10))
ax.set_title("Immigration from the bottom 5 countries")
ax.set_xlabel("Years")
ax.set_ylabel("Total Number of Immigrants")


# ## Plotting Histograms

# A histogram is a way of representing the frequency distribution of numeric dataset. The way it works is it partitions the x-axis into bins, assigns each data point in our dataset to a bin, and then counts the number of data points that have been assigned to each bin. So the y-axis is the frequency or the number of data points in each bin. Note that we can change the bin size and usually one needs to tweak it so that the distribution is displayed nicely.

# Before we plot the hsitograms, we use the numpy hist option to define the bin sizes and the frequency.
# 
# Suppose we want to plot the histogram of the total immigration for the year 2013.  Let us first check how the data looks for the year 2013:

# In[52]:


df_can["2013"].head(10)


# Now let us set the bin widths using the numpy histogram function:

# In[53]:


count, bin_edges = np.histogram(df_can["2013"])
bin_edges


# We use the scripting layer to plot the graphs:

# In[54]:


df_can["2013"].plot(kind="hist", figsize=(8,5), xticks=bin_edges)
plt.title("Histogram of Immigration from 195 countries in 2013")
plt.ylabel("Number of countries")
plt.xlabel("Number of immigrants")


# Next, let us find the distribution of immigration for the 3 Scandinavian countries, Denmark, Norway and Sweden. To do that, we first need to slice the dataset and save the data for these 3 countires in a separate dataset. The codes to do that are as follows:

# In[55]:


df_scn = df_can.loc[["Denmark", "Sweden", "Norway"], years]
df_scn.head()


# In[56]:


df_scn = df_scn.transpose()
df_scn


# Now we will plot the histogram. The codes to plot the histogram are as follows:

# In[57]:


df_scn.plot(kind="hist", figsize=(20,10))
plt.title("Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013")
plt.xlabel("Number of immigrants")
plt.ylabel("Years")


# Since the bins for the histogram are not uniform, we use the numpy histogram function to create the bins. The code to do that are as follows:

# In[59]:


count,bin_edges = np.histogram(df_scn,15)
df_scn.plot(kind="hist", figsize=(20,10), xticks=bin_edges)
plt.title("Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013")
plt.xlabel("Number of immigrants")
plt.ylabel("Years")


# Now suppose we want to create an unstacked histogram. The code to do that is:

# In[63]:


count,bin_edges = np.histogram(df_scn,15)
df_scn.plot(kind="hist", 
            figsize=(20,10), 
            xticks=bin_edges,
            bins=15,
           color = ["darkslateblue", "coral","yellow"])
plt.title("Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013")
plt.xlabel("Number of immigrants")
plt.ylabel("Years")


# If we want to plot the same graph, then using the transparency parameter, we can increase or reduce the transparency of the bars. The following is the necessary code:

# In[67]:


count,bin_edges = np.histogram(df_scn,15)
df_scn.plot(kind="hist", 
            figsize=(20,10), 
            xticks=bin_edges,
            bins=15,
            alpha = 0.35,
           color = ["darkslateblue", "coral","red"])
plt.title("Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013")
plt.xlabel("Number of immigrants")
plt.ylabel("Years")


# To view all the possible number of available colors on matplotlib, we can use the following command:

# In[65]:


import matplotlib
for name, hex in matplotlib.colors.cnames.items():
    print(name, hex)


# If we do no want the plots to overlap each other, we can stack them using the stacked paramemter. Let's also adjust the min and max x-axis labels to remove the extra gap on the edges of the plot. We can pass a tuple (min,max) using the xlim paramater, as show below.

# In[70]:


count,bin_edges = np.histogram(df_scn,15)
xmin = bin_edges[0] - 10   #  first bin value is 31.0, adding buffer of 10 for aesthetic purposes 
xmax = bin_edges[-1] + 10  #  last bin value is 308.0, adding buffer of 10 for aesthetic purposes

count,bin_edges = np.histogram(df_scn,20)
df_scn.plot(kind="hist", 
            figsize=(20,10), 
            xticks=bin_edges,
            bins=15,
            alpha = 0.35,
            stacked = True,
            xlim = (xmin,xmax),
           color = ["darkslateblue", "coral","red"])
plt.title("Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013")
plt.xlabel("Number of immigrants")
plt.ylabel("Years")


# Now we plot transparent, stacked histogram of total number of immigrants for the countries Grrece, Albani and Bulgaria. The following codes serves the purpose:

# In[86]:


df_bal = df_can.loc[["Greece", "Albania", "Bulgaria"], years]
df_bal = df_bal.transpose()


# In[92]:


count,bin_edges = np.histogram(df_bal,15)
xmin = bin_edges[0]-1
xmax = bin_edges[-1] + 10
df_bal.plot(kind = "hist",
           figsize = (20,10),
           xticks = bin_edges,
           stacked = False,
           alpha=0.5, 
           xlim = (xmin,xmax),
           color=["blue","red","green"]
          )
plt.title("Histogram of Immigration from Greece, Albania, and Bulgaria from 1980 - 2013")
plt.xlabel("Number of immigrants")
plt.ylabel("Years")


# ## Plotting Bar Graphs

# A bar plot is a way of representing data where the length of the bars represents the magnitude/size of the feature/variable. Bar graphs usually represent numerical and categorical variables grouped in intervals.
# 
# To create a bar plot, we can pass one of two arguments via kind parameter in plot():
# 
# (1)kind=bar creates a vertical bar plot
# 
# (2)kind=barh creates a horizontal bar plot

# The 2008 - 2011 Icelandic Financial Crisis was a major economic and political event in Iceland. Relative to the size of its economy, Iceland's systemic banking collapse was the largest experienced by any country in economic history. The crisis led to a severe economic depression in 2008 - 2011 and significant political unrest.

# Let's compare the number of Icelandic immigrants (country = 'Iceland') to Canada from year 1980 to 2013.

# In[97]:


df_icl = df_can.loc[["Iceland"], years]
df_icl = df_icl.transpose()
df_icl.head()


# Now let us plot the simple bar graph for the immigration from Iceland:

# In[99]:


df_icl.plot(kind="bar",figsize=(20,10))
plt.title("Immigration to Canada from Iceland")
plt.xlabel("Years")
plt.ylabel("Number of immigrants")


# The bar plot above shows the total number of immigrants broken down by each year. We can clearly see the impact of the financial crisis; the number of immigrants to Canada started increasing rapidly after 2008.
# 
# Let's annotate this on the plot using the annotate method of the scripting layer or the pyplot interface.

# In[104]:


df_icl.plot(kind="bar",figsize=(20,10), rot=90)
plt.title("Immigration to Canada from Iceland")
plt.xlabel("Years")
plt.ylabel("Number of immigrants")
##Annotate arrow
plt.annotate('',
            xy=(32,70),
            xytext=(28,20),
            xycoords="data",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="green",lw=4))


# Let's also annotate a text to go over the arrow. We will pass in the following additional parameters:
# 
# rotation: rotation angle of text in degrees (counter clockwise)
# 
# va: vertical alignment of text [‘center’ | ‘top’ | ‘bottom’ | ‘baseline’]
# 
# ha: horizontal alignment of text [‘center’ | ‘right’ | ‘left’]

# In[108]:


df_icl.plot(kind="bar",figsize=(20,10), rot=90)
plt.title("Immigration to Canada from Iceland")
plt.xlabel("Years")
plt.ylabel("Number of immigrants")
##Annotate arrow
plt.annotate('',
            xy=(32,70),
            xytext=(28,20),
            xycoords="data",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="green",lw=4))
##Annotate text
plt.annotate("2008-2011 Financial Crisis",
            xy=(28,30),
           rotation = 71,
            va="bottom",
            ha="left")


# Let us now look at some horizontal bar graphs. We take the example of the top15 countries with the highest number of immigration to Canada and plot it in a horizontal bar garph. Let us first slice the required data:

# In[112]:


df_can.sort_values(by = "Total", ascending=False, axis=0, inplace=True)
df_top15 = df_can["Total"].head(15)
df_top15.head()


# Now we plot the horizontal bar graphs using the following set of commands:

# In[114]:


df_top15.plot(kind="barh", figsize=(20,10), color="green")
plt.xlabel("Number of Immigrants")
plt.title("Top 15 countries in terms of immigration to Canada")


# ## Plotting Pie Charts

# A pie chart is a circualr graphic that displays numeric proportions by dividing a circle (or pie) into proportional slices. You are most likely already familiar with pie charts as it is widely used in business and media. We can create pie charts in Matplotlib by passing in the kind=pie keyword.
# 
# Let's use a pie chart to explore the proportion (percentage) of new immigrants grouped by continents for the entire time period from 1980 to 2013.
# 
# Step 1: Gather data.
# 
# We will use pandas groupby method to summarize the immigration data by Continent. The general process of groupby involves the following steps:
# 
# Split: Splitting the data into groups based on some criteria.
# 
# Apply: Applying a function to each group independently:
# 
# .sum()
# 
# .count()
# 
# .mean() 
# 
# .std() 
# 
# .aggregate()
# 
# .apply()
# 
# .etc..
# 
# Combine: Combining the results into a data structure.

# To understand how pie-charts work, we will take the example of the total immigration from each of the continents and how much is the share of each of the continent to the total immigration. In order to do this, we need to group each country by their respective continent and then take the sum over all the years. The codes to do is as follows:

# In[115]:


df_continents = df_can.groupby("Continent", axis=0).sum()
df_continents


# ### Plot 1: Simple Pie Chart of total immigration by Contitnent

# In[116]:


df_continents["Total"].plot(kind="pie", 
                           figsize = (5,6),
                           autopct = "%1.1f%%",
                           startangle = 90,
                           shadow=True)
plt.title("Immigration to Canada by Continent [1980 - 2013]")
plt.axis("equal")


# The above visual is not very clear, the numbers and text overlap in some instances. Let's make a few modifications to improve the visuals:
# 
# Remove the text labels on the pie chart by passing in legend and add it as a seperate legend using plt.legend().
# 
# Push out the percentages to sit just outside the pie chart by passing in pctdistance parameter.
# 
# Pass in a custom set of colors for continents by passing in colors parameter.
# 
# Explode the pie chart to emphasize the lowest three continents (Africa, North America, and Latin America and Carribbean) by pasing in explode parameter.

# ### Plot 2: A Simple Pie Chart to Show Share of Total Immigrants by Continents with cleaner features

# In[125]:


color_list = ["green", "blue", "coral", "red","orange", "beige"]
explode_list = [0.1, 0, 0, 0, 0.1, 0.1] # ratio for each continent with which to offset each wedge
df_continents["Total"].plot(kind="pie", 
                           figsize = (15,6),
                           autopct = "%1.1f%%",
                           startangle = 90,
                           labels=None,
                           pctdistance=1.12,
                           colors=color_list,
                           explode=explode_list, 
                           shadow=True)
plt.title("Immigration to Canada by Continent [1980 - 2013]", y = 1.12)
plt.axis("equal")
plt.legend(labels=df_continents.index, loc="best")


# Now by using a pie-chart, we want to explore the proportion (percentage) of new immigrants grouped by continents in the year 2013.
# 
# In order to do this, we need to slice the data first and take the subset for 2013 only. The codes to do so are as follows:

# ### Plot 3: A Simple Pie Chart to Show Share of Total Immigrants by Continents in 2013

# In[126]:


color_list = ["green", "blue", "coral", "red","orange", "beige"]
explode_list = [0.1, 0, 0, 0, 0.1, 0.1] # ratio for each continent with which to offset each wedge
df_continents["2013"].plot(kind="pie", 
                           figsize = (15,6),
                           autopct = "%1.1f%%",
                           startangle = 90,
                           labels=None,
                           pctdistance=1.12,
                           colors=color_list,
                           explode=explode_list, 
                           shadow=True)
plt.title("Immigration to Canada by Continent in 2013", y = 1.12)
plt.axis("equal")
plt.legend(labels=df_continents.index, loc="best")


# ### Plotting Box Plots

# A box plot is a way of statistically representing the distribution of the data through five main dimensions:
# 
# Minimun: Smallest number in the dataset.
# 
# First quartile: Middle number between the minimum and the median.
# 
# Second quartile (Median): Middle number of the (sorted) dataset.
# 
# Third quartile: Middle number between median and maximum.
# 
# Maximum: Highest number in the dataset.

# To make a box plot, we can use kind=box in plot method invoked on a pandas series or dataframe.
# 
# Let's plot the box plot for the Japanese immigrants between 1980 - 2013.
# 
# Step 1: Get the dataset. Even though we are extracting the data for just one country, we will obtain it as a dataframe. This will help us with calling the dataframe.describe() method to view the percentiles.
# 

# In[124]:


df_japan = df_can.loc[["Japan"],years].transpose()
df_japan.head()


# ### Plot 1:Box Plot of the Distribution of Immigrants from Japan in 2013

# In[127]:


df_japan.plot(kind="box", figsize=(8,6))
plt.title("Box Plot of Japanese immigrants from 1980-2013")
plt.ylabel("Number of Immigrants")


# We can immediately make a few key observations from the plot above:
# 
# 1. The minimum number of immigrants is around 200 (min), maximum number is around 1300 (max), and  median number of immigrants is around 900 (median).
# 
# 2. 25% of the years for period 1980 - 2013 had an annual immigrant count of ~500 or fewer (First quartile).
# 
# 3. 75% of the years for period 1980 - 2013 had an annual immigrant count of ~1100 or fewer (Third quartile).
# 
# We can view the actual numbers by calling the `describe()` method on the dataframe.

# In[128]:


df_japan.describe()


# Now let us compare the distribution of Immigration from India and China:

# ### Plot 2:Box Plot of the Distribution of Total Immigrants from China and India 

# In[133]:


df_CI = df_can.loc[["China", "India"], years].transpose()
df_CI.plot(kind="box", figsize=(10,7))
plt.title("Box Plot of the distribution of Immigrants from China and India from 1980-2013")
plt.ylabel("Number of Immigrants")


# We can observe that, while both countries have around the same median immigrant population (~20,000), China's immigrant population range is more spread out than India's. The maximum population from India for any year (36,210) is around 15% lower than the maximum population from China (42,584).
# 
# If we want to create horizontal box plots, we can pass the vert parameter in the plot function and assign it to False. We can also specify a different color in case we are not a big fan of the default red color.

# ### Plot 3:Horizontal Box Plot of the Distribution of Total Immigrants from China and India 

# In[134]:


df_CI.plot(kind="box", figsize=(10,7), color="green", vert=False)
plt.title("Box Plot of the distribution of Immigrants from China and India from 1980-2013")
plt.xlabel("Number of Immigrants")


# ### Subplots

# Often times we might want to plot multiple plots within the same figure. For example, we might want to perform a side by side comparison of the box plot with the line plot of China and India's immigration.
# 
# To visualize multiple plots together, we can create a figure (overall canvas) and divide it into subplots, each containing a plot. With subplots, we usually work with the artist layer instead of the scripting layer.
# 
# Typical syntax is :
# 
#     fig = plt.figure() # create figure
#     
#     ax = fig.add_subplot(nrows, ncols, plot_number) # create subplots
# 
# Where
# 
# nrows and ncols are used to notionally split the figure into (nrows * ncols) sub-axes,
# 
# plot_number is used to identify the particular subplot that this function is to create within the notional grid. plot_number 
# 
# starts at 1, increments across rows first and has a maximum of nrows * ncols as shown below.

# ### Plot 4:Box Plot of the Distribution and Time Trends of Total Immigration from China and India 

# In order to use the subplot option, we need to use the artist layer of matplotlib. The following are the codes to have two subplots, one showing the comparision of the distribution of the total immigrants from China and India to Canada and the other one showing the time trends of the total number of immigrants from China and India:

# In[135]:


fig = plt.figure()

ax0 = fig.add_subplot(1, 2, 1) # add subplot 1 (1 row, 2 columns, first plot)
ax1 = fig.add_subplot(1, 2, 2) # add subplot 2 (1 row, 2 columns, second plot)

### Subplot 1:Boxplot

df_CI.plot(kind="box", figsize=(20,6), color="blue", vert=False,ax=ax0)
ax0.set_title('Box Plots of Immigrants from China and India (1980 - 2013)')
ax0.set_xlabel('Number of Immigrants')
ax0.set_ylabel('Countries')

### Subplot 2:Time Trend Plot

df_CI.plot(kind="line",figsize=(20,6),ax=ax1)
ax1.set_title("Line Plots of Immigrants from China and India (1980 - 2013)")
ax1.set_ylabel("Number of Immigrants")
ax1.set_xlabel("Years")


# Let us now create a box plot to visualize the distribution of the top 15 countries (based on total immigration) grouped by the decades 1980s, 1990s, and 2000s.
# 
# In order to do this, we first need to slice the data. The following line of codes helps to slice the data:

# In[136]:


df_top15 = df_can.sort_values(["Total"], ascending=False, axis=0).head(15)
df_top15.head()


# Now we create a new dataframe which contains the aggregate immigration for each decade. 
# 
# First we create the list of each decades using the map function of Numpy:

# In[137]:


year_80s = list(map(str, range(1980,1990)))
year_90s = list(map(str, range(1990,2000)))
year_00s = list(map(str, range(2000,2010)))


# Next, we find the decade sums for each country in the dataframe and the collect the dataframe for each decade and form the dataframe we need to compare the boxplots for immigration accross the decades:

# In[139]:


df_80s = df_top15.loc[:,year_80s].sum(axis=1)
df_90s = df_top15.loc[:,year_90s].sum(axis=1)
df_00s = df_top15.loc[:,year_00s].sum(axis=1)

new_df = pd.DataFrame({"1980":df_80s,"1990":df_90s,"2000":df_00s})
new_df.head()


# Now we can plot the box plot comparing the distribution of immigration accross the decades:

# ### Plot 5: Comparision of Immigration across the decades:

# In[141]:


new_df.plot(kind="box", figsize=(10,6), color="green")
plt.title("Immigration from top 15 countries for decades 80s, 90s and 2000s")
plt.ylabel("Immigrants")


# ## Plotting Scatter Plots

# A scatter plot (2D) is a useful method of comparing variables against each other. Scatter plots look similar to line plots in that they both map independent and dependent variables on a 2D graph. While the datapoints are connected together by a line in a line plot, they are not connected in a scatter plot. The data in a scatter plot is considered to express a trend. With further analysis using tools like regression, we can mathematically calculate this relationship and use it to predict trends outside the dataset.
# 
# Let's start by exploring the following:
# 
# Using a scatter plot, let's visualize the trend of total immigrantion to Canada (all countries combined) for the years 1980 - 2013.

# In[143]:


df_tot = pd.DataFrame(df_can[years].sum(axis=0))
df_tot.index = map(int,df_tot.index)
df_tot.reset_index(inplace=True)
df_tot.columns=["year", "total"]
df_tot.head()


# ### Plot 1: Scatter Plot Showing the Relationship between Year and Total Immigration

# In[144]:


df_tot.plot(kind="scatter", x="year", y="total", figsize=(10,6), color="blue")
plt.title("Total Immigration to Canada between 1980-2013")
plt.xlabel("Years")
plt.ylabel("Number of Immigrants")


# So let's try to plot a linear line of best fit, and use it to  predict the number of immigrants in 2015.
# 
# Step 1: Get the equation of line of best fit. We will use **Numpy**'s `polyfit()` method by passing in the following:
# 
# - `x`: x-coordinates of the data. 
# 
# - `y`: y-coordinates of the data. 
# 
# - `deg`: Degree of fitting polynomial. 1 = linear, 2 = quadratic, and so on.

# In[145]:


x = df_tot["year"]
y = df_tot["total"]
fit = np.polyfit(x,y,deg=1)
fit


# The output is an array with the polynomial coefficients, highest powers first. Since we are plotting a linear regression $y= a\times x + b$, our output has 2 elements [5.56709228e+03, -1.09261952e+07] with the the slope in position 0 and intercept in position 1.

# Now we plot the regression line on the scatter plot using the following codes:

# ### Plot 2: Scatter Plot with Linear Fit Showing the Relationship between Year and Total Immigration

# In[154]:


df_tot.plot(kind="scatter", x="year", y="total", figsize=(10,6), color="blue")
plt.title("Total Immigration to Canada between 1980-2013")
plt.xlabel("Years")
plt.ylabel("Number of Immigrants")
plt.plot(x,fit[0]*x + fit[1], color="green")
plt.annotate("y={0:.0f}x + {1:.0f}".format(fit[0], fit[1]), xy=(2000,150000))


# Now let us create a scatter plot of the total immigration from Denmark, Norway, and Sweden to Canada from 1980 to 2013.
# 
# The codes are as follows:

# ### Plot 3: Scatter Plot Showing the Relationship between Year and Total Immigration from Denmark, Sweden and Norway

# In[153]:


df_countries = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()
df_total = pd.DataFrame(df_countries.sum(axis=1))
df_total.reset_index(inplace=True)
df_total.columns = ['year', 'total']
df_total['year'] = df_total['year'].astype(int)
df_total.head()


# In[156]:


df_total.plot(kind="scatter", x="year", y="total", color="red",figsize=(10,6))
plt.title('Immigration from Denmark, Norway, and Sweden to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')


# ## Plotting Bubble Plots

# A bubble plot is a variation of the scatter plot that displays three dimensions of data (x, y, z). The datapoints are replaced with bubbles, and the size of the bubble is determined by the third variable 'z', also known as the weight. In maplotlib, we can pass in an array or scalar to the keyword s to plot(), that contains the weight of each point.
# 
# Let's start by analyzing the effect of Argentina's great depression.
# 
# Argentina suffered a great depression from 1998 - 2002, which caused widespread unemployment, riots, the fall of the government, and a default on the country's foreign debt. In terms of income, over 50% of Argentines were poor, and seven out of ten Argentine children were poor at the depth of the crisis in 2002.
# 
# Let's analyze the effect of this crisis, and compare Argentina's immigration to that of it's neighbour Brazil. Let's do that using a bubble plot of immigration from Brazil and Argentina for the years 1980 - 2013. We will set the weights for the bubble as the normalized value of the population for each year.

# In[161]:


df_arg = df_can[years].transpose()
df_arg.index = map(int,df_arg.index)
df_arg.index.name="year"
df_arg.reset_index(inplace=True)
df_arg.head()


# Now we create the normalized weights so that we can weight the bubbles.
# 
# There are several methods of normalizations in statistics, each with its own use. In this case, we will use feature scaling to bring all values into the range [0,1]. The general formula is:
# 
# $$X^{\prime}=\frac{X-X_{min}}{X_{max}-X_{min}}$$
# 
# where X is an original value, X' is the normalized value. The formula sets the max value in the dataset to 1, and sets the min value to 0. The rest of the datapoints are scaled to a value between 0-1 accordingly.

# Using the weights, we normalize the data for Brazil and that for Argentina. The following codes serves this purpose:

# In[165]:


norm_bra = ((df_arg["Brazil"]-df_arg["Brazil"].min())/(df_arg["Brazil"].max()-df_arg["Brazil"].min()))
norm_arg = ((df_arg["Argentina"]-df_arg["Argentina"].min())/(df_arg["Argentina"].max()-df_arg["Argentina"].min()))


# ### Plot 1: Bubble Plot for Comparison of Immigration between Brazil and Argentina

# To plot two different scatter plots in one plot, we can include the axes one plot into the other by passing it via the ax parameter.
# 
# We will also pass in the weights using the s parameter. Given that the normalized weights are between 0-1, they won't be visible on the plot. Therefore we will:
# 
# multiply weights by 2000 to scale it up on the graph, and,
# 
# add 10 to compensate for the min value (which has a 0 weight and therefore scale with x2000)

# In[166]:


### Brazil

ax0 = df_arg.plot(kind="scatter",
                x="year",
                y="Brazil",
                figsize=(14,8),
                alpha=0.5,
                color="yellow",
                s=norm_bra*2000+10,
                xlim=(1975,2010))

### Argentina

ax0 = df_arg.plot(kind="scatter",
                x="year",
                y="Argentina",
                figsize=(14,8),
                alpha=0.5,
                color="blue",
                s=norm_arg*2000+10,
                ax=ax0)
ax0.set_ylabel("Number of Immigrants")
ax0.set_title("Immigration from Brazil and Argentina from 1980 - 2013")
ax0.legend(["Brazil", "Argentina"],loc="upper left", fontsize="x-large")


# The size of the bubble corresponds to the magnitude of immigrating population for that year, compared to the 1980 - 2013 data. The larger the bubble, the more immigrants in that year.
# 
# From the plot above, we can see a corresponding increase in immigration from Argentina during the 1998 - 2002 great depression. We can also observe a similar spike around 1985 to 1993. In fact, Argentina had suffered a great depression from 1974 - 1990, just before the onset of 1998 - 2002 great depression.
# 
# On a similar note, Brazil suffered the Samba Effect where the Brazilian real (currency) dropped nearly 35% in 1999. There was a fear of a South American financial crisis as many South American countries were heavily dependent on industrial exports from Brazil. The Brazilian government subsequently adopted an austerity program, and the economy slowly recovered over the years, culminating in a surge in 2010. The immigration data reflect these events.

# Now let us create bubble plots of immigration from China and India to visualize any differences with time from 1980 to 2013.
# 
# First, we need to arrange the data so that we can perform this analysis. The codes for data reshaping is as follows:

# In[167]:


df_CI = df_can[years].transpose()
df_CI.index = map(int,df_CI.index)
df_CI.index.name="year"
df_CI.reset_index(inplace=True)
df_CI.head()


# Next, we create the weights for the bubbles. The codes to do so are as follows:

# In[169]:


norm_china = ((df_CI["China"]-df_CI["China"].min())/(df_CI["China"].max()-df_CI["China"].min()))
norm_india = ((df_CI["India"]-df_CI["India"].min())/(df_CI["India"].max()-df_CI["India"].min()))


# ### Plot 2: Bubble Plot for Comparison of Immigration between China and India

# In[170]:


### China

ax0 = df_CI.plot(kind="scatter",
                x="year",
                y="China",
                figsize=(14,8),
                alpha=0.5,
                color="red",
                s=norm_china*2000+10,
                xlim=(1975,2010))

### India

ax0 = df_CI.plot(kind="scatter",
                x="year",
                y="India",
                figsize=(14,8),
                alpha=0.5,
                color="blue",
                s=norm_india*2000+10,
                ax=ax0)
ax0.set_ylabel("Number of Immigrants")
ax0.set_title("Immigration from China and India from 1980 - 2013")
ax0.legend(["China", "India"],loc="upper left", fontsize="x-large")


# ## Plotting Waffle Charts

# A `waffle chart` is an interesting visualization that is normally created to display progress toward goals. It is commonly an effective option when you are trying to add interesting visualization features to a visual that consists mainly of cells, such as an Excel dashboard.

# Unfortunately, unlike R, `waffle` charts are not built into any of the Python visualization libraries. Therefore, we will learn how to create them from scratch.

# To create the waffle chart, we first select the data for which we want to create the waffle chart:

# In[171]:


df_scn = df_can.loc[["Denmark", "Norway", "Sweden"],:]
df_scn.head()


# ### Step 1. The first step into creating a waffle chart is determing the proportion of each category with respect to the total.

# So at first, we need to compute the proportion of immigration for each country. So we compute the proportion of each category with respect to the total: 

# In[172]:


total_values = sum(df_scn["Total"])
proportions = [(float(value)/total_values) for value in df_scn["Total"]]


# Printing out the proportions:

# In[174]:


for i, proportion in enumerate(proportions):
     print (df_scn.index.values[i] + ': ' + str(proportion)) 


# ### Step 2. The second step is defining the overall size of the waffle chart.

# In[175]:


width = 40
height = 10

tot_num_tiles = width*height


# ### Step 3. The third step is using the proportion of each category to determe it respective number of tiles

# In[176]:


tiles_per_category = [round(proportion*tot_num_tiles) for proportion in proportions]
for i, tiles in enumerate(tiles_per_category):
    print (df_scn.index.values[i] + ': ' + str(tiles))


# Based on the calculated proportions, Denmark will occupy 129 tiles of the waffle chart, Norway will occupy 77 tiles, and Sweden will occupy 194 tiles.

# ### Step 4. The fourth step is creating a matrix that resembles the waffle chart and populating it.

# We start by initializing the waffle chart as an empty matrix:

# In[177]:


waffle_chart = np.zeros((height,width))


# Then we define indices to loop through the waffle charts:

# In[178]:


category_index = 0
tile_index=0


# Next, we populate the waffle chart:

# In[179]:


for col in range(width):
    for row in range(height):
        tile_index += 1
        if tile_index > sum(tiles_per_category[0:category_index]):
             category_index += 1 
        waffle_chart[row, col] = category_index
        
waffle_chart        


# ### Step 5. Map the waffle chart matrix into a visual.

# In[180]:


fig = plt.figure()
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap = colormap)
plt.colorbar()


# ### Step 6. Prettify the chart.

# In[185]:


fig = plt.figure()
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap = colormap)
plt.colorbar()

# get the axis
ax = plt.gca()

# set minor ticks
ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
ax.set_yticks(np.arange(-.5, (height), 1), minor=True)

# add gridlines based on minor ticks
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

plt.xticks([])
plt.yticks([])


# ### Creating a function which would plot the waffle chart:

# In[186]:


def create_waffle_chart(categories, values, height, width, colormap, value_sign=''):

    # compute the proportion of each category with respect to the total
    total_values = sum(values)
    category_proportions = [(float(value) / total_values) for value in values]

    # compute the total number of tiles
    total_num_tiles = width * height # total number of tiles
    print ('Total number of tiles is', total_num_tiles)
    
    # compute the number of tiles for each catagory
    tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]

    # print out number of tiles per category
    for i, tiles in enumerate(tiles_per_category):
        print (df_scn.index.values[i] + ': ' + str(tiles))
    
    # initialize the waffle chart as an empty matrix
    waffle_chart = np.zeros((height, width))

    # define indices to loop through waffle chart
    category_index = 0
    tile_index = 0

    # populate the waffle chart
    for col in range(width):
        for row in range(height):
            tile_index += 1

            # if the number of tiles populated for the current category 
            # is equal to its corresponding allocated tiles...
            if tile_index > sum(tiles_per_category[0:category_index]):
                # ...proceed to the next category
                category_index += 1       
            
            # set the class value to an integer, which increases with class
            waffle_chart[row, col] = category_index
    
    # instantiate a new figure object
    fig = plt.figure()

    # use matshow to display the waffle chart
    colormap = plt.cm.coolwarm
    plt.matshow(waffle_chart, cmap=colormap)
    plt.colorbar()

    # get the axis
    ax = plt.gca()

    # set minor ticks
    ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
    ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
    
    # add dridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    plt.xticks([])
    plt.yticks([])

    # compute cumulative sum of individual categories to match color schemes between chart and legend
    values_cumsum = np.cumsum(values)
    total_values = values_cumsum[len(values_cumsum) - 1]

    # create legend
    legend_handles = []
    for i, category in enumerate(categories):
        if value_sign == '%':
            label_str = category + ' (' + str(values[i]) + value_sign + ')'
        else:
            label_str = category + ' (' + value_sign + str(values[i]) + ')'
            
        color_val = colormap(float(values_cumsum[i])/total_values)
        legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

    # add legend to chart
    plt.legend(
        handles=legend_handles,
        loc='lower center', 
        ncol=len(categories),
        bbox_to_anchor=(0., -0.2, 0.95, .1)
    )


# Now to create a `waffle` chart, all we have to do is call the function `create_waffle_chart`. Let's define the input parameters:

# In[188]:


width = 40 # width of chart
height = 10 # height of chart

categories = df_scn.index.values # categories
values = df_scn['Total'] # correponding values of categories

colormap = plt.cm.coolwarm # color map class


# And now let's call our function to create a `waffle` chart.

# In[189]:


create_waffle_chart(categories, values, height, width, colormap)


# ## Plotting Word Clouds

# Word clouds (also known as text clouds or tag clouds) work in a simple way: the more a specific word appears in a source of textual data (such as a speech, blog post, or database), the bigger and bolder it appears in the word cloud.
# 
# Luckily, a Python package already exists in Python for generating word clouds. The package, called word_cloud was developed by Andreas Mueller. You can learn more about the package by following this link.
# 
# Let's use this package to learn how to generate a word cloud for a given text document.
# 
# First, let's install the package.

# In[192]:


conda install -c conda-forge wordcloud


# Importing Package and its set of stopwords:

# In[193]:


from wordcloud import WordCloud, STOPWORDS


# Word clouds are commonly used to perform high-level analysis and visualization of text data. Accordinly, let's digress from the immigration dataset and work with an example that involves analyzing text data. Let's try to analyze a short novel written by Lewis Carroll titled Alice's Adventures in Wonderland. Let's go ahead and download a .txt file of the novel.

# In[198]:


pip install PyPDF2


# In[199]:


import PyPDF2 as pyd


# Download file and save as Alice novel text:

# In[201]:


das_capital = open(r"C:\Users\das90\Desktop\Capital-Volume-I.pdf")


# Next, let's use the stopwords that we imported from word_cloud. We use the function set to remove any redundant stopwords.

# In[202]:


stopwords = set(STOPWORDS)


# Create a word cloud object and generate a word cloud. For simplicity, let's generate a word cloud using only the first 20000 words in the novel.

# First we instantiate a word cloud object:

# In[210]:


das_capital_wc = WordCloud(background_color="red",
                       max_words=20000,
                       stopwords=stopwords   
                       )
das_capital_wc


# Next, we generate the word cloud:

# In[212]:


das_capital_wc.generate(das_capital)


# ## Plotting Regressions using Seaborn

# Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics.
# 

# Let us first install Seaborn:

# In[213]:


import seaborn as sns


# Let us now create a new dataframe that stores that total number of landed immigrants to Canada per year from 1980 to 2013.
# 
# The codes to do so are as follows:

# In[214]:


df_tot = pd.DataFrame(df_can[years].sum(axis=0))


# Let us change the type of years to a float variable:

# In[215]:


df_tot.index = map(float,df_tot.index)


# The we reset the index to put in back in as a column in the df_tot dataframe:

# In[216]:


df_tot.reset_index(inplace=True)


# Next, we rename the columns of the dataframe:

# In[217]:


df_tot.columns = ["year", "total"]
df_tot.head()


# With seaborn, generating a regression plot is as simple as calling the regplot function:
# 
# Let us plot a regression plot with year as the explanatory variable and the total number of immigrants as the outcome variable.

# In[218]:


ax = sns.regplot(x="year", y = "total", data=df_tot)


# Suppose we want to change the color of the dots in the plot. For instance, we want them to be green. The we write the code:

# In[219]:


ax= sns.regplot(x="year", y ="total", data=df_tot, color="green")


# Now we want to do two things.
# 
# (1) Expand the plot a bit
# 
# (2) Change the marker
# 
# In order to do this, we write the foloowing lines of code:

# In[220]:


plt.figure(figsize=(20,10))
ax = sns.regplot(x="year", y = "total", data=df_tot,marker="+", color="blue")


# Now let us increase the size of the markers and add labels and titles to the plot:

# In[222]:


plt.figure(figsize=(15,10))
ax = sns.regplot(x="year", y = "total", data=df_tot,marker="+", color="blue", scatter_kws={"s":200})
ax.set(xlabel="Year", ylabel = "Total Immigrants")
ax.set_title("Total Immigration to Canada from 1980 - 2013")


# And finally increase the font size of the tickmark labels, the title, and the x-label and y-label.

# In[223]:


plt.figure(figsize=(15,10))
sns.set(font_scale=1.5)
ax = sns.regplot(x="year", y = "total", data=df_tot,marker="+", color="blue", scatter_kws={"s":200})
ax.set(xlabel="Year", ylabel = "Total Immigrants")
ax.set_title("Total Immigration to Canada from 1980 - 2013")


# Suppose we want to change the purple background to a plain white backhground, we run the following codes:

# In[224]:


plt.figure(figsize=(15,10))
sns.set(font_scale=1.5)
sns.set_style("ticks")
ax = sns.regplot(x="year", y = "total", data=df_tot,marker="+", color="blue", scatter_kws={"s":200})
ax.set(xlabel="Year", ylabel = "Total Immigrants")
ax.set_title("Total Immigration to Canada from 1980 - 2013")


# Or if we want to change the purple background to a plain white background, we run the following codes:

# In[225]:


plt.figure(figsize=(15,10))
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
ax = sns.regplot(x="year", y = "total", data=df_tot,marker="+", color="blue", scatter_kws={"s":200})
ax.set(xlabel="Year", ylabel = "Total Immigrants")
ax.set_title("Total Immigration to Canada from 1980 - 2013")


# Using seaborn to create a scatter plot with a regression line to visualize the total immigration from Denmark, Sweden, and Norway to Canada from 1980 to 2013.
# 
# First we slice the data:
# 

# In[234]:


df_scn = df_can.loc[["Denmark", "Norway", "Sweden"], years].transpose()
df_total = pd.DataFrame(df_scn.sum(axis=1))
df_total.reset_index(inplace=True)
df_total.columns=["year", "total"]
df_total["year"] = df_total["year"].astype("int")
df_total.head()


# Now we plot the regression of the data using Seaborn:

# In[238]:


import seaborn as sns

plt.figure(figsize=(15,10))
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
ax = sns.regplot(x="year", y = "total", data=df_total,marker="+", color="red", scatter_kws={"s":200})
ax.set(xlabel="Year", ylabel = "Total Immigrants")
ax.set_title("Total Immigration to Canada from Denmark, Norway and Sweden from 1980 - 2013")


# ## Generating Maps with Python:Folium Package

# Folium is a powerful Python library that helps you create several types of Leaflet maps. The fact that the Folium results are interactive makes this library very useful for dashboard building.
# 
# From the official Folium documentation page:
# 
# Folium builds on the data wrangling strengths of the Python ecosystem and the mapping strengths of the Leaflet.js library. Manipulate your data in Python, then visualize it in on a Leaflet map via Folium.
# 
# Folium makes it easy to visualize data that's been manipulated in Python on an interactive Leaflet map. It enables both the binding of data to a map for choropleth visualizations as well as passing Vincent/Vega visualizations as markers on the map.
# 
# The library has a number of built-in tilesets from OpenStreetMap, Mapbox, and Stamen, and supports custom tilesets with Mapbox or Cloudmade API keys. Folium supports both GeoJSON and TopoJSON overlays, as well as the binding of data to those overlays to create choropleth maps with color-brewer color schemes.

# Since Folium is not available by default, we need to load it:

# In[239]:


get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')


# In[272]:


import folium


# Generating the world map is straigtforward in Folium. You simply create a Folium Map object and then you display it. What is attactive about Folium maps is that they are interactive, so you can zoom into any region of interest despite the initial zoom level.

# We first need to define the world map as a folium object:

# In[273]:


world_map = folium.Map()


# Next we display the folium object:

# In[274]:


world_map


# Let us define the map centered around Kolkata with a low zoom level:

# In[275]:


world_map = folium.Map(location = [22.5726,88.3639], zoom_start=4)
world_map


# Let us now increase the zoom:

# In[276]:


world_map = folium.Map(location = [22.5726,88.3639], zoom_start=10)
world_map


# ### Stamen Toner Maps

# These are high-contrast B+W (black and white) maps. They are perfect for data mashups and exploring river meanders and coastal zones.
# 
# Let's create a Stamen Toner map of Kolkata with a zoom level of 4.

# In[313]:


world_map = folium.Map(location = [22.5726,88.3639], zoom_start=8,tiles="Stamen Toner")
world_map


# ### Stamen Terrain Map

# These are maps that feature hill shading and natural vegetation colors. They showcase advanced labeling and linework generalization of dual-carriageway roads.
# 
# Let's create a Stamen Terrain map of Kolkata with zoom level 4.

# In[278]:


world_map = folium.Map(location = [22.5726,88.3639], zoom_start=6,tiles="Stamen Terrain")
world_map


# ## Maps with markers

# Let's download and import the data on police department incidents using pandas read_csv() method.
# 
# Download the dataset and read it into a pandas dataframe:

# In[279]:


df_incidents = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/Police_Department_Incidents_-_Previous_Year__2016_.csv')
df_incidents.head()


# Let's find out how many entries there are in our dataset.

# In[280]:


df_incidents.shape


# So the dataframe consists of 150,500 crimes, which took place in the year 2016. In order to reduce computational cost, let's just work with the first 100 incidents in this dataset.
# 
# Let us get the first 100 crimes in the df_incidents dataframe:

# In[281]:


limit = 100
df_incidents = df_incidents.iloc[0:limit, :]


# Let's confirm that our dataframe now consists only of 100 crimes.

# In[282]:


df_incidents.shape


# Now that we reduced the data a little bit, let's visualize where these crimes took place in the city of San Francisco. We will use the default style and we will initialize the zoom level to 12.

# First we define the latitude and longitude of San Francisco:

# In[283]:


latitude = 37.77
longitude = -122.42


# Then using Folium,we define the map of San Francisco:

# In[284]:


sanfran_map = folium.Map(location=[latitude,longitude], zoom_start=12)
sanfran_map


# Now let's superimpose the locations of the crimes onto the map. The way to do that in Folium is to create a feature group with its own features and style and then add it to the sanfran_map.

# Lets first instantiate a feature group for the incidents in the dataframe:

# In[285]:


incidents = folium.map.FeatureGroup()


# Now we loop through the 100 crimes and add each to the incidents feature group:

# In[286]:


for lat, lng, in zip(df_incidents.Y, df_incidents.X):
    incidents.add_child(
        folium.features.CircleMarker(
            [lat, lng],
            radius=5, # define how big you want the circle markers to be
            color='yellow',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        )
    )


# Next, we add the incidents to the map:

# In[287]:


sanfran_map.add_child(incidents)


# We can also add some pop-up text that would get displayed when you hover over a marker. Let's make each marker display the category of the crime when hovered over.

# In[288]:


incidents = folium.map.FeatureGroup()
for lat, lng, in zip(df_incidents.Y, df_incidents.X):
    incidents.add_child(
        folium.features.CircleMarker(
            [lat, lng],
            radius=5, # define how big you want the circle markers to be
            color='yellow',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        )
    )


# Now we add a pop-up text for each marker on the map: 

# In[289]:


latitudes = list(df_incidents.Y)
longitudes = list(df_incidents.X)
labels = list(df_incidents.Category)

for lat, lng, label in zip(latitudes, longitudes, labels):
    folium.Marker([lat, lng], popup=label).add_to(sanfran_map)


# Adding incidents to the map:

# In[290]:


sanfran_map.add_child(incidents)


# If the map seems to be so congested will all these markers, there are two remedies to this problem. The simpler solution is to remove these location markers and just add the text to the circle markers themselves as follows:

# In[291]:


sanfran_map = folium.Map(location=[latitude, longitude], zoom_start=12)

for lat, lng, label in zip(df_incidents.Y, df_incidents.X, df_incidents.Category):
    folium.features.CircleMarker(
        [lat, lng],
        radius=5, 
        color='yellow',
        fill=True,
        popup=label,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(sanfran_map)
    
sanfran_map    


# The other proper remedy is to group the markers into different clusters. Each cluster is then represented by the number of crimes in each neighborhood. These clusters can be thought of as pockets of San Francisco which you can then analyze separately.

# To implement this, we start off by instantiating a MarkerCluster object and adding all the data points in the dataframe to this object.

# In[292]:


from folium import plugins

# let's start again with a clean copy of the map of San Francisco
sanfran_map = folium.Map(location = [latitude, longitude], zoom_start = 12)

# instantiate a mark cluster object for the incidents in the dataframe
incidents = plugins.MarkerCluster().add_to(sanfran_map)

# loop through the dataframe and add each data point to the mark cluster
for lat, lng, label, in zip(df_incidents.Y, df_incidents.X, df_incidents.Category):
    folium.Marker(
        location=[lat, lng],
        icon=None,
        popup=label,
    ).add_to(incidents)

# display map
sanfran_map


# ## Plotting Chloropeth Maps

# A Choropleth map is a thematic map in which areas are shaded or patterned in proportion to the measurement of the statistical variable being displayed on the map, such as population density or per-capita income. The choropleth map provides an easy way to visualize how a measurement varies across a geographic area or it shows the level of variability within a region.

# Now, let's create our own Choropleth map of the world depicting immigration from various countries to Canada.

# First, we download the dataset and read it into a pandas dataframe:

# In[293]:


df_can = pd.read_excel(r"C:\Users\das90\OneDrive\Coursera courses\IBM Python\Data Visualization\Matplotlib\UN_MigFlow_All_CountryFiles\Canada.xlsx",
                      skiprows=range(20),
                      skipfooter=2,
                      sheet_name="Canada by Citizenship")
df_can.head()


# let's rename the columns:

# In[294]:


df_can.rename(columns={"OdName":"Country","AreaName":"Continent","RegName":"Region"}, inplace=True)
df_can.head(2)


# clean up the dataset to remove unnecessary columns (eg. REG): 

# In[295]:


df_can.drop(["Coverage", "AREA", "REG", "DEV"], axis=1, inplace=True)
df_can.head()


# Column names that are integers (such as the years) might introduce some confusion. For example, when we are referencing the year 2013, one might confuse that when the 2013th positional index.
# 
# To avoid this ambuigity, let's convert the column names into strings: '1980' to '2013'.

# In[297]:


df_can.columns=list(map(str,df_can.columns))


# We want to add the column called "total" to the dataset. We run the following command:

# In[299]:


df_can['Total'] = df_can.sum(axis=1)


# Years that we will be using in this lesson - useful for plotting later on

# In[300]:


years = list(map(str,range(1980,2014)))


# In order to create a Choropleth map, we need a GeoJSON file that defines the areas/boundaries of the state, county, or country that we are interested in. In our case, since we are endeavoring to create a world map, we want a GeoJSON that defines the boundaries of all world countries. For your convenience, we will be providing you with this file, so let's go ahead and download it. Let's name it world_countries.json.

# In[305]:


get_ipython().system('wget --quiet https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/world_countries.json -O world_countries.json')


# Now that we have the GeoJSON file, let's create a world map, centered around [0, 0] latitude and longitude values, with an intial zoom level of 2, and using Mapbox Bright style.

# In[306]:


world_geo = r"world_countries.json"


# Creating a plain world map:

# In[307]:


world_map = folium.Map(location=[0, 0], zoom_start=2, tiles='Mapbox Bright')


# And now to create a Choropleth map, we will use the choropleth method with the following main parameters:
# 
# (1)geo_data, which is the GeoJSON file.
# 
# (2)data, which is the dataframe containing the data.
# 
# (3)columns, which represents the columns in the dataframe that will be used to create the Choropleth map.
# 
# (4)key_on, which is the key or variable in the GeoJSON file that contains the name of the variable of interest. To determine that, you will need to open the GeoJSON file using any text editor and note the name of the key or variable that contains the name of the countries, since the countries are our variable of interest. In this case, name is the key in the GeoJSON file that contains the name of the countries. Note that this key is case_sensitive, so you need to pass exactly as it exists in the GeoJSON file.

# Now we generate choropleth map using the total immigration of each country to Canada from 1980 to 2013:

# In[308]:


world_map.choropleth(
    geo_data=world_geo,
    data=df_can,
    columns=['Country', 'Total'],
    key_on='feature.properties.name',
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Immigration to Canada'
)

world_map


# As per our Choropleth map legend, the darker the color of a country and the closer the color to red, the higher the number of immigrants from that country. Accordingly, the highest immigration over the course of 33 years (from 1980 to 2013) was from China, India, and the Philippines, followed by Poland, Pakistan, and interestingly, the US.
# 
# Notice how the legend is displaying a negative boundary or threshold. Let's fix that by defining our own thresholds and starting with 0 instead of -6,918.

# In[309]:


world_geo = r"world_countries.json"


# Then we create a numpy array of length 6 and has linear spacing from the minium total immigration to the maximum total immigration:

# In[310]:


threshold_scale = np.linspace(df_can['Total'].min(),
                              df_can['Total'].max(),
                              6, dtype=int)
threshold_scale = threshold_scale.tolist() # change the numpy array to a list
threshold_scale[-1] = threshold_scale[-1] + 1 # make sure that the last value of the list is greater than the maximum immigration


# Now we let Folium determine the scale:

# In[311]:


world_map = folium.Map(location=[0, 0], zoom_start=2, tiles='Mapbox Bright')
world_map.choropleth(
    geo_data=world_geo,
    data=df_can,
    columns=['Country', 'Total'],
    key_on='feature.properties.name',
    threshold_scale=threshold_scale,
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Immigration to Canada',
    reset=True
)
world_map


# In[ ]:




