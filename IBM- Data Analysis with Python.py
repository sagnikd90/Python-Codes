#!/usr/bin/env python
# coding: utf-8

# ### Importing Datasets into Python

# Let us import a CSV dataset into python:
# 

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv(r"C:\Users\das90\OneDrive\Coursera courses\IBM Python\Machine Learning Datasets\Car Import Data\imports-85.data", header=None)
df.head(5)


# Now we import the names of the columns:
# 

# In[7]:


headers = ["symboling", 
           "normalized losses", 
           "make", 
           "fuel-type",
           "aspirations", 
           "num-of-doors", 
           "body-style", 
           "drive-wheels", 
           "engine-location",
          "engine-location,"
          "wheel-base",
          "length",
          "width",
          "height",
          "curb-weight",
          "engine-type",
          "num-of-cylinders",
          "engine-size",
          "fuel-system",
          "bore",
          "stroke",
          "compression-ratio",
          "horse-power",
          "peak-rpm",
          "city-mpg",
          "highway-mpg",
          "price"]


# In[8]:


df.columns=headers

df.head()


# Checking the data type of the columns:

# In[40]:


df.dtypes


# Checking the summary of data in each column:
# 

# In[41]:


df.describe()


# To check the information about the data,we can use the following command:

# In[42]:


df.info()


# ### Data Cleaning/Data Wrangling in Python

# #### Dealing with missing values in Python

# First we check if any of the columns of the data has a missing value or not. The best way to check for missing values if using the isnull() function of pandas.

# First we check if any of the columns has any missing values. This returns a logical vector.

# In[43]:


df.isnull()


# Next we check the total number of missing values for each column:

# In[44]:


df.isnull().sum()


# We can clearly see that none of the columns in this dataset has any missing values.

# If we did have missing values, we can use multiple methods to deal with the missing values. 
# 
# (1) Drop the missing values:

# If we want to drop a specific row which has a missing value, we write the following command:

# In[45]:


df.dropna(subset=["stroke"], axis=0, inplace=True)


# The above command drops all those rows of the dataset, for which the variable "price" has a missing value. The same code can be written as follows:

# In[46]:


df = df.dropna(subset=["stroke"], axis=0)


# In the first case, if we didn't write the argument (inplace=True), then the dataset would not have changed.

# (2) Replacing missing data:
# 
# We can replace the missing value with the mean of the column where the data is missing if it is an integer or a float. 
# 
# We can also replace the missing value with the most common value when we are dealing with categorical variables.

# Let us first look at the case, where we replace the missing values with the mean of the value:

# In[47]:


mean = df["length"].mean()

df["length"] = df["length"].replace(np.nan,mean)

df.head()


# ### Data Formatting in Python

# Suppose we want to convert the column of mpg which is in miles per gallon to litres per km. Then we should write the following code:

# In[48]:


df["city-mpg"] = 235/df["city-mpg"]

df.rename(columns={"city-mpg":"city-L/100km"}, inplace=True)

df.head()


# #### Correcting datatype

# In our dataset, we have the price variable as an object. We want to convert that into an integer. 
# 
# However, we see that the variable price has '?' in place of some values. So we first need to get rid of those values and replace them with NaN values. 
# 
# Once we replace them with NaN values, we can then drop those values.
# 
# The following code would help us do so:

# In[49]:


df["price"]=df["price"].replace('?', np.nan)

df.dropna(subset=["price"], axis=0,inplace=True)


# Now we convert the variable "price", which is an object data type to an integer data type.
# 
# The following code will help us do so:

# In[50]:


df["price"]=df["price"].astype("int")


# #### Data Normalization in Python

# Sometimes the variables are in varying magnitudes and that might affect the results of the models we run. So it is important to normalize the variables so that we don't have to face this problem.
# 
# There are numerous ways to normalize data:
# 
# (1) Simple Feature Scaling: $x_{new} = \frac{x_{old}}{x_{max}}$
# 
# (2) Min-Max Scaling:$x_{new} = \frac{x_{old}-x_{min}}{x_{max}-x{min}}$
# 
# (3) Z-score: $x_{new}=\frac{x_{old}-\mu}{\sigma}$

# Suppose we want to normalize the variable length using a simple feature scaling. Then we need to run the following code:

# In[51]:


df["length"] = df["length"]/df["length"].max()
df["length"].head(10)


# Now let us normalize the width variable using the Min-Max scaling technique:

# In[52]:


df["width"] = (df["width"]-df["width"].min())/(df["width"].max()-df["width"].min())
df["width"].head(10)


# Finally, let us normalize the height variable using the z-score normalization:

# In[53]:


df["height"] = (df["height"]-df["height"].mean())/df["height"].std()
df["height"].head(10)


# #### Data Binning in Python

# While analyzing data, we often use binning methods to group data into binned categories.
# 
# In our datatset, the variable price has 201 unique values. Suppose we want to categorize the prices as Low, Medium and High. To do this, we need to first find 4 equidistant numbers, which divides the price series into 3 equal categories. The code to do so is as follows:

# First, we create the bins:

# In[54]:


bins = np.linspace(min(df["price"]), max(df["price"]),4)


# Next, we create the group names:

# In[55]:


group = ["Low", "Medium", "High"]


# Lastly, we create the variable "price-binned" using the cut() function of pandas:

# In[56]:


df["price-binned"] = pd.cut(df["price"], bins, labels=group,include_lowest=True)
df["price-binned"]


# #### Converting categorical variables into quantitative variables

# We have the variable "fuel-type", which is a categorical variable in a string format. We want to quantify the categories in the form of a dummy variable:

# In[57]:


pd.get_dummies(df["fuel-type"])


# ### Exploratory Data Analysis with Python

# #### Descriptive Statistics

# The very simple way to see the data sumamry is to use the function describe():

# In[58]:


df.describe()


# To see the summary or count of the categorical variables, we can use the value_counts function. 
# 
# If we want to see the count of the categorical variable "drive-wheels", we use the following set of codes:

# In[72]:


drive_wheels_count = df["drive-wheels"].value_counts()
drive_wheels_count.rename({"drive-wheels":"value-counts"}, inplace=True)
drive_wheels_count.index.name="drive-wheels"
drive_wheels_count


# #### Groupby() function in python

# We can group the data according to one or multiple categorical variables and the find the summary statistics.
# 
# First, we select the categorical variables and the variable whose summary statistics we are concerned about.
# 
# 

# In[74]:


df1 = df[["drive-wheels", "body-style","price"]]
df_group = df1.groupby(["drive-wheels","body-style"], as_index=False).mean()
df_group


# To find summay statistics of variables by groups, we can also use the pivot table:

# In[76]:


df_pivot = df_group.pivot(index="drive-wheels",columns="body-style")
df_pivot


# #### Correlation

# In[77]:


from scipy import stats


# Suppose we want to find the correlation between Horsepower and Price.
# 
# To find the normal correlation, we use the following code:

# In[85]:


df["horse-power"]=df["horse-power"].replace('?', np.nan)

df.dropna(subset=["horse-power"], axis=0,inplace=True)

df["horse-power"] = df["horse-power"].astype("float")

df["horse-power"].corr(df["price"])


# To find the Spearman Rank Correlation, we use the following code:

# In[89]:


pearson_coeff,pvalue=stats.pearsonr(df["horse-power"], df["price"])

pearson_coeff,pvalue


# ### Model Development in Python

# #### Simple linear regression in Pandas

# In[16]:


import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\das90\OneDrive\Coursera courses\IBM Python\Machine Learning Datasets\Car Import Data\imports-85.data", header=None)

headers = ["symboling", 
           "normalized losses", 
           "make", 
           "fuel-type",
           "aspirations", 
           "num-of-doors", 
           "body-style", 
           "drive-wheels", 
           "engine-location",
          "engine-location,"
          "wheel-base",
          "length",
          "width",
          "height",
          "curb-weight",
          "engine-type",
          "num-of-cylinders",
          "engine-size",
          "fuel-system",
          "bore",
          "stroke",
          "compression-ratio",
          "horse-power",
          "peak-rpm",
          "city-mpg",
          "highway-mpg",
          "price"]
df.columns=headers

df = df.replace('?',np.nan)

df.dropna(inplace=True)

df.head()


# Now suppose we want to see the linear relationship between "highway-mpg" and "price".

# To run a simple linear regression,we need to first import the sklearn.linear_model from scikit package. 

# In[13]:


from sklearn.linear_model import LinearRegression


# Next,we create the linear regression object:

# In[14]:


lm=LinearRegression()


# Now we define the predicted and the predictor variable.

# In[17]:


x = df[["highway-mpg"]]
y = df[["price"]]


# Lastly, we run the linear regression:

# In[18]:


lm.fit(x,y)


# The regression intercept, the coeffecient and the predicted value are given by:

# In[31]:


yhat = lm.predict(x)
beta = lm.intercept_
beta1 = lm.coef_
result = [beta,beta1]
result


# Now let us run a multiple lienar regression model. To do that, we need to first define the explanatory variables:

# In[32]:


z = df[["horse-power", "curb-weight", "engine-size", "highway-mpg"]]


# In[33]:


lm.fit(z,y)


# In[36]:


yhat=lm.predict(z)
alpha = lm.intercept_
beta = lm.coef_
result = [alpha,beta]
result


# #### Visualization of regression

# Now we look at some regression visualizations. 
# 
# We can visualize the regression data in many ways. One of the ways to do so is to use the regplot.
# 
# To use regplot, we need to import the seasborn package:

# In[37]:


import seaborn as sns


# Now we plot the regression plot:

# In[41]:


df["highway-mpg"] = df["highway-mpg"].astype("int")
df["price"] = df["price"].astype("int")
df["horse-power"] = df["horse-power"].astype("int")
df["curb-weight"] = df["curb-weight"].astype("int")
df["engine-size"] = df["engine-size"].astype("int")


sns.regplot(x="highway-mpg", y = "price", data=df)


# We can also plot the residuals of a regression:

# In[45]:


sns.residplot(df["highway-mpg"], df["price"])


# We can also plot the distributions of the population of the target variable and compare it with the distribution of the predicted target variable:

# In[46]:


ax1 = sns.distplot(df["price"], hist=False,color="r", label = "Actual Value")
sns.distplot(yhat, hist=False,color="b", label = "Predicted Value", ax=ax1)


# 

# In[ ]:





# #### Polynomial Regression and using Pipelines

# For a polynomial/non-linear fit, we cannot use the LinearRegression package. To get a polynomial fit we need to run the following codes:

# In[49]:


x=df["highway-mpg"]
y=df["price"]
f = np.polyfit(x,y,3)
p = np.poly1d(f)
print(p)


# The above example was for a polynomial regression of dimension 1. 
# 
# Now suppose we are concerned with a polynomial regression of more than dimension 1. Then we cannot use only the numpy package and polyfit. We need the seaborn package. The code to run a polynomial fit of multiple dimensions, we need to run the following codes:

# In[52]:


from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=3,include_bias=False)


# Running polynomial regressions often need standardizing the variables. We can standardize the variables altogether by using the following set of codes:

# In[58]:


df = pd.read_csv(r"C:\Users\das90\OneDrive\Coursera courses\IBM Python\Machine Learning Datasets\Car Import Data\imports-85.data", header=None)

headers = ["symboling", 
           "normalized losses", 
           "make", 
           "fuel-type",
           "aspirations", 
           "num-of-doors", 
           "body-style", 
           "drive-wheels", 
           "engine-location",
          "engine-location,"
          "wheel-base",
          "length",
          "width",
          "height",
          "curb-weight",
          "engine-type",
          "num-of-cylinders",
          "engine-size",
          "fuel-system",
          "bore",
          "stroke",
          "compression-ratio",
          "horse-power",
          "peak-rpm",
          "city-mpg",
          "highway-mpg",
          "price"]
df.columns=headers

df = df.replace('?',np.nan)

df.dropna(inplace=True)

df["highway-mpg"] = df["highway-mpg"].astype("int")
df["price"] = df["price"].astype("int")
df["horse-power"] = df["horse-power"].astype("int")
df["curb-weight"] = df["curb-weight"].astype("int")
df["engine-size"] = df["engine-size"].astype("int")

from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
scale.fit(df[["horse-power","highway-mpg"]])
df_scale = scale.transform(df[["horse-power","highway-mpg"]])


# Now we will run the same operation as done in the previous line of codes, using the Pipeline operator.
# 
# If we use the pipeline operator to perform the above mentioned operations, we run the following line of codes:
# 
# (1) First, we import the required packages:

# In[59]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# (2) Then, we define the pipeline input using the following code:

# In[60]:


Input = [('scale', StandardScaler()),('polynomial', PolynomialFeatures(degree=3)), ('mode', LinearRegression())]


# (3) Now in this step, we define the pipeline operator:

# In[61]:


pipe=Pipeline(Input)


# (4) We use the pipeline operator to perform the operation:

# In[62]:


pipe.fit(df[["horse-power", "highway-mpg", "curb-weight", "engine-size"]], df["price"])


# In[65]:


yhat = pipe.predict(df[["horse-power", "highway-mpg", "curb-weight", "engine-size"]])


# #### Measures for in sample evaluation

# We can check the goodness of fit using the Mean Squared Error Method or R-Square Method.
# 
# Let us consider the following example of a simple linear regression model, where we find the R-Square of the fit:

# In[74]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r"C:\Users\das90\OneDrive\Coursera courses\IBM Python\Machine Learning Datasets\Car Import Data\imports-85.data", header=None)

headers = ["symboling", 
           "normalized losses", 
           "make", 
           "fuel-type",
           "aspirations", 
           "num-of-doors", 
           "body-style", 
           "drive-wheels", 
           "engine-location",
          "engine-location,"
          "wheel-base",
          "length",
          "width",
          "height",
          "curb-weight",
          "engine-type",
          "num-of-cylinders",
          "engine-size",
          "fuel-system",
          "bore",
          "stroke",
          "compression-ratio",
          "horse-power",
          "peak-rpm",
          "city-mpg",
          "highway-mpg",
          "price"]
df.columns=headers

df = df.replace('?',np.nan)

df.dropna(inplace=True)

lm = LinearRegression()

x = df[["highway-mpg"]]
y = df[["price"]]

lm.fit(x,y)


# To check the goodness of fit of the model, we run the following command:

# In[76]:


lm.score(x,y)


# #### Using the model for prediction and decision making

# Suppose we want to predict the value of the price of a car for a particular value of highway-mpg using our model.
# 
# The codes to do so are as follows:

# In[78]:


lm.fit(df[["highway-mpg"]],df[["price"]])


# In[79]:


lm.predict(np.array(30).reshape(-1,1))

