#!/usr/bin/env python
# coding: utf-8

# ## Training and Testing Models: First step towards Machine Learning

# When we are dealing with in-sample evaluations, we are only focussing on how well our model fits the data that has been used to train the model. It does not say anything about how well our model will perform when we are predicting new data/out of sample data. That is why we can use the sample data we have and split it into training and test data annd use these datasets to predict how well our model will perform when we are dealing with new out of sample data. In this module, we will learn about how to split the data into training, testing and validation datasets and how to optimize the parameters involved so that we can get the best model when it comes to predicting out of sample data.

# ### Splitting Data into Training and Testing

# First,we will learn about how to split the given sample into Training Data and Testing Data.
# 
# But first, let us import the data and the primary packages we need to use.

# In[5]:


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

df["price"] = df["price"].astype("int")


# Since we are only concerned with the numeric data, we will only load the numeric elements of the dataset and drop all the other variables which are not numeric in nature. 
# 
# The following command would only keep the variables which are numeric in nature and ignore the rest of the columns:

# In[9]:


df = df._get_numeric_data()
df.head()


# To split the data into Training and Testing, we need to install the package model_slection from the scikit library:

# In[10]:


from sklearn.model_selection import train_test_split


# Next, we assign the variables of the datatset as the Target and the Predictor variables. 
# 
# First we assign the Target variable:

# In[12]:


y_data = df["price"]


# Next, as explanatory or predictor variables, we want to assign all the other variables in the dataset, except for the variable "price". So in the array matrix x_data, we assign all the variables in the dataset as predictor variables, except for the variable "price" which is the target variable in the dataset. 
# 
# The command to assign as the predictor matrix all the variables as the predictor variable except for "price" is:

# In[13]:


x_data = df.drop("price",axis=1)


# Now we have all the predictor variables, as well as the target variable for our model. In the next step, we split the dataset into Training Dataset and Testing Dataset using the following command:

# In[16]:


x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.30,random_state=1)


# Now to view how many samples we have for the Training and the Testing Dataset, we run the following command:

# In[17]:


print("Number of Observations in Training Samples:", x_train.shape[0], "Number of Observations in Testing Sample:", x_test.shape[0])


# Now let us test a model on our Training and Testing Samples and check the predictive capabilities of our model.
# 
# In order to do this, what we do is run a simple linear regression and then compare the $R^2$ of the regression with the Training Sample with the $R^2$ of the Testing Sample.

# To run the linear regression, we need to first load the linear regression package:

# In[18]:


from sklearn.linear_model import LinearRegression


# Next, we define a linear regression object "lm":
# 

# In[19]:


lm = LinearRegression()


# Since we are running a simple linear regression model, we need to consider only one predictor variable. Let us consider "highway-mpg" as the predictor variable.
# 
# We will run the linear regression of "highway-mpg" on the target variable "price" using first the training sample and then the test sample:

# In[22]:


lm.fit(x_train[["highway-mpg"]], y_train)
lm.fit(x_test[["highway-mpg"]],y_test)


# Now we can compare between the $R^2$ of both the Training and the Testing Sample:

# In[25]:


lm.score(x_train[["highway-mpg"]], y_train)


# In[26]:


lm.score(x_test[["highway-mpg"]], y_test)


# We notice that the $R^2$ for the Training Sample is better than the $R^2$ for the Testing Sample. This implies that our model fits the Training Data well. However, when it comes to out of sample data, it's predictive power is reduced.

# ### Cross Validation Method

# Although, splitting the data into Training Sample and Testing Sample is an efficient way to test the predictive capacity of the model, sometimes the sample is not bigh enough to split the data and get meaningful results. In such cases, we can use the Cross validation method to create a pool of Training and Testing Sample. 
# 
# What the cross validation method basically does is, divides the total sample into blocks according to speicified parameters and then uses each block to create a Training Sample, as well as for creating a Test Sample. So each block of data are used as a Training Sample, as well as a Testing Sample, iteratively.
# 
# There are many ways cross valdiation can be done. Let us look at some of the ways we can use the Cross-Valdiation method.

# To use the Cross Validation Method, we can need to import the cross_val_score package from the scikit.learn library:

# In[27]:


from sklearn.model_selection import cross_val_score


# We use the following command to do a cross-validation by dividing the total sample into 4 folds:

# In[28]:


Rscore = cross_val_score(lm, x_data[["highway-mpg"]],y_data,cv=4)


# In the above command, the first argument of the cross_val_score, "lm" implies the regression method we are implementing. In this case it is a simple linear regression. The second and the third arguments specify the predictor and the target variables. The last argument, "cv" specifies the number of folds/groups we are splitting the total sample. In this case, "cv=4" implies that we have divided the total sample into 4 folds.
# 
# Now we can print Rscore to see what it returns:

# In[29]:


print(Rscore)


# From printing Rscore, we see that it returns an array. This array is basically an array of $R^2$ for each of the fold samples. Since we have 4 folds, we get 4 $R^2$'s.  

# We can also use the cross_val_predict to get the predicted values of the target variable for each of the 4 fold samples.
# 
# We need to first import the cross_val_predict package:

# In[31]:


from sklearn.model_selection import cross_val_predict


# Now the code to get the predicted target variables, we run the following command:

# In[32]:


Predict = cross_val_predict(lm,x_data[["highway-mpg"]], y_data,cv=4)


# Now if we print the stored result Predict, we get:

# In[37]:


print(Predict[0:10])


# As can be seen, we get an array consisting of the predicted values of the Target variable for each of the 4 folds.
# 
# Let us now look at the mean and the Standard Deviation of $R^2$'s for our estimate:

# In[34]:


print("The mean R-Square is:", Rscore.mean(), "and the SD of the R-Squares are:", Rscore.std())


# Instead of looking at $R^2$, we can also look at the Mean Squared Error values for each of the folds. The following code does that:

# In[36]:


-1*cross_val_score(lm,x_data[["highway-mpg"]], y_data,cv=4,scoring="neg_mean_squared_error")


# ### Dealing with underfitting and overfitting:Correct Model Selection

# We often come across data where a linear fit does not result in a good fit. Rather, a higher order polynomial fit is more appropriate. In such cases, we need to adjust the degree of the polynomial so that we don't have underfitting. as well as overfitting. 
# 
# Underfitting is the scenario when the fitted line does not explain the variation in the target variable appropriately, i.e. we have a low value of the $R^2$. Overfitting is the scenario when the fitted line fits the data points very closely, but also leads to explaining the variation in the random error alongwith the variation in the target variable. We want to avoid both these situations. So when we are choosing our model, we have to be mindful about both underfitting and overfitting.

# Let us start by running a multiple linear regression model on both the training and the test sample and compare the results. But first we need to load the data and the required packages. The following set of commands does that:

# In[164]:


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

df["price"] = df["price"].astype("int")
df["horse-power"] = df["horse-power"].astype("int")
df.head()


# Converting the data into numeric data:

# In[165]:


df = df._get_numeric_data()
df.head()


# To split the data into Training sample and testing sample, we need the following package:

# In[166]:


from sklearn.model_selection import train_test_split


# We set the target variable and the predictor variables using the following set of commands:

# In[167]:


y_data = df[["price"]]
x_data=df.drop("price", axis=1)


# In the next step, we split the data:

# In[168]:


x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.30,random_state=1)


# Now we import the LinearRegression package from sklearn.linear_model library and define the regression element:

# In[169]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# We now run the multiple linear regression on the training and the testing samples:

# In[170]:


lm.fit(x_train[["wheel-base","length","width","height","curb-weight","engine-size","city-mpg","highway-mpg"]],y_train)
lm.fit(x_test[["wheel-base","length","width","height","curb-weight","engine-size","city-mpg","highway-mpg"]],y_test)


# Now we save the predicted values from the regression on each samples and plot the predicted values and look if they are different:

# In[171]:


yhat_train = lm.predict(x_train[["wheel-base","length","width","height","curb-weight","engine-size","city-mpg","highway-mpg"]])
yhat_train[0:5]


# In[172]:


yhat_test = lm.predict(x_test[["wheel-base","length","width","height","curb-weight","engine-size","city-mpg","highway-mpg"]])
yhat_test[0:5]


# We will now plot the results from the set of regressions on the training and the testing data and see if there are any differences. 
# 
# For plotting, we need to load the following packages:

# In[259]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# First let us define and create a plotting object.
# 
# The following are the codes to define and create a plotting object:

# In[174]:


def DistributionPlot(RedFunction,BlueFunction,RedName,BlueName,Title):
    width=12
    height=10
    plt.figure(figsize=(width,height))
    ax1 = sns.distplot(RedFunction,hist=False,color="r",label=RedName)
    ax2 = sns.distplot(BlueFunction,hist=False,color="b",label=BlueName)
    plt.title(Title)
    plt.xlabel("Price (in dollars)")
    plt.ylabel("Proportion of cars")
    plt.show()
    plt.close()
    return DistributionPlot    


# We will first plot the distribution of the predicted values from the training data:

# In[175]:


Title = "Distribution plot of Predicted Values Using Training Data vs Training Data Distribution"
DistributionPlot(y_train,yhat_train,"Actual Values(Train)", "Predicted Values(Train)", Title)


# Now we plot the distrbutions of the test data and the predicted data from the test data:

# In[176]:


Title = "Distribution plot of Predicted Values Using Testing Data vs Testing Data Distribution"
DistributionPlot(y_test,yhat_test,"Actual Values(Test)", "Predicted Values(Test)", Title)


# From the above two figures, we can conclude that when it comes to Training sample, our model performs well and the predicted values almost mimics the actual values. However, when we look at the distributions of the test data and the predicted values from the test data, we see that the prediction performs much worse compared to the training data. So, it can be concluded that there are some issues with the model.

# #### Polynomial regressions and the issues with overfitting and underfitting

# Since our multiple linear regression model fails to accurately do out of sample predictions, we consider a polynomial fit. However, when we are considering polynomial regressions of degrees greater than 1, we need to be mindful of the problems of underfitting and overfitting. 

# Let us start by importing the package necessary for running polynomial regressions:

# In[177]:


from sklearn.preprocessing import PolynomialFeatures


# Next, we again split the data. Now we will use 55% of the data for training and 45% Testing:

# In[178]:


x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.45,random_state=0)


# We will use a degree 5 polynomial and the variable "horse-power" as the predictor variable. Since we are using a degree 5 polynomial, we need to first transform the variable "horse-power". The following code does that:

# In[234]:


pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[["horse-power"]])
x_test_pr = pr.fit_transform(x_test[["horse-power"]])
pr


# In the next step, we need to create a linear regression model called "poly" and then train it:

# In[235]:


poly= LinearRegression()
poly.fit(x_train_pr,y_train)


# Now let us save the predicted values of the regression on the training sample using the following code:

# In[236]:


yhat_train = poly.predict(x_train_pr)


# Next, we run the regression object "poly" on the testing dataset:

# In[237]:


poly.fit(x_test_pr,y_test)


# For the testing data also, we save the predicted values in the object yhat_test:

# In[238]:


yhat_test = poly.predict(x_test_pr)


# Now we compare the predicted values of the training data with the actual values and also the predicted values from the testing data with the actual values of the testing data:

# Comparison from the training sample:

# In[239]:


print("Actual values of Training sample", y_train[0:5], "Predicted values from the training sample", yhat_train[0:5])


# Comparison from the testing sample:

# In[240]:


print("Actual values of Testing sample", y_test[0:5], "Predicted values from the Testing sample", yhat_test[0:5])


# We can also visualize the differences in the predicted and the actual values for both the training and the testing samples. 
# 
# We first need to define the visualization functions and the environments we want to use. The following codes define the plotting function:

# In[241]:


def PollyPlot(x_train, x_test, y_train, y_test, lm, poly_transform):
    width = 12
    height = 10
    plt.figure(figsize = (width, height))
    xmax = max([x_train.values.max(), x_test.values.min()])
    xmin = min([x_train.values.min(), x_test.values.min()])
    x = np.arange(xmin,xmax,0.1)
    plt.plot(x_train,y_train,"ro", label = "Training Data")
    plt.plot(x_test, y_test, "go", label = "Test Data")
    plt.plot(x, lm.predict(poly_transform.fit_transform(x.reshape(-1,1))), label='Predicted Function')
    plt.ylim([-10000,60000])
    plt.ylabel("Price")
    plt.legend()
    return PollyPlot


# In[242]:


PollyPlot(x_train[["horse-power"]],x_test[["horse-power"]], y_train,y_test,poly,pr)


# From the above figure, it is evident that our model predicts well up till 160 Horse-Power. After that the fitted line shoots upwards exponentially. So there is definitely some problem with our model.
# 
# In order to check the $R^2$ of our model, we run the following command:

# For the training data, the $R^2$ is given by:

# In[243]:


print("For training data", poly.score(x_train_pr,y_train))


# For the testing sample, the $R^2$ is given by:

# In[244]:


print("For testing data", poly.score(x_test_pr,y_test))


# Comparing the $R^2$'s, we see that although the model is a good predictor for our testing sample, for the training sample, there is definite overfitting, implied by the negative $R^2$.

# Now we can check how our model performs over a range of degrees of polynomials. We can check this by plotting the $R^2$'s of the fits for different degrees of polynomials.
# 
# The following code helps to perform this exercise:

# In[248]:


rsqu_test = []

order = [1,2,3,4,5,6,7,8,9,10]

for n in order:
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train[["horse-power"]])
    x_test_pr = pr.fit_transform(x_test[["horse-power"]])
    lm.fit(x_train_pr,y_train)
    rsqu_test.append(lm.score(x_test_pr,y_test))
    
plt.plot(order,rsqu_test)
plt.xlabel("order")
plt.ylabel("R-Squared")
plt.title("R-Squared using test data")
plt.text(3,0.75,"Maximum R-Squared")


# From the above diagram, we see that the $R^2$ for the test data reaches a maximum at degree 9 and then slowly diminishes.
# 

# To check how the fit varies according to the order of the polynomial and the proportion of the testing data that is been used, we can check the following interactive graph:

# In[260]:


from IPython.display import display
from IPython.html import widgets 
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual


# In[262]:


def f(order,test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[["horse-power"]])
    x_test_pr = pr.fit_transform(x_test[["horse-power"]])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train[["horse-power"]], x_test[["horse-power"]],y_train,y_test,poly,pr)


interact(f,order=(1,6,10), test_data=(0.05,0.30,0.80))


# ## Ridge Regression

# When we are using higher order polynomial fits, the coefficients of the higher order polynomials often tend to be very high. Not having the appropriate degree of the polynomial leads to overfitting or underfitting. In order to control the coefficients of the higher order polynomial terms, we use Ridge Regressions. 
# 
# In Ridge Regressions, we introduce a parameter alpha. The parameter alpha controls the coefficients of the higher order polynomial terms and hence can be a good method to check for overfitting or underfitting. 

# In[8]:


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

df["price"] = df["price"].astype("int")
df["horse-power"] = df["horse-power"].astype("int")
df = df._get_numeric_data()
df.head()


# When we are using Ridge regressions, we assign a parameter alpha, which determines the values of the coefficients of the higher order polynomials.

# Let us start with a 2 degree polynomial transformation of the data:

# In[24]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

y_data = df[["price"]]
x_data=df.drop("price", axis=1)

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.4,random_state=0)

pr = PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[["horse-power","curb-weight","engine-size","highway-mpg","city-mpg"]])
x_test_pr=pr.fit_transform(x_test[["horse-power","curb-weight","engine-size","highway-mpg","city-mpg"]])


# Now we import the ridge regression package:

# In[25]:


from sklearn.linear_model import Ridge


# When we run a ridge regression, we need to specify the alpha for the regression. 
# 
# The following is the command to do so:

# In[26]:


RidgeModel = Ridge(alpha=0.1)


# Next, we run the ridge regression using the following command:

# In[27]:


RidgeModel.fit(x_train_pr,y_train)


# In[14]:


RidgeModel.fit(x_test_pr,y_test)


# Now we save the predicted values from the ridge regressions for both the training and the test data using the following commands:

# In[28]:


yhat_train = RidgeModel.predict(x_train_pr)
yhat_test = RidgeModel.predict(x_test_pr)


# We can now check on the predicted values from the model and compare them with the actual values for both the training and the validation data:
# 
# For the training data:

# In[29]:


print(y_train[0:4], yhat_train[0:4])


# For the validation data:

# In[30]:


print(y_test[0:4], yhat_test[0:4])


# Now we would like to choose an alpha which minimizes the test error but also the $R^2$ is maximized. We can compare between the different values of alpha and it's effect on the test error and the $R^2$'s by running the following loop:

# In[31]:


rsqu_test=[]
rsqu_train = []
dummy1=[]

ALFA = 10*np.array(range(0,1000))

for alfa in ALFA:
    RidgeModel=Ridge(alpha=alfa)
    RidgeModel.fit(x_train_pr,y_train)
    RidgeModel.fit(x_test_pr,y_test)
    rsqu_test.append(RidgeModel.score(x_test_pr,y_test))
    rsqu_train.append(RidgeModel.score(x_train_pr,y_train))    


# Now we can plot the $R^2$ of the polynomial fit for the training data and for the test data and then compare how the $R^2$'s change over the different values of alpha:

# In[32]:


import matplotlib.pyplot as plt

width = 12
height=10
plt.figure(figsize=(width,height))

plt.plot(ALFA, rsqu_test,label="Validation Data")
plt.plot(ALFA,rsqu_train,label="Training Data")
plt.xlabel("Alpha")
plt.ylabel("R-Squared")
plt.legend()


# From the above figure, we can conlclude that as the value of alpha increases, the model becomes a better predictor of the out of sample data. However, as alpha increases, the model explains less and less of the training data. So we have a trade off and thus our choice of alpha shoulbe be made keeping in mind this trade off.

# ### Grid Search

# Choosing the right alpha when doing ridge regressions can be done with much less coding by using the Grid Search option. 
# 
# Parameters like alpha are called Hyperparameters and we can choose the appropriate hyperparameter by using the Grid Search packagae in python. 
# 
# Let us do an example of grid search in python:

# In[39]:


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

df["price"] = df["price"].astype("int")
df["horse-power"] = df["horse-power"].astype("int")
df = df._get_numeric_data()
df.head()

y_data = df[["price"]]
x_data=df.drop("price", axis=1)


# First we load the necessary packages:

# In[40]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# Next,we define a range of parameters over which python will peform the grid search:

# In[36]:


parameters1 = [{'alpha':[0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]}]
parameters1


# Now we create the ridge regression object:

# In[41]:


RR = Ridge()
RR


# Then we create a ridge object by using the following line of codes:

# In[42]:


Grid1 = GridSearchCV(RR,parameters1,cv=4)


# Next, we fit the model:

# In[44]:


Grid1.fit(x_data[["horse-power","city-mpg","highway-mpg","engine-size","curb-weight"]],y_data)


# The object finds the best parameter values on the valdiation data. We can find the values of the best parameters and assign it to a new variable to view the best estimators of the hyperparameters:

# In[45]:


bestRR = Grid1.best_estimator_
bestRR


# After we obtain the best estimator of the parameter value, we fit our model with the best parameter on the test data:

# In[46]:


x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.4,random_state=0)

bestRR.score(x_test[["horse-power","city-mpg","highway-mpg","engine-size","curb-weight"]],y_test)


# We can also view the different values of the $R^2$'s for the different values of the paramaters:

# In[48]:


scores = Grid1.cv_results_
scores["mean_test_score"]


# When using the grid search package, we can also normalize the data and see the grid search results for the normalized and the non-normalized data. The following commands help us to achieve this objective:

# In[49]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[50]:


RR = Ridge()


# In[52]:


parameters2 = [{'alpha':[0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000],'normalize':[True,False]}]


# In[53]:


Grid2 = GridSearchCV(RR,parameters2,cv=4)


# In[54]:


Grid2.fit(x_data[["highway-mpg","city-mpg","engine-size","curb-weight","horse-power"]],y_data)


# In[55]:


Grid2.best_estimator_


# In[58]:


scores = Grid2.cv_results_
scores["mean_test_score"]


# In the above result, we get two values for each value of the parameter value alpha.For each alpha, one corresponds to the the Normalization parameter being True and the other corresponds to the Normalization parameter being False. 

# In[ ]:




