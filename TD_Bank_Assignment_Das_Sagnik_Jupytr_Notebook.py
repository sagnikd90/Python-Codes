#!/usr/bin/env python
# coding: utf-8

# # TD Bank Assignment
# 
# ## Sagnik Das

# In[27]:


import numpy as np
import pandas as pd
import statistics as stat
import math
import statsmodels.api as statm
import matplotlib as mlb
import matplotlib.pyplot as plt
import seaborn as sns


# ### Importing the dataset

# In[28]:


df=pd.read_excel(r"C:\Users\das90\OneDrive - City University of New York\Job market\TD Bank Assignment\TD Bank assignment data.xlsm")


# ### Question 1)
# 
# Divergence is defined as the squared difference between the mean score of good and bad accounts divided by their average variance, as the formula below:  
#                                                                                                                          
#    Divergence=  〖2(μ_G  -μ_B )〗^2/(σ_G^2+σ_B^2 )
# 
# where μ_G= mean of good accounts, μ_B= mean of bad accounts, σ_G= standard deviation of good accounts, and σ_B=  standard deviation of bad accounts. 
# Use the data provided and the formula above, write sample code to calculate the Divergence. 

# ### Solution:

# In[29]:


bad_score1_mean=df.loc[df["bad"]==1,"score1"].mean()
good_score1_mean=df.loc[df["bad"]==0,"score1"].mean()
bad_score1_var=(df.loc[df["bad"]==1,"score1"].std())**2
good_score1_var=(df.loc[df["bad"]==0,"score1"].std())**2

divergence_score1=2*((good_score1_mean-bad_score1_mean)**2/(good_score1_var+bad_score1_var))

bad_score2_mean=df.loc[df["bad"]==1,"score2"].mean()
good_score2_mean=df.loc[df["bad"]==0,"score2"].mean()
bad_score2_var=(df.loc[df["bad"]==1,"score2"].std())**2
good_score2_var=(df.loc[df["bad"]==0,"score2"].std())**2

divergence_score2=2*((good_score2_mean-bad_score2_mean)**2/(good_score2_var+bad_score2_var))

print("Bad score1 mean: %.3f"% bad_score1_mean,
      "Bad score2 mean: %.3f"% bad_score2_mean,
     "Good score1 mean:%.3f"%good_score1_mean,
      "Good score2 mean:%.3f"%good_score2_mean,
      "Bad score1 variance:%.3f"%bad_score1_var,
      "Bad score2 variance:%.3f"%bad_score2_var,
      "Good score1 variance:%.3f"%good_score1_var,
      "Good score2 variance:%.3f"%good_score2_var,
      "Divergence for score1:%.3f"%divergence_score1,
      "Divergence for score2:%.3f"%divergence_score2,sep="\n")


# ### Question 2)
# 
# Write code to generate a table as below.
# 
# •	Numbers in the example table are for presentation purpose only. The correct answer will be different.
# 
# •	Formatting and font do not matter.
# 
# •	Feel free to adjust the width of the score bands as you see fit.

# In[30]:


df_score1=df.filter(["ID","bad","score1"])

df_score1["score_bands"] = np.where((df_score1["score1"]<600),"<600",
                            np.where((df_score1["score1"]<620),"[600:620[",
                                    np.where((df_score1["score1"]<640),"[620:640[",
                                            np.where((df_score1["score1"]<660),"[640:660[",
                                                    np.where((df_score1["score1"]<680),"[660:680[",
                                                            np.where((df_score1["score1"]<700),"[680:700[",
                                                                    np.where((df_score1["score1"]<720),"[700:720[",
                                                                            np.where((df_score1["score1"]<740),"[720:740[",
                                                                                     np.where((df_score1["score1"]<760),"[740:760[",
                                                                                              np.where((df_score1["score1"]<780),"[760:780[",
                                                                                                       np.where((df_score1["score1"]<800),"[780:800[",">800")))))))))))

df_score1["score_bands"]=df_score1["score_bands"].astype("str")

df_score1["score_bands"]=df_score1["score_bands"].astype(pd.CategoricalDtype(["<600",
                                                     "[600:620[", 
                                                     "[620:640[", 
                                                     "[640:660[",
                                                     "[660:680[",
                                                     "[680:700[",
                                                     "[700:720[", 
                                                    "[720:740[",
                                                    "[740:760[",
                                                    "[760:780[",
                                                    "[780:800[",
                                                    ">800"],ordered=True))

df_score1=df_score1.set_index("score_bands")

df_score1_table=df_score1.groupby(["score_bands","bad"])["bad"].count().to_frame("count").reset_index()
df_score1_table=df_score1_table.set_index("score_bands")

df_score1_table=pd.pivot_table(df_score1_table,index="score_bands",columns="bad",values="count")
df_score1_table= df_score1_table.rename(columns={0:"Score1_good",1:"Score1_bad"})

df_score1_table["Total_score1"]=df_score1_table["Score1_good"]+df_score1_table["Score1_bad"]
df_score1_table["Bad_rate_score1"]=round(df_score1_table["Score1_bad"]/df_score1_table["Total_score1"],3)

df_score1_table=df_score1_table.filter(["score_bands","Total_score1","Score1_bad","Bad_rate_score1"])


df_score2=df.filter(["ID","bad","score2"])

df_score2["score_bands"] = np.where((df_score2["score2"]<600),"<600",
                            np.where((df_score2["score2"]<620),"[600:620[",
                                    np.where((df_score2["score2"]<640),"[620:640[",
                                            np.where((df_score2["score2"]<660),"[640:660[",
                                                    np.where((df_score2["score2"]<680),"[660:680[",
                                                            np.where((df_score2["score2"]<700),"[680:700[",
                                                                    np.where((df_score2["score2"]<720),"[700:720[",
                                                                            np.where((df_score2["score2"]<740),"[720:740[",
                                                                                     np.where((df_score2["score2"]<760),"[740:760[",
                                                                                              np.where((df_score2["score2"]<780),"[760:780[",
                                                                                                       np.where((df_score2["score2"]<800),"[780:800[",">800")))))))))))

df_score2["score_bands"]=df_score2["score_bands"].astype("str")

df_score2["score_bands"]=df_score2["score_bands"].astype(pd.CategoricalDtype(["<600",
                                                     "[600:620[", 
                                                     "[620:640[", 
                                                     "[640:660[",
                                                     "[660:680[",
                                                     "[680:700[",
                                                     "[700:720[", 
                                                    "[720:740[",
                                                    "[740:760[",
                                                    "[760:780[",
                                                    "[780:800[",
                                                    ">800"],ordered=True))

df_score2=df_score2.set_index("score_bands")

df_score2_table=df_score2.groupby(["score_bands","bad"])["bad"].count().to_frame("count").reset_index()
df_score2_table=df_score2_table.set_index("score_bands")

df_score2_table=pd.pivot_table(df_score2_table,index="score_bands",columns="bad",values="count")
df_score2_table= df_score2_table.rename(columns={0:"Score2_good",1:"Score2_bad"})

df_score2_table["Total_score2"]=df_score2_table["Score2_good"]+df_score2_table["Score2_bad"]
df_score2_table["Bad_rate_score2"]=round(df_score2_table["Score2_bad"]/df_score2_table["Total_score2"],3)

df_score2_table=df_score2_table.filter(["score_bands","Total_score2","Score2_bad","Bad_rate_score2"])

df_final_table=pd.merge(df_score1_table,df_score2_table,left_index=True,right_index=True)


# ### Solution:

# In[31]:


df_final_table.head(12)


# ### Question 3)
# 
# Based on question 2, write code to run a linear regression, fill in the table and generate a figure like the one below. (The exact values will not necessarily be the same.)
# 
# •	On the y-axis, the value of each point is the "log-odds" of each score band, which = log (good #/bad #).
# 
# •	On the x-axis, the value of each point is the upper boundary of each score band, e.g. 720 for the band [700-720).
# 
# •	Color, font or label does not matter.

# In[32]:


X_score1=df_score1[["score1"]]
X_score1=statm.add_constant(X_score1)
Y_score1=df_score1[["bad"]]

score1_logit=statm.Logit(Y_score1,X_score1).fit()
df_score1["log_odds"]=score1_logit.fittedvalues

X_score2=df_score2[["score2"]]
X_score2=statm.add_constant(X_score2)
Y_score2=df_score2[["bad"]]

score2_logit=statm.Logit(Y_score2,X_score2).fit()
df_score2["log_odds"]=score2_logit.fittedvalues

df_score1_fig=df_score1.filter(["score_bands","log_odds"])
df_score2_fig=df_score1.filter(["score_bands","log_odds"])


# ### Solution:

# #### Regression table

# In[33]:


results_score1=score1_logit.summary()
results_score2=score2_logit.summary()

results_score1_html=results_score1.tables[1].as_html()
results_score2_html=results_score2.tables[1].as_html()

df_reg_score1=pd.DataFrame(pd.read_html(results_score1_html, header=0, index_col=0)[0])
df_reg_score1=df_reg_score1.reset_index()
df_reg_score1=df_reg_score1.filter(["coef","std err","P>|z|"])
df_reg_score1=df_reg_score1.rename(columns={"coef":"Score1_coefficients","std err":"Score1_SE","P>|z|":"Score1_pVal"})
df_reg_score1=df_reg_score1.rename(index={0:"Intercept",1:"Coefficient"})

df_reg_score2=pd.DataFrame(pd.read_html(results_score2_html, header=0, index_col=0)[0])
df_reg_score2=df_reg_score2.reset_index()
df_reg_score2=df_reg_score2.filter(["coef","std err","P>|z|"])
df_reg_score2=df_reg_score2.rename(columns={"coef":"Score2_coefficients","std err":"Score2_SE","P>|z|":"Score2_pVal"})
df_reg_score2=df_reg_score2.rename(index={0:"Intercept",1:"Coefficient"})

df_reg_results=pd.merge(df_reg_score1,df_reg_score2,left_index=True,right_index=True)

PDO_score1=math.log(2)/(-0.0186)
PDO_score2=math.log(2)/(-0.0140)

df_reg_results.loc[len(df_reg_results.index)]=[divergence_score1, "-", "-", divergence_score2, "-","-"]
df_reg_results.loc[len(df_reg_results.index)]=[PDO_score1, "-", "-", PDO_score2, "-","-"]

df_reg_results=df_reg_results.rename(index={2:"Divergence",3:"PDO"})
df_reg_results


# #### Figure

# In[34]:


df_score1_fig=df_score1.filter(["score_bands","log_odds"])
df_score1_fig=df_score1_fig.reset_index()
df_score1_fig=pd.DataFrame(df_score1_fig.groupby(["score_bands"]).mean()["log_odds"]).reset_index()

df_score2_fig=df_score2.filter(["score_bands","log_odds"])
df_score2_fig=df_score2_fig.reset_index()
df_score2_fig=pd.DataFrame(df_score2_fig.groupby(["score_bands"]).mean()["log_odds"]).reset_index()

fig=plt.figure(figsize=(20,10))
ax=fig.add_subplot(111)
ax.scatter(df_score1_fig["score_bands"],
           df_score1_fig["log_odds"],
           c="red",
          marker="s",
          label="Score 1",
          alpha=0.5,
          s=20**2)
ax.scatter(df_score2_fig["score_bands"],
           df_score2_fig["log_odds"],
           c="blue",
          marker="o",
          label="Score 1",
          alpha=0.5,
          s=20**2)
ax.plot(df_score1_fig["score_bands"],
       df_score1_fig["log_odds"],
       color="black",
       linestyle="dotted")
ax.plot(df_score2_fig["score_bands"],
       df_score2_fig["log_odds"],
       color="black",
       linestyle="dashed")
plt.legend(bbox_to_anchor=(1.2,1),loc="upper right")
plt.ylabel("Log Odds")
plt.xlabel("Score Bands")
plt.rc("axes",labelsize=20)
plt.rc("xtick",labelsize=10)
plt.rc("ytick",labelsize=10)
plt.rc("legend",fontsize=20)
plt.grid(color='g', linestyle='-', linewidth=0.5)


# ### Question 4)
# 
# If you used any fitting package for question #3, try to write your own code, without the use of model-fitting software such as statsmodels OLS or numpy's polyfit, to calculate again the Intercept β0 and coefficient β1 in question #3.

# ### Solutions:

# In[35]:


class LogisticRegression:
    def initialize(self,X):
        weights = np.zeros((shape(X)[1]+1,1))
        X = np.c_[np.ones((shape(X)[0],1)),X]
        return weights,X
    def sigmoid(self,z):
        sig = 1/(1+e**(-z))
        return sig
    def cost(theta):
            z = dot(X,theta)
            cost0 = y.T.dot(log(self.sigmoid(z)))
            cost1 = (1-y).T.dot(log(1-self.sigmoid(z)))
            cost = -((cost1 + cost0))/len(y) 
            return cost
    def fit(self,X,y,alpha=0.001,iter=100):
        params,X = self.initialize(X)
        cost_list = np.zeros(iter,)
        for i in range(iter):
            params = params - alpha * dot(X.T, self.sigmoid(dot(X,params)) - np.reshape(y,(len(y),1)))
            cost_list[i] = cost(params)
        self.params = params
        return cost_list
    def predict(self,X):
        z = dot(self.initialize(X)[1],self.weights)
        lis = []
        for i in self.sigmoid(z):
            if i>0.5:
                lis.append(1)
            else:
                lis.append(0)
        return lis    


# I apologize for this question. I am not so good at python programming. I tried to do it, but I think there is a bug in this code which I had pulled off the internet. It is not working when I am putting in the inputs. But I don't know enough python programming to figure out where exactly the bug is. Again, I apologize for not being able to answer this particular coding question.

# ### Question 5)
# 
# Based on the questions above, what do you think the purpose of these tests (divergence and PDO) is? Do you have any suggestions to improve upon these tests, e.g. any limitations of the existing tests, possible improvements or additional tests that could be performed? We are looking for a 1-2 paragraph discussion. 

# ### Solutions:
# 
# Divergence is a measure of the difference in credit scores between the mean of the defaulters to the non-defaulters. If the divergence score is high, it implies that the model is successfull in separating the defaulters from the non-defaulters. PDO or points to double the odds is a measure to determine the value of a factor/score. It means, what amount of change in points is required to double the odds of defaulting. It tells us what factor/score is more important in determining the changes in odds of a default.
# 
# The purpose of these tests is to see which score is better at predicting the risks of default. In this case, it is used to see which score is better at determining which is a bad account and which is a good account. 
# 
# Other machine learning models like decision trees or gradient boosting models can be used to test which score is better at predicting the bad accounts. Then the results of these models can be compared to the logistic regression models and model validation metrics can be used to see which model does a better job at predicting the chances of default.

# ## Coding for the writing assignment

# #### Scikit Learn packages for validation

# In[36]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# #### Scikit Learn packages for modelling

# In[37]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


# #### Scikit Learn packages for model metrics

# In[38]:


import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score as pscore
from sklearn.metrics import recall_score as rscore
from sklearn.metrics import f1_score as f1score


# #### Splitting the data into training and testing sets

# In[39]:


df_score1=df.filter(["ID","bad","score1"])
df_score2=df.filter(["ID","bad","score2"])

X_score1=df_score1["score1"]
X_score2=df_score2["score2"]
Y=df_score1["bad"]

train_X_score1,test_X_score1,train_Y_score1,test_Y_score1=train_test_split(X_score1,
                                                             Y,
                                                             test_size=0.25,
                                                             random_state=1)

train_X_score2,test_X_score2,train_Y_score2,test_Y_score2=train_test_split(X_score2,
                                                             Y,
                                                             test_size=0.25,
                                                             random_state=1)

train_X_score1= train_X_score1.values.reshape(-1, 1)
test_X_score1 = test_X_score1.values.reshape(-1, 1)

train_X_score2= train_X_score2.values.reshape(-1, 1)
test_X_score2 = test_X_score2.values.reshape(-1, 1)


# #### Logistic regression with Score1

# In[40]:


model_log=LogisticRegression(fit_intercept=True)

model_log.fit(train_X_score1,train_Y_score1)

log_predict_score1=model_log.predict(test_X_score1)


# #### Logistic regression with Score2

# In[41]:


model_log=LogisticRegression(fit_intercept=True)

model_log.fit(train_X_score2,train_Y_score2)

log_predict_score2=model_log.predict(test_X_score2)


# #### Gradient boosting with Score1

# In[42]:


model_gb=GradientBoostingClassifier(n_estimators=20,
                            learning_rate=0.1,
                                    max_depth=8)
model_gb.fit(train_X_score1,
             train_Y_score1)

gb_predict_score1=model_gb.predict(test_X_score1)


# #### Gradient boosting with Score2

# In[43]:


model_gb.fit(train_X_score2,
             train_Y_score2)

gb_predict_score2=model_gb.predict(test_X_score2)


# ### Model metrics

# #### Accuracy scores

# In[44]:


acc_score_log_score1=metrics.accuracy_score(test_Y_score1,
                                     log_predict_score1)

acc_score_log_score2=metrics.accuracy_score(test_Y_score2,
                                     log_predict_score2)

acc_score_gb_score1=metrics.accuracy_score(test_Y_score1,
                                     gb_predict_score1)

acc_score_gb_score2=metrics.accuracy_score(test_Y_score2,
                                     gb_predict_score2)

print("Accuracy score for logistic model with score1: %.3f"% acc_score_log_score1,
     "Accuracy score for gradient boosting model with score1: %.3f"% acc_score_gb_score1,
     "Accuracy score for logistic model with score2: %.3f"% acc_score_log_score2,
     "Accuracy score for gradient boosting model with score2: %.3f"% acc_score_gb_score2,sep="\n")


# #### Confusion matrix

# In[45]:


cmat_log_score1 = confusion_matrix(test_Y_score1,
                            log_predict_score1)

ax=plt.subplot()
sns.heatmap(cmat_log_score1/np.sum(cmat_log_score1),
           annot=True,
           fmt=".2%",
           ax=ax,
           cmap="Greens")
ax.set_xlabel("Predicted labels");ax.set_ylabel("True labels"); 
ax.set_title("Confusion matrix for logistic model with score1"); 
ax.xaxis.set_ticklabels(["good", "bad"]); ax.yaxis.set_ticklabels(["good", "bad"])


# In[46]:


cmat_gb_score1 = confusion_matrix(test_Y_score1,
                            gb_predict_score1)

ax=plt.subplot()
sns.heatmap(cmat_gb_score1/np.sum(cmat_gb_score1),
           annot=True,
           fmt=".2%",
           ax=ax,
           cmap="Greens")
ax.set_xlabel("Predicted labels");ax.set_ylabel("True labels"); 
ax.set_title("Confusion matrix for GB model with score1"); 
ax.xaxis.set_ticklabels(["good", "bad"]); ax.yaxis.set_ticklabels(["good", "bad"])


# In[47]:


cmat_log_score2 = confusion_matrix(test_Y_score2,
                            log_predict_score2)

ax=plt.subplot()
sns.heatmap(cmat_log_score2/np.sum(cmat_log_score2),
           annot=True,
           fmt=".2%",
           ax=ax,
           cmap="Greens")
ax.set_xlabel("Predicted labels");ax.set_ylabel("True labels"); 
ax.set_title("Confusion matrix for logistic model with score2"); 
ax.xaxis.set_ticklabels(["good", "bad"]); ax.yaxis.set_ticklabels(["good", "bad"])


# In[48]:


cmat_gb_score2 = confusion_matrix(test_Y_score2,
                            gb_predict_score2)

ax=plt.subplot()
sns.heatmap(cmat_gb_score2/np.sum(cmat_gb_score2),
           annot=True,
           fmt=".2%",
           ax=ax,
           cmap="Greens")
ax.set_xlabel("Predicted labels");ax.set_ylabel("True labels"); 
ax.set_title("Confusion matrix for GB model with score2"); 
ax.xaxis.set_ticklabels(["good", "bad"]); ax.yaxis.set_ticklabels(["good", "bad"])


# #### Precision scores

# In[49]:


print("Precision scores for logistic model with score1:%.3f"%pscore(test_Y_score1,log_predict_score1,average="weighted"),
     "Precision scores for GB model with score1:%.3f"%pscore(test_Y_score1,gb_predict_score1,average="weighted"),
     "Precision scores for logistic model with score2:%.3f"%pscore(test_Y_score2,log_predict_score2,average="weighted"),
     "Precision scores for GB model with score2:%.3f"%pscore(test_Y_score2,gb_predict_score2,average="weighted"),sep="\n")


# #### Recall scores

# In[50]:


print("Recall scores for logistic model with score1:%.3f"%rscore(test_Y_score1,log_predict_score1,average="weighted"),
     "Recall scores for GB model with score1:%.3f"%rscore(test_Y_score1,gb_predict_score1,average="weighted"),
     "Recall scores for logistic model with score2:%.3f"%rscore(test_Y_score2,log_predict_score2,average="weighted"),
     "Recall scores for GB model with score2:%.3f"%rscore(test_Y_score2,gb_predict_score2,average="weighted"),sep="\n")


# #### F-1 Scores

# In[51]:


print("F-1 scores for logistic model with score1:%.3f"%f1score(test_Y_score1,log_predict_score1,average="weighted"),
     "F-1 scores for GB model with score1:%.3f"%f1score(test_Y_score1,gb_predict_score1,average="weighted"),
     "F-1 scores for logistic model with score2:%.3f"%f1score(test_Y_score2,log_predict_score2,average="weighted"),
     "F-1 scores for GB model with score2:%.3f"%f1score(test_Y_score2,gb_predict_score2,average="weighted"),sep="\n")


# In[ ]:




