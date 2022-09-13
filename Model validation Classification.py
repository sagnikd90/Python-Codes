#!/usr/bin/env python
# coding: utf-8

# # Practice for Model Validation: Classification

# ## Model Validation: Kaggle Exercise

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import mean
from numpy import std


# ### Importing the datasets

# In[5]:


train=pd.read_csv(r"C:\Users\das90\OneDrive - City University of New York\Job market\Model validation exercise\Kaggle\Classification problem\train.csv")
test=pd.read_csv(r"C:\Users\das90\OneDrive - City University of New York\Job market\Model validation exercise\Kaggle\Classification problem\test.csv")


# In[6]:


train.head()


# In[8]:


pd.crosstab(index=train["price_range"],
              columns="count")


# ### Setting the variables for target and feature

# In[10]:


Y=train["price_range"]
X=train.drop("price_range",axis=1)


# ### Let us now import the necessary libraries for validation

# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# ### Importing the libraries for the models to be used for classification

# In[14]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# ### Importing the libraries of metrics for model validation

# In[54]:


import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score as pscore
from sklearn.metrics import recall_score as rscore
from sklearn.metrics import f1_score as f1score


# ### Splitting the data into training and testing dataset

# In[19]:


train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.25,random_state=1)

cv=KFold(n_splits=10,
        random_state=1,
        shuffle=True)


# ### K-Nearest Neighbor Classification

# In[35]:


model_knn=KNeighborsClassifier(n_neighbors=5)

model_knn.fit(train_X,train_Y)

knn_predict=model_knn.predict(test_X)

acc_score_knn=metrics.accuracy_score(test_Y,
                                     knn_predict)

cv_score_knn=cross_val_score(model_knn,
                         X,
                         Y,
                         cv=cv)


# In[40]:


print("Accuracy score for K-Nearest Neighbors:", format(metrics.accuracy_score(test_Y,knn_predict),".2f"))

print('Cross Validation Accuracy for K-Nearest Neighbors : %.3f (%.3f)' % (mean(cv_score_knn), std(cv_score_knn)))


# #### Precision, Recall and F1 Score for K-Nearest Neighbor Classification

# In[67]:


print("Precision score for K-Nearest Neighbor:", format(pscore(test_Y,knn_predict,average="weighted"),".3f"))

print("Recall score for K-Nearest Neighbor:", format(rscore(test_Y,knn_predict,average="weighted"),".3f"))

print("F-1 score for K-Nearest Neighbor:", format(f1score(test_Y,knn_predict,average="weighted"),".3f"))


# #### Confusion matrix for K-Nearest Neighbor Classification 

# In[68]:


cmat_KNN = confusion_matrix(test_Y,
                            knn_predict)

sns.heatmap(cmat_KNN/np.sum(cmat_KNN),
           annot=True,
           fmt=".2%")


# ### Decision Trees Classification

# In[69]:


model_dtc=DecisionTreeClassifier(max_depth=10,
                                random_state=1)

model_dtc.fit(train_X,
             train_Y)

dt_predict=model_dtc.predict(test_X)

acc_score_dt = metrics.accuracy_score(test_Y,
                                     dt_predict)

cv_score_dt = cross_val_score(model_dtc,
                             X,
                             Y,
                             cv=cv)


# In[70]:


print("Accuracy score for Decision Tree Classification:", format(metrics.accuracy_score(test_Y,dt_predict),".2f"))

print('Cross Validation Accuracy for Decision Tree Classification : %.3f (%.3f)' % (mean(cv_score_dt), std(cv_score_dt)))


# #### Precision, Recall and F-1 Score for Decision Trees

# In[71]:


print("Precision score for Decision Trees:", format(pscore(test_Y,dt_predict,average="weighted"),".3f"))

print("Recall score for Decision Trees:", format(rscore(test_Y,dt_predict,average="weighted"),".3f"))

print("F-1 score for Decision Trees:", format(f1score(test_Y,dt_predict,average="weighted"),".3f"))


# #### Confusion matrix for Decision Trees

# In[72]:


cmat_dt = confusion_matrix(test_Y,
                            dt_predict)

sns.heatmap(cmat_dt/np.sum(cmat_dt),
           annot=True,
           fmt=".2%")


# ### AdaBoost Classification

# In[80]:


model_ada=AdaBoostClassifier(n_estimators=20,
                            learning_rate=0.1)
model_ada.fit(train_X,
             train_Y)

ada_predict=model_ada.predict(test_X)

acc_score_ada = metrics.accuracy_score(test_Y,
                                     ada_predict)

cv_score_ada = cross_val_score(model_ada,
                             X,
                             Y,
                             cv=cv)


# In[81]:


print("Accuracy score for AdaBoost Classification:", format(metrics.accuracy_score(test_Y,ada_predict),".2f"))

print('Cross Validation Accuracy for AdaBoost Classification : %.3f (%.3f)' % (mean(cv_score_ada), std(cv_score_ada)))


# #### Precision, recall and F1 score for AdaBoost Classifier

# In[82]:


print("Precision score for AdaBoost Classifier:", format(pscore(test_Y,ada_predict,average="weighted"),".3f"))

print("Recall score for AdaBoost Classifier:", format(rscore(test_Y,ada_predict,average="weighted"),".3f"))

print("F-1 score for AdaBoost Classifier:", format(f1score(test_Y,ada_predict,average="weighted"),".3f"))


# #### Confusion matrix for AdaBoost Classifier

# In[83]:


cmat_ada = confusion_matrix(test_Y,
                            ada_predict)

sns.heatmap(cmat_ada/np.sum(cmat_ada),
           annot=True,
           fmt=".2%")


# ### Gradient Boosting Classifier

# In[85]:


model_gb=GradientBoostingClassifier(n_estimators=20,
                            learning_rate=0.1,
                                    max_features=5,
                                    max_depth=8)
model_gb.fit(train_X,
             train_Y)

gb_predict=model_gb.predict(test_X)

acc_score_gb = metrics.accuracy_score(test_Y,
                                     gb_predict)

cv_score_gb = cross_val_score(model_gb,
                             X,
                             Y,
                             cv=cv)


# In[86]:


print("Accuracy score for Gradient Boost Classification:", format(metrics.accuracy_score(test_Y,gb_predict),".2f"))

print('Cross Validation Accuracy for Gradient Boost Classification : %.3f (%.3f)' % (mean(cv_score_gb), std(cv_score_gb)))


# #### Precision, recall and F-1 Score for Gradient Boosting Classifier

# In[87]:


print("Precision score for Gradient Boosting Classifier:", format(pscore(test_Y,gb_predict,average="weighted"),".3f"))

print("Recall score for Gradient Boosting Classifier:", format(rscore(test_Y,gb_predict,average="weighted"),".3f"))

print("F-1 score for Gradient Boosting Classifier:", format(f1score(test_Y,gb_predict,average="weighted"),".3f"))


# #### Confusion matrix for Gradient Boosting Classifier

# In[88]:


cmat_gb = confusion_matrix(test_Y,
                            gb_predict)

sns.heatmap(cmat_gb/np.sum(cmat_gb),
           annot=True,
           fmt=".2%")


# ### XGBoost Classifier

# In[92]:


model_xgb=XGBClassifier(eta=0.1,
                      gamma=2,
                      max_depth=8,
                      min_child_weight=2,
                      reg_lambda=0.3,
                      reg_alpha=0.2)
model_xgb.fit(train_X,
             train_Y)

xgb_predict=model_xgb.predict(test_X)

acc_score_xgb = metrics.accuracy_score(test_Y,
                                     xgb_predict)

cv_score_xgb = cross_val_score(model_xgb,
                             X,
                             Y,
                             cv=cv)


# In[93]:


print("Accuracy score for XGBoost Classification:", format(metrics.accuracy_score(test_Y,xgb_predict),".2f"))

print('Cross Validation Accuracy for XGBoost Classification : %.3f (%.3f)' % (mean(cv_score_xgb), std(cv_score_xgb)))


# #### Precision, recall and F1 score of XGBoost Classifier

# In[94]:


print("Precision score for XGBoost Classifier:", format(pscore(test_Y,xgb_predict,average="weighted"),".3f"))

print("Recall score for XGBoost Classifier:", format(rscore(test_Y,xgb_predict,average="weighted"),".3f"))

print("F-1 score for XGBoost Classifier:", format(f1score(test_Y,xgb_predict,average="weighted"),".3f"))


# #### Confusion matrix for XGBoost Classifier

# In[95]:


cmat_xgb = confusion_matrix(test_Y,
                            xgb_predict)

sns.heatmap(cmat_xgb/np.sum(cmat_xgb),
           annot=True,
           fmt=".2%")


# ### Model selection based on accuracy scores

# In[96]:


print('Cross Validation Accuracy for K-Nearest Neighbors : %.3f (%.3f)' % (mean(cv_score_knn), std(cv_score_knn)))
print('Cross Validation Accuracy for Decision Tree Classification : %.3f (%.3f)' % (mean(cv_score_dt), std(cv_score_dt)))
print('Cross Validation Accuracy for AdaBoost Classification : %.3f (%.3f)' % (mean(cv_score_ada), std(cv_score_ada)))
print('Cross Validation Accuracy for Gradient Boost Classification : %.3f (%.3f)' % (mean(cv_score_gb), std(cv_score_gb)))
print('Cross Validation Accuracy for XGBoost Classification : %.3f (%.3f)' % (mean(cv_score_xgb), std(cv_score_xgb)))


# ### Model selection based on precision scores

# In[97]:


print("Precision score for K-Nearest Neighbor:", format(pscore(test_Y,knn_predict,average="weighted"),".3f"))
print("Precision score for Decision Trees:", format(pscore(test_Y,dt_predict,average="weighted"),".3f"))
print("Precision score for AdaBoost Classifier:", format(pscore(test_Y,ada_predict,average="weighted"),".3f"))
print("Precision score for Gradient Boosting Classifier:", format(pscore(test_Y,gb_predict,average="weighted"),".3f"))
print("Precision score for XGBoost Classifier:", format(pscore(test_Y,xgb_predict,average="weighted"),".3f"))


# ### Model selection based on recall scores

# In[98]:


print("Recall score for K-Nearest Neighbor:", format(rscore(test_Y,knn_predict,average="weighted"),".3f"))
print("Recall score for Decision Trees:", format(rscore(test_Y,dt_predict,average="weighted"),".3f"))
print("Recall score for AdaBoost Classifier:", format(rscore(test_Y,ada_predict,average="weighted"),".3f"))
print("Recall score for Gradient Boosting Classifier:", format(rscore(test_Y,gb_predict,average="weighted"),".3f"))
print("Recall score for XGBoost Classifier:", format(rscore(test_Y,xgb_predict,average="weighted"),".3f"))


# ### Model selection based on F-1 Scores

# In[99]:


print("F-1 score for K-Nearest Neighbor:", format(f1score(test_Y,knn_predict,average="weighted"),".3f"))
print("F-1 score for Decision Trees:", format(f1score(test_Y,dt_predict,average="weighted"),".3f"))
print("F-1 score for AdaBoost Classifier:", format(f1score(test_Y,ada_predict,average="weighted"),".3f"))
print("F-1 score for Gradient Boosting Classifier:", format(f1score(test_Y,gb_predict,average="weighted"),".3f"))
print("F-1 score for XGBoost Classifier:", format(f1score(test_Y,xgb_predict,average="weighted"),".3f"))


# In[ ]:




