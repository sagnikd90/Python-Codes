#!/usr/bin/env python
# coding: utf-8

# ## Loading the required packages

# In[140]:


from IPython.display import display
from IPython.display import HTML
import IPython.core.display as di 
di.display_html('<script>jQuery(function() {if (jQuery("body.notebook_app").length == 0) { jQuery(".input_area").toggle(); jQuery(".prompt").toggle();}});</script>', raw=True)

di.display_html('''<button onclick="jQuery('.input_area').toggle(); jQuery('.prompt').toggle();">Toggle code</button>''', raw=True)

import pandas as pd

low_memory=False

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np

import matplotlib as mlb
import matplotlib.pyplot as plt
import seaborn as sns


# ## Loading the data and performing some preliminary cleaning

# In[142]:


df_long= pd.read_csv(r"C:\Users\das90\OneDrive\Revelio\assignment_202110_long_file_0000_part_00.csv")

df_long=df_long.dropna()

df_long["year"] = pd.DatetimeIndex(df_long["month"]).year

df_long=df_long[(df_long["job_category"]!="empty")]

df_long["seniority"]=df_long["seniority"].astype("str")


# ## Subsetting the data for further analysis

# In[143]:


df_jobs=pd.get_dummies(df_long,columns=["gender"])

df_af=df_jobs[(df_jobs["region"]=="Sub-Saharan Africa")|
             (df_jobs["region"]=="Northern Africa")]

df_subaf=df_jobs[(df_jobs["region"]=="Sub-Saharan Africa")]

df_naf=df_jobs[(df_jobs["region"]=="Northern Africa")]

df_gender_af=df_af[["job_category",
                    "year",
                                 "seniority",
                    "gender_female",
                    "gender_male"]]

df_gender_subaf=df_subaf[["job_category",
                          "year",
                                 "seniority",
                    "gender_female",
                    "gender_male"]]

df_gender_naf=df_naf[["job_category",
                      "year",
                                 "seniority",
                    "gender_female",
                    "gender_male"]]


# ## Computing the average share of employees by gender for each job category. Subsetting the data to include only African regions.

# In[144]:


df_gender_jobs_af=df_gender_af.groupby(["job_category"],as_index=False).mean()

df_gender_jobs_subaf=df_gender_subaf.groupby(["job_category"],as_index=False).mean()

df_gender_jobs_naf=df_gender_naf.groupby(["job_category"],as_index=False).mean()


# ### Northern Africa and Sub-Saharan Africa combined

# In[145]:


cmap = plt.cm.cividis

fig, ax = plt.subplots(figsize=(20,10))
df_gender_jobs_af.plot(x="job_category",y=["gender_female","gender_male"],kind="barh",label=["female","male"],ax=ax,cmap=cmap,alpha=0.5)
plt.legend(bbox_to_anchor=(1.13,1),loc="upper right")
plt.ylabel("Job categories")
plt.xlabel("Share of workers across gender")
plt.grid(True,alpha=0.2,color="black")
plt.rc("axes",labelsize=25)
plt.rc("xtick",labelsize=25)
plt.rc("ytick",labelsize=25)
plt.rc("legend",fontsize=25)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)


# ### Sub-Saharan Africa

# In[147]:


cmap = plt.cm.cividis

fig, ax = plt.subplots(figsize=(20,10))
df_gender_jobs_subaf.plot(x="job_category",y=["gender_female","gender_male"],kind="barh",label=["female","male"],ax=ax,cmap=cmap,alpha=0.5)
plt.legend(bbox_to_anchor=(1.13,1),loc="upper right")
plt.ylabel("Job categories")
plt.xlabel("Share of workers across gender")
plt.grid(True,alpha=0.2,color="black")
plt.rc("axes",labelsize=25)
plt.rc("xtick",labelsize=25)
plt.rc("ytick",labelsize=25)
plt.rc("legend",fontsize=25)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)


# ### Northern Africa

# In[148]:


cmap = plt.cm.cividis

fig, ax = plt.subplots(figsize=(20,10))
df_gender_jobs_naf.plot(x="job_category",y=["gender_female","gender_male"],kind="barh",label=["female","male"],ax=ax,cmap=cmap,alpha=0.5)
plt.legend(bbox_to_anchor=(1.13,1),loc="upper right")
plt.ylabel("Job categories")
plt.xlabel("Share of workers across gender")
plt.grid(True,alpha=0.2,color="black")
plt.rc("axes",labelsize=25)
plt.rc("xtick",labelsize=25)
plt.rc("ytick",labelsize=25)
plt.rc("legend",fontsize=25)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)


# ## Computing the trend in the gap between the share of men and women employees for three job categories: Engineer, Finance and Scientist

# In[149]:


df_gender_af_year=df_gender_af[(df_gender_af["job_category"]=="Engineer")|
                              (df_gender_af["job_category"]=="Finance")|
                              (df_gender_af["job_category"]=="Scientist")]

df_gender_subaf_year=df_gender_subaf[(df_gender_subaf["job_category"]=="Engineer")|
                              (df_gender_subaf["job_category"]=="Finance")|
                              (df_gender_subaf["job_category"]=="Scientist")]

df_gender_naf_year=df_gender_naf[(df_gender_naf["job_category"]=="Engineer")|
                              (df_gender_naf["job_category"]=="Finance")|
                              (df_gender_naf["job_category"]=="Scientist")]

df_gender_years_af=df_gender_af_year.groupby(["year","job_category"],as_index=False).mean()

df_gender_years_subaf=df_gender_subaf_year.groupby(["year","job_category"],as_index=False).mean()

df_gender_years_naf=df_gender_naf_year.groupby(["year","job_category"],as_index=False).mean()

df_gender_years_af["gender_diff"]=df_gender_years_af["gender_male"]-df_gender_years_af["gender_female"]

df_gender_years_subaf["gender_diff"]=df_gender_years_subaf["gender_male"]-df_gender_years_subaf["gender_female"]

df_gender_years_naf["gender_diff"]=df_gender_years_naf["gender_male"]-df_gender_years_naf["gender_female"]


# ### Northern Africa and Sub-Saharan Africa combined

# In[150]:


cmap = plt.cm.cividis

fig, ax = plt.subplots(figsize=(20,10))
df_gender_years_af.groupby(["year","job_category"]).mean()["gender_diff"].unstack().plot(ax=ax,lw=5,cmap=cmap,alpha=0.7)
plt.legend(bbox_to_anchor=(1.2,1),loc="upper right")
plt.ylabel("Difference between share of men and women workers")
plt.xlabel("Year")
plt.rc("axes",labelsize=25)
plt.rc("xtick",labelsize=25)
plt.rc("ytick",labelsize=25)
plt.rc("legend",fontsize=25)
plt.grid(True,alpha=0.2,color="black")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)


# ### Sub-Saharan Africa

# In[151]:


cmap = plt.cm.cividis

fig, ax = plt.subplots(figsize=(20,10))
df_gender_years_subaf.groupby(["year","job_category"]).mean()["gender_diff"].unstack().plot(ax=ax,lw=5,cmap=cmap,alpha=0.7)
plt.legend(bbox_to_anchor=(1.2,1),loc="upper right")
plt.ylabel("Difference between share of men and women workers")
plt.xlabel("Year")
plt.rc("axes",labelsize=25)
plt.rc("xtick",labelsize=25)
plt.rc("ytick",labelsize=25)
plt.rc("legend",fontsize=25)
plt.grid(True,alpha=0.2,color="black")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)


# ### Northern Africa

# In[152]:


cmap = plt.cm.cividis

fig, ax = plt.subplots(figsize=(20,10))
df_gender_years_naf.groupby(["year","job_category"]).mean()["gender_diff"].unstack().plot(ax=ax,lw=5,cmap=cmap,alpha=0.7)
plt.legend(bbox_to_anchor=(1.2,1),loc="upper right")
plt.ylabel("Difference between share of men and women workers")
plt.xlabel("Year")
plt.rc("axes",labelsize=25)
plt.rc("xtick",labelsize=25)
plt.rc("ytick",labelsize=25)
plt.rc("legend",fontsize=25)
plt.grid(True,alpha=0.2,color="black")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)


# ## Gender pay gap

# ### Trend in pay-gap for three job categories: Engineer, Finance and Scientist

# In[153]:


df_salary=df_long[["job_category",
                    "year",
                   "region",
                   "gender",
                  "seniority",
                   "role_k50",
                  "salary"]]

df_salary_af=df_salary[(df_salary["region"]=="Sub-Saharan Africa")|
                    (df_salary["region"]=="Northern Africa")]




df_salary_jobs_year_af_mean=df_salary_af.groupby(["job_category","year","gender"])["salary"].mean().unstack("gender")

df_salary_jobs_year_af_mean=df_salary_jobs_year_af_mean.reset_index()

df_salary_jobs_year_af_mean["gender_gap"]=df_salary_jobs_year_af_mean["male"]-df_salary_jobs_year_af_mean["female"]

df_salary_jobs_year_af_mean=df_salary_jobs_year_af_mean[(df_salary_jobs_year_af_mean["job_category"]=="Engineer")|
                                                       (df_salary_jobs_year_af_mean["job_category"]=="Finance")|
                                                       (df_salary_jobs_year_af_mean["job_category"]=="Scientist")]


# In[154]:


cmap = plt.cm.cividis

fig, ax = plt.subplots(figsize=(20,10))
df_salary_jobs_year_af_mean.groupby(["year","job_category"]).mean()["gender_gap"].unstack().plot(ax=ax,lw=5,cmap=cmap,alpha=0.7)
plt.legend(bbox_to_anchor=(1.2,1),loc="upper right")
plt.ylabel("Gender pay gap")
plt.xlabel("Year")
plt.rc("axes",labelsize=25)
plt.rc("xtick",labelsize=25)
plt.rc("ytick",labelsize=25)
plt.rc("legend",fontsize=25)
plt.grid(True,alpha=0.2,color="black")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)


# ### Sub-Saharan Africa

# In[155]:


df_salary_subaf=df_salary[(df_salary["region"]=="Sub-Saharan Africa")]



df_salary_jobs_year_subaf_mean=df_salary_subaf.groupby(["job_category","year","gender"])["salary"].mean().unstack("gender")

df_salary_jobs_year_subaf_mean=df_salary_jobs_year_subaf_mean.reset_index()

df_salary_jobs_year_subaf_mean["gender_gap"]=df_salary_jobs_year_subaf_mean["male"]-df_salary_jobs_year_subaf_mean["female"]

df_salary_jobs_year_subaf_mean=df_salary_jobs_year_subaf_mean[(df_salary_jobs_year_subaf_mean["job_category"]=="Engineer")|
                                                       (df_salary_jobs_year_subaf_mean["job_category"]=="Finance")|
                                                       (df_salary_jobs_year_subaf_mean["job_category"]=="Scientist")]


# In[156]:


cmap = plt.cm.cividis

fig, ax = plt.subplots(figsize=(20,10))
df_salary_jobs_year_subaf_mean.groupby(["year","job_category"]).mean()["gender_gap"].unstack().plot(ax=ax,lw=5,cmap=cmap,alpha=0.7)
plt.legend(bbox_to_anchor=(1.2,1),loc="upper right")
plt.ylabel("Gender pay gap")
plt.xlabel("Year")
plt.rc("axes",labelsize=25)
plt.rc("xtick",labelsize=25)
plt.rc("ytick",labelsize=25)
plt.rc("legend",fontsize=25)
plt.grid(True,alpha=0.2,color="black")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)


# ### Northern Africa

# In[157]:


df_salary_naf=df_salary[(df_salary["region"]=="Northern Africa")]


df_salary_jobs_year_naf_mean=df_salary_naf.groupby(["job_category","year","gender"])["salary"].mean().unstack("gender")

df_salary_jobs_year_naf_mean=df_salary_jobs_year_naf_mean.reset_index()

df_salary_jobs_year_naf_mean["gender_gap"]=df_salary_jobs_year_naf_mean["male"]-df_salary_jobs_year_naf_mean["female"]

df_salary_jobs_year_naf_mean=df_salary_jobs_year_naf_mean[(df_salary_jobs_year_naf_mean["job_category"]=="Engineer")|
                                                       (df_salary_jobs_year_naf_mean["job_category"]=="Finance")|
                                                       (df_salary_jobs_year_naf_mean["job_category"]=="Scientist")]


# In[158]:


cmap = plt.cm.cividis

fig, ax = plt.subplots(figsize=(20,10))
df_salary_jobs_year_naf_mean.groupby(["year","job_category"]).mean()["gender_gap"].unstack().plot(ax=ax,lw=5,cmap=cmap,alpha=0.7)
plt.legend(bbox_to_anchor=(1.2,1),loc="upper right")
plt.ylabel("Gender pay gap")
plt.xlabel("Year")
plt.rc("axes",labelsize=25)
plt.rc("xtick",labelsize=25)
plt.rc("ytick",labelsize=25)
plt.rc("legend",fontsize=25)
plt.grid(True,alpha=0.2,color="black")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)


# ### Gender pay gap by job-category

# ### Northern Africa and Sub-Saharan Africa combined

# In[159]:


df_salary_jobs_af_mean=df_salary_af.groupby(["job_category","gender"],as_index=False).mean()

df_salary_jobs_af_mean=df_salary_jobs_af_mean.pivot(index="job_category",columns="gender",values="salary")

df_salary_jobs_af_mean=df_salary_jobs_af_mean.reset_index()


# In[160]:


fig, ax = plt.subplots(figsize=(20,10))

plt.hlines(y=df_salary_jobs_af_mean["job_category"], xmin=df_salary_jobs_af_mean["female"], xmax=df_salary_jobs_af_mean["male"], color="black", alpha=0.5)
plt.scatter(df_salary_jobs_af_mean["female"], df_salary_jobs_af_mean["job_category"],s=200,marker="D",cmap="cividis", alpha=0.5, label="female")
plt.scatter(df_salary_jobs_af_mean["male"], df_salary_jobs_af_mean["job_category"],s=200,marker="X",cmap="cividis", alpha=0.5 , label="male")
plt.xlabel("Average salary")
plt.ylabel("Job categories")
plt.legend(bbox_to_anchor=(1.2,1),loc="upper right")
plt.grid(True,alpha=0.2,color="black")
plt.rc("axes",labelsize=25)
plt.rc("xtick",labelsize=25)
plt.rc("ytick",labelsize=25)
plt.rc("legend",fontsize=25)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)


# ### Sub-Saharan Africa

# In[161]:


df_salary_jobs_subaf_mean=df_salary_subaf.groupby(["job_category","gender"],as_index=False).mean()

df_salary_jobs_subaf_mean=df_salary_jobs_subaf_mean.pivot(index="job_category",columns="gender",values="salary")

df_salary_jobs_subaf_mean=df_salary_jobs_subaf_mean.reset_index()


# In[162]:


fig, ax = plt.subplots(figsize=(20,10))

plt.hlines(y=df_salary_jobs_subaf_mean["job_category"], xmin=df_salary_jobs_subaf_mean["female"], xmax=df_salary_jobs_subaf_mean["male"], color="black", alpha=0.5)
plt.scatter(df_salary_jobs_subaf_mean["female"], df_salary_jobs_subaf_mean["job_category"],s=200,marker="D",cmap="cividis", alpha=0.5, label="female")
plt.scatter(df_salary_jobs_subaf_mean["male"], df_salary_jobs_subaf_mean["job_category"],s=200,marker="X",cmap="cividis", alpha=0.5 , label="male")
plt.xlabel("Average salary")
plt.ylabel("Job categories")
plt.legend(bbox_to_anchor=(1.2,1),loc="upper right")
plt.grid(True,alpha=0.2,color="black")
plt.rc("axes",labelsize=25)
plt.rc("xtick",labelsize=25)
plt.rc("ytick",labelsize=25)
plt.rc("legend",fontsize=25)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)


# ### Northern Africa

# In[163]:


df_salary_jobs_naf_mean=df_salary_naf.groupby(["job_category","gender"],as_index=False).mean()

df_salary_jobs_naf_mean=df_salary_jobs_naf_mean.pivot(index="job_category",columns="gender",values="salary")

df_salary_jobs_naf_mean=df_salary_jobs_naf_mean.reset_index()


# In[164]:


fig, ax = plt.subplots(figsize=(20,10))

plt.hlines(y=df_salary_jobs_naf_mean["job_category"], xmin=df_salary_jobs_naf_mean["female"], xmax=df_salary_jobs_naf_mean["male"], color="black", alpha=0.5)
plt.scatter(df_salary_jobs_naf_mean["female"], df_salary_jobs_naf_mean["job_category"],s=200,marker="D",cmap="cividis", alpha=0.5, label="female")
plt.scatter(df_salary_jobs_naf_mean["male"], df_salary_jobs_naf_mean["job_category"],s=200,marker="X",cmap="cividis", alpha=0.5 , label="male")
plt.xlabel("Average salary")
plt.ylabel("Job categories")
plt.legend(bbox_to_anchor=(1.2,1),loc="upper right")
plt.grid(True,alpha=0.2,color="black")
plt.rc("axes",labelsize=25)
plt.rc("xtick",labelsize=25)
plt.rc("ytick",labelsize=25)
plt.rc("legend",fontsize=25)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)


# ## Gender pay gap by job-role

# ### Northern Africa and Sub-Saharan Africa combined

# In[165]:


df_salary_role_af_mean=df_salary_af.groupby(["role_k50","gender"],as_index=False).mean()

df_salary_role_af_mean=df_salary_role_af_mean.pivot(index="role_k50",columns="gender",values="salary")

df_salary_role_af_mean=df_salary_role_af_mean.reset_index()

df_salary_role_af_mean["gender_gap"]=df_salary_role_af_mean["male"]-df_salary_role_af_mean["female"]


# In[166]:


cmap = plt.cm.cividis

fig, ax = plt.subplots(figsize=(20,10))
df_salary_role_af_mean.plot(x="role_k50",y=["gender_gap"],kind="bar",ax=ax,cmap=cmap,alpha=0.5)
plt.legend(bbox_to_anchor=(1.13,1),loc="upper right")
plt.ylabel("Gap in pay")
plt.xlabel("Job roles")
plt.grid(True,alpha=0.2,color="black")
plt.rc("axes",labelsize=25)
plt.rc("xtick",labelsize=25)
plt.rc("ytick",labelsize=25)
ax.get_legend().remove()


# ### Sub-Saharan Africa

# In[167]:


df_salary_role_subaf_mean=df_salary_subaf.groupby(["role_k50","gender"],as_index=False).mean()

df_salary_role_subaf_mean=df_salary_role_subaf_mean.pivot(index="role_k50",columns="gender",values="salary")

df_salary_role_subaf_mean=df_salary_role_subaf_mean.reset_index()

df_salary_role_subaf_mean["gender_gap"]=df_salary_role_subaf_mean["male"]-df_salary_role_subaf_mean["female"]


# In[168]:


cmap = plt.cm.cividis

fig, ax = plt.subplots(figsize=(20,10))
df_salary_role_subaf_mean.plot(x="role_k50",y=["gender_gap"],kind="bar",ax=ax,cmap=cmap,alpha=0.5)
plt.legend(bbox_to_anchor=(1.13,1),loc="upper right")
plt.ylabel("Gap in pay")
plt.xlabel("Job roles")
plt.grid(True,alpha=0.2,color="black")
plt.rc("axes",labelsize=25)
plt.rc("xtick",labelsize=25)
plt.rc("ytick",labelsize=25)
ax.get_legend().remove()


# ### Northern Africa

# In[169]:


df_salary_role_naf_mean=df_salary_naf.groupby(["role_k50","gender"],as_index=False).mean()

df_salary_role_naf_mean=df_salary_role_naf_mean.pivot(index="role_k50",columns="gender",values="salary")

df_salary_role_naf_mean=df_salary_role_naf_mean.reset_index()

df_salary_role_naf_mean["gender_gap"]=df_salary_role_naf_mean["male"]-df_salary_role_naf_mean["female"]


# In[170]:


cmap = plt.cm.cividis

fig, ax = plt.subplots(figsize=(20,10))
df_salary_role_naf_mean.plot(x="role_k50",y=["gender_gap"],kind="bar",ax=ax,cmap=cmap,alpha=0.5)
plt.legend(bbox_to_anchor=(1.13,1),loc="upper right")
plt.ylabel("Gap in pay")
plt.xlabel("Job roles")
plt.grid(True,alpha=0.2,color="black")
plt.rc("axes",labelsize=25)
plt.rc("xtick",labelsize=25)
plt.rc("ytick",labelsize=25)
ax.get_legend().remove()


# ## Gender pay gap by seniority

# ### Northern Africa and Sub-Saharan Africa combined

# In[171]:


df_salary_seniority_af_mean=df_salary_af.groupby(["seniority","gender"],as_index=False).mean()

df_salary_seniority_af_mean=df_salary_seniority_af_mean.pivot(index="seniority",columns="gender",values="salary")

df_salary_seniority_af_mean=df_salary_seniority_af_mean.reset_index()

df_salary_seniority_af_mean["gender_gap"]=df_salary_seniority_af_mean["male"]-df_salary_seniority_af_mean["female"]


# In[172]:


cmap = plt.cm.cividis

fig, ax = plt.subplots(figsize=(20,10))

(markers, stemlines, baseline) = plt.stem(df_salary_seniority_af_mean["seniority"],df_salary_seniority_af_mean["gender_gap"],linefmt="grey")
plt.setp(stemlines,linewidth=5)
plt.setp(stemlines,"linestyle","dotted")
plt.setp(markers, marker='D', markersize=30, markeredgecolor="orange", markeredgewidth=2)
plt.rc("axes",labelsize=25)
plt.rc("xtick",labelsize=25)
plt.rc("ytick",labelsize=25)
plt.rc("legend",fontsize=25)
plt.grid(True,alpha=0.2,color="black")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
plt.xlabel("Seniority level")
plt.ylabel("Gender gap in salaries")


# ### Sub- Saharan Africa

# In[173]:


df_salary_seniority_subaf_mean=df_salary_subaf.groupby(["seniority","gender"],as_index=False).mean()

df_salary_seniority_subaf_mean=df_salary_seniority_subaf_mean.pivot(index="seniority",columns="gender",values="salary")

df_salary_seniority_subaf_mean=df_salary_seniority_subaf_mean.reset_index()

df_salary_seniority_subaf_mean["gender_gap"]=df_salary_seniority_subaf_mean["male"]-df_salary_seniority_subaf_mean["female"]


# In[174]:



cmap = plt.cm.cividis

fig, ax = plt.subplots(figsize=(20,10))

(markers, stemlines, baseline) = plt.stem(df_salary_seniority_subaf_mean["seniority"],df_salary_seniority_subaf_mean["gender_gap"],linefmt="grey")
plt.setp(stemlines,linewidth=5)
plt.setp(stemlines,"linestyle","dotted")
plt.setp(markers, marker='D', markersize=30, markeredgecolor="orange", markeredgewidth=2)
plt.rc("axes",labelsize=25)
plt.rc("xtick",labelsize=25)
plt.rc("ytick",labelsize=25)
plt.rc("legend",fontsize=25)
plt.grid(True,alpha=0.2,color="black")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
plt.xlabel("Seniority level")
plt.ylabel("Gender gap in salaries")


# ### Northern Africa

# In[175]:


df_salary_seniority_naf_mean=df_salary_naf.groupby(["seniority","gender"],as_index=False).mean()

df_salary_seniority_naf_mean=df_salary_seniority_naf_mean.pivot(index="seniority",columns="gender",values="salary")

df_salary_seniority_naf_mean=df_salary_seniority_naf_mean.reset_index()

df_salary_seniority_naf_mean["gender_gap"]=df_salary_seniority_naf_mean["male"]-df_salary_seniority_naf_mean["female"]


# In[176]:


cmap = plt.cm.cividis

fig, ax = plt.subplots(figsize=(20,10))

(markers, stemlines, baseline) = plt.stem(df_salary_seniority_naf_mean["seniority"],df_salary_seniority_naf_mean["gender_gap"],linefmt="grey")
plt.setp(stemlines,linewidth=5)
plt.setp(stemlines,"linestyle","dotted")
plt.setp(markers, marker='D', markersize=30, markeredgecolor="orange", markeredgewidth=2)
plt.rc("axes",labelsize=25)
plt.rc("xtick",labelsize=25)
plt.rc("ytick",labelsize=25)
plt.rc("legend",fontsize=25)
plt.grid(True,alpha=0.2,color="black")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
plt.xlabel("Seniority level")
plt.ylabel("Gender gap in salaries")

