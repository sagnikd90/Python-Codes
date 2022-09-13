#!/usr/bin/env python
# coding: utf-8

# # Python programming practice

# In[1]:


Name_1="Sagnik Das"


# In[2]:


Name_1[1]


# In[3]:


Name_1[0]


# In[4]:


Name_1[6]


# In[5]:


Name_1[7]


# In[6]:


Name_1[-10]


# In[7]:


Name_1[-1]


# ## Separating strings

# In[10]:


first_part=Name_1[0:4]
second_part=Name_1[7:10]
print(first_part,second_part)


# ## Suppose we want to choose every second variable in the string

# In[11]:


every_second=Name_1[::2]
print(every_second)


# ## Now suppose we want to separate every second value upto index 6

# In[12]:


upto_6 = Name_1[0:6:2]
print(upto_6)


# ## Computing the length of a string

# In[13]:


len(Name_1)


# ## Concatenating strings

# In[18]:


first_name=Name_1[0:7]
second_name=Name_1[7:10]
full_name = first_name + second_name
print(full_name + " is an asshole")


# ## Escape sequences in strings

# ### Suppose we want to separate the string and write it in two lines

# In[23]:


print("Sagnik\nDas")


# ### Suppose we want to separate the string by a tab

# In[24]:


print("Sagnik\tDas")


# ### Uppercasing strings

# In[25]:


Name_1.upper()


# ### Lower casing strings

# In[26]:


Name_1.lower()


# ### Replacing a segment of the string

# In[27]:


Name_1.replace("Sagnik", "Satadru")


# In[28]:


Name_1.replace(Name_1[0:6], "Satadru")


# ### Finding the index of a string

# In[32]:


Name_1.find("Das")


# # Understanding lists and tuples in Python

# ## Tuples

# #### Tuples are an ordered sequence.
# #### Tuples are written as comma separated values within a parenthesis.
# #### Example of a tupple

# In[33]:


tup_1=(1,2,4,2,5,7,3,7,0,3,6)


# #### A tuple in python can contain all types of variables including string, integer and float. In the following example, the tupple consists of all the three types of variables

# In[38]:


tup_2=("Sagnik",1,1.15)


# #### Using indexing to access the elements of the tuple

# In[39]:


print(tup_2[0])
print(tup_2[1])
print(tup_2[2])


# #### Concatenating tuples in python

# In[40]:


tup_3=tup_1+tup_2
print(tup_3)


# #### Slicing the tuples

# In[42]:


tup_3[0:8]


# In[47]:


tup_3[9:11]


# #### Sorting a tuple

# In[51]:


tup_1=sorted(tup_1)
print(tup_1)


# #### Indexing complex tuples: Nested tuples

# In[52]:


tup_4=(6,8,2,3,("sagnik","satadru"),8.6,4,5,("bipasha","pranab"))
tup_4[0:5]


# In[55]:


print(tup_4[4][0])
print(tup_4[4][1])


# ## Lists in python

# ### Lists are also ordered sequences like tuples
# ### Lists are represented by using square brackets.
# ### An example of a list is shown below

# In[57]:


list_1=["Sagnik Das", "Satadru Das",1,2,3,4,5,6,7,8,9]


# ### While tuples are not immutable, lists are immutable. A list can contain a string, an integer and a float. We can even nest other lists within a list and also other tuples.
# 
# #### An example of such a list is

# In[60]:


list_2=[2,3,4,[7,9,10],("Sagnik","Satadru"),(8,15,20),["Bipasha","Pranab"]]
list_2


# ### Indexing lists

# In[81]:


print(list_2[4][0])
print(list_2[5][0])
print(list_2[6][0])
print(list_2[0:6][4][0])


# ### Splitting a string to a list

# In[92]:


string_A="Sagnik is an asshole"
list_A=string_A.split()
list_A


# ### Splitting a string by commas

# In[95]:


list_B="Sagnik,Satadru,Pranab,Bipasha".split(",")
print(list_B)


# ### Cloning lists in python

# In[97]:


list_C=list_B[:]
print(list_C)


# ## Dictionaries in python

# ### Dictionaries in python are similar to lists. For lists, we have indexes, which have to be integers. For dictionaries, we have keys in place of index. These keys do not necessarily have to be integers. We can have non-integers as keys for a dictionary.

# #### Let us now build a dictionary and see how to work with a python dictionary

# In[98]:


dict_1={"Sagnik":1990,"Satadru":1983,"Bipasha":1961,"Pranab":1948}
dict_1


# #### Let us know check the keys of the dictionary

# In[99]:


dict_1.keys()


# #### Now we check the values in the dictionary

# In[100]:


dict_1.values()


# #### We can check individual values assigned to each key

# In[103]:


print(dict_1["Sagnik"],
     dict_1["Satadru"],
     dict_1["Bipasha"],
     dict_1["Pranab"])


# #### We can also add a new element to the dictionary by assigning it a new key

# In[110]:


dict_1["Debopriti"]=1995
dict_1


# #### We can search for an element of the dictionary by putting in it's key

# In[105]:


"Debopriti" in dict_1


# In[106]:


"Shilpi" in dict_1


# #### We can also delete an element of the dictionary

# In[111]:


del(dict_1["Debopriti"])
dict_1


# ## Sets in Python

# #### Sets are similar to dictionaries or lists, but the elements of a set are unordered. There are no indices or keys.
# #### The following is an example of a couple of sets

# In[120]:


A={"Lionel Messi",
   "Cristiano Ronaldo",
  "Luis Figo",
  "Zinedine Zidane",
  "Ronaldo Luiz Nazario De Lima"}


# In[123]:


B={"Lionel Messi",
  "David Beckham",
  "Cristiano Ronaldo",
  "Luis Figo",
  "Andrea Pirlo"}


# #### Now we can add the elements of one set to the other

# In[125]:


A.add("Luis Suarez")
A


# #### We can perform the different set operations on sets in Python

# In[127]:


intersect=A&B
union=A.union(B)

print(intersect,union)


# #### We can check if one set is a subset of the other

# In[129]:


B.issubset(A)


# #### We can turn a list into a set

# In[131]:


set_1=set(list_1)
set_1


# ## Conditions and branching in Python

# ### Using conditions in Python

# In[1]:


a=8
a==7


# In[2]:


a>8
a==7


# In[7]:


a=8
a>=7


# In[9]:


a=6
a<7


# In[10]:


"Lionel Messi"=="Cristiano Ronaldo"


# In[11]:


"Lionel Messi"!="Cristiano Ronaldo"


# ### Branching in Python

# #### Using the if statement in python

# In[13]:


goal_contributions=500

if goal_contributions>=400:
    print("Amazing player")


# #### Using the if else statement in python

# In[14]:


goal_contributions=500

if goal_contributions>500:
    print("Amazing player")
else:
    print("Average player")


# #### Using the elif statement in python

# In[17]:


goal_contributions=500

if goal_contributions>500:
    print("Amazing player")
elif goal_contributions<500:
    print("Bad player")
else:
    print("Average player")


# #### Using conditions and branching together

# In[19]:


goal_contributions=500

if goal_contributions>400 or goal_contributions<500:
    print("Average player")
elif goal_contributions>=500 or goal_contributions<600:
    print("Amazing player")
else:
    print("Bad player")


# In[21]:


goal_contributions=500

if goal_contributions>400 and goal_contributions<500:
    print("Average player")
elif goal_contributions>=500 and goal_contributions<600:
    print("Amazing player")
else:
    print("Bad player")


# ## Loops in python

# #### For loops in python

# #### Let us first create a list

# In[22]:


list_1=["Lionel Messi",
        "Cristiano Ronaldo",
        "Luis Figo",
       "Zinedine Zidane",
       "Marco Reus",
       "Ronaldinho Gaucho",
       "Paulo Maldini"]


# In[30]:


for i in range(0,7):
    list_1[i]="Football player"
    
list_1    


# In[39]:


list_1=["Lionel Messi",
        "Cristiano Ronaldo",
        "Luis Figo",
       "Zinedine Zidane",
       "Marco Reus",
       "Ronaldinho Gaucho",
       "Paulo Maldini"]
list_1


# In[47]:


for i in list_1:
    i    
i    


# #### Finding the index of the elements in the list by using enumerate

# In[50]:


for i, players in enumerate(list_1):
    players
    i
print(players,i)    


# In[56]:


list_2=["red",
        "blue",
        "yellow",
        "green",
        "brown",
        "white"]

print(list_2)

for colors in list_2:
    print(colors)


# #### Printing the index along with the values in a for loop

# In[58]:


index=0

for colors in list_2:
    print(index,colors)
    index+=1 ## Updating the index


# #### Another way of printing the index along with the values is using the range function of python. If we use the range function for indexing, we don't need to update the index

# In[59]:


for index in range(len(list_2)):
    colors=list_2[index]
    print(index,colors)


# In[68]:


list_3 = ["Lionel Messi",
        "Cristiano Ronaldo",
        "Luis Figo",
       "Zinedine Zidane",
       "Marco Reus",
       "Ronaldinho Gaucho",
       "Paulo Maldini"]
i=0

for i in range(len(list_1)):
    players=list_1[i]
    print(i,players)


# #### Using enumerate

# In[69]:


for i,players in enumerate(list_1):
    print(i,players)


# In[70]:


list_4=["coldplay",
       "pink floyd",
       "green day",
       "nirvana",
       "the doors",
       "iron maiden",
       "metallica"]


# In[71]:


for i in range(len(list_4)):
    bands=list_4[i]
    print(i,bands)


# In[72]:


for i, bands in enumerate(list_4):
    print(i,bands)


# In[73]:


list_1=["red",
       "blue",
       "green",
       "yellow",
       "brown",
       "purple"]

for i in range(len(list_1)):
    colors=list_1[i]
    print(i,colors)


# In[74]:


for i,colors in enumerate(list_1,start=1):
    print(i,colors)


# #### Using the while loop

# #### Unlike for loop, a while loop runs only if a mentioned condition is met

# In[81]:


list_1=["red",
       "green",
       "yellow",
       "white",
       "yellow",
       "brown",
       "black",
       "orange",
       "green",
       "yellow",
       "red",
       "magenta",
       "violet"]

newlist=[]

i=0

while list_1[i]=="yellow" or list_1[i]=="green" or list_1[i]=="red":
    newlist.append(list_1[i])
    i=i+1
    
print(newlist)    


# ### Using for loop and while loop to select even numbers between 0 and 100

# #### For loops

# In[1]:


list_1=list(range(20))


# In[2]:


newlist=[]

i=0

for i in range(len(list_1)):
    if i%2==0:
        newlist.append(list_1[i])
newlist        


# #### While loops

# In[3]:


newlist=[]

i=0

while i<len(list_1):
    if list_1[i]%2==0:
        newlist.append(list_1[i])
    i=i+1 
    
newlist    


# #### Now selecting odd numbers

# #### For loops

# In[4]:


newlist=[]

i=0

for i in range(len(list_1)):
    if i%2!=0:
        newlist.append(list_1[i])
newlist        


# #### While loops

# In[5]:


newlist=[]

i=0

while i<len(list_1):
    if list_1[i]%2!=0:
        newlist.append(list_1[i])
    i=i+1 
    
newlist    


# ## Functions in python

# #### Writing simple functions in Python

# In[9]:


def sagnik(fname):
    print(fname + "Das")
    
sagnik("Sagnik ")
sagnik("Satadru ")
sagnik("Bipasha ")
sagnik("Pranab ")


# #### A function always has to be called with the correct number of arguments. If the number of arguments are misspecified, then the function will return an error. If we are not sure about the number of arguments to be passed, we use the * to indicate the unknown number of arguments

# In[12]:


def sagnik(*fname):
    print("The youngest child is " + fname[2])
    
sagnik("Bipasha","Satadru","Sagnik","Pranab")    


# #### Passing a list as an argument in a function

# In[13]:


def func_1(color):
    for i in color:
        print(i)
color=["red","green","brown"]

func_1(color)


# #### A simple function to compute the mean of a list of variables

# In[32]:


def avg(list):
    return sum(list)/len(list)

list_1=list(range(100))

avg(list_1)


# #### A simple function which returns the absolute value of a number

# In[26]:


def abs_num(num):
    if num>0:
        return num
    else:
        return -num
    
abs_num(-5)    


# #### A simple function to check if a number is a prime number or not

# In[52]:


def prime_finder(num):
    if num>=2:                 ## The number has to be greater than 1 and 2 even though 1 and 2 are obvious prime numbers
        for y in range(2,num): ## We choose a number between 2 and the input number for which we are checking the prime status
                               ## We do so because we want to see if there are any numbers between 2 and the input number which
                               ## is a factor of the input number. If there is such a number then the input number is not a prime
                               ## number.
            if num%y==0:       ## We check if the input value is divisible by any number between 2 and the input number.
                return False   ## If there exists such a number, then the function returns the value False.
            else:              ## If not, it will return true.   
                return True
            
prime_finder(23)            


# #### A simple function to compute x^n with a recurrence relation

# In[51]:


def power_computer(x,n):
    if n==0:
        return 1
    else:
        return (x**(n)) 
    
power_computer(100,-2)    


# In[ ]:




