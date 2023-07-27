#Pandas
import pandas as pd
import numpy as np 

iris =   pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
print(type(iris))

iris =   pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
iris

df = iris               #copies the reference so any change in one will reflect in the other 
df = iris.copy()        #copies the datas only

df.head()               #look at few initial enteries, by default it shows first five

df.head(3)

df.columns = ['sl', 'sw', 'pl', 'pw', 'flower_type']


print(df.shape)
print(df.dtypes)

df.describe()

df.sl
#df["sl"]   can be used also 

df.isnull()

df.isnull().sum()

# number of null enteries in respective columns

#access a slice/part of the data

print(df.iloc[8:15, 2:4])       #works by position not index
df.loc[12]                      #works by label not position 

df.head()

#droping by label

a = df.drop(1)  # copies the data deleting 1st row 

df.drop(1, inplace = True)  # copies the data deleting 1st row in the dataset itself .  
df.head()

df.index

df.index[1]  #shows the label/index of the position 1

#Droping by position 
df.drop(df.index[1], inplace = True)
df.head()


df.drop(df.index[[2,3]], inplace = True)         #remove more than one entries
df.head()

df.sl > 5      #which rows has this condition true or false

#only those rows for which the condition is true 
df[df.sl > 5] 

df[df.flower_type == "Iris-virginica"].describe()

df.head()

print(df.iloc[1])      #non labelled by position 
print(df.loc[3])       #labelled  by index

#Add the row at the end with index 

df.loc[2] = [2.1,1.2,4.5,5.1, "iris"]
df.tail()

#resets the index 
df.reset_index()

#reset the index droping previous indices
df.reset_index(drop = True, inplace = True)      #inorder to make changes in df itself set inplace to True 
df.head()

df.index

#delete a columnm 

df.drop("sl", axis = 1, inplace = True)      #axis 1 means look column wise

df.head()

del df["pl"]
df.describe()

df = iris.copy()
df.columns = ['sl', 'sw', 'pl', 'pw', 'flower_type']
df.describe()

df["diff_pl_pw"] = df["pl"] - df["pw"]
df.tail()

df.iloc[2:4,1:3] = np.nan
df.head()

df.describe()

df.dropna(inplace = True)      #drop rows having NaN enteris.

df.head()

df.reset_index(drop = True, inplace = True)
df.head()

df.iloc[2:4,1:3] = np.nan
df.head()

#fill NaN enteries

df.sw.fillna(df.sw.mean(), inplace = True )
df.pl.fillna(df.pl.mean(), inplace = True )
df.head()

#String based Data
df["Gender"] = "Female"
df.iloc[2:10, 5] = "Male"
df.head()

def f(s):
    if s == "Male":
        return 0
    else:
        return 1
df["Sex"] =df.Gender.apply(f)       #apply rule/function f
df.head()

df.dtypes



