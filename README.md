# EX-05-Feature-Generation


## AIM
To read the given data and perform Feature Generation process and save the data to a file. 

# Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Generation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
```
Program Developed: RAJESHKANNAN M
Register number:212221230082

```
### Data.csv
```
import pandas as pd
df=pd.read_csv("data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce
be=ce.BinaryEncoder()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()
oe=OrdinalEncoder()


df1["City"] = ohe.fit_transform(df1[["City"]])

temp=['Cold','Warm','Hot','Very Hot']
oe1=OrdinalEncoder(categories=[temp])
df1['Ord_1'] = oe1.fit_transform(df1[["Ord_1"]])

edu=['High School','Diploma','Bachelors','Masters','PhD']
oe2=OrdinalEncoder(categories=[edu])
df1['Ord_2']= oe2.fit_transform(df1[["Ord_2"]])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5

```

### Encoding .csv
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
le=LabelEncoder()
oe=OrdinalEncoder()

df1["nom_0"] = oe.fit_transform(df1[["nom_0"]])
temp=['Cold','Warm','Hot']
oe2=OrdinalEncoder(categories=[temp])
df1['ord_2'] = oe2.fit_transform(df1[['ord_2']])

df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df0=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df0

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df2=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df2

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df3=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df3

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df4=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df4
```

### Titanic.csv
```
import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df

#removing unwanted data
df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)

#data cleaning
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df.isnull().sum()

df

#feature encoding
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
ndf=be.fit_transform(df["Sex"])
ndf

df1=df.copy()
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
e1=OrdinalEncoder(categories=[embark])
df1['Embarked'] = e1.fit_transform(df[['Embarked']])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df5
```

# OUTPUT
## Data.csv


### Initial Dataset:
![1](https://user-images.githubusercontent.com/93901857/195382562-e891c15d-2358-4407-88e3-ea54e225030e.jpg)

### Binary Encoding:
![2](https://user-images.githubusercontent.com/93901857/195382578-862551b4-da34-4f54-bcd9-6a1432b6b001.jpg)

### Encoded Dataset:
![3](https://user-images.githubusercontent.com/93901857/195383733-b2cf9b3f-fa07-4653-a4b6-e6c1e864e537.jpg)

### Data Scaling using MinMaxScaler:
![4](https://user-images.githubusercontent.com/93901857/195382213-2af5f193-7fa3-4027-ae1f-1b04312164dc.jpg)

### Data Scaling using StandardScaler:
![5](https://user-images.githubusercontent.com/93901857/195382225-e0f65d76-06c5-4b16-9935-25bfd119023c.jpg)

### Data Scaling using MaxAbsScaler:
![6](https://user-images.githubusercontent.com/93901857/195382232-0e924370-1189-4c46-a428-866fafa41808.jpg)

### Data Scaling using RobustScaler:
![7](https://user-images.githubusercontent.com/93901857/195382244-81a0a8a9-2ca2-423b-b5ac-b5203321b95a.jpg)



## Encoding.csv

### Initial Dataset:
![8](https://user-images.githubusercontent.com/93901857/195382253-790ea6fa-c988-4d30-a2c7-2e53ba82593f.jpg)

### Binary Encoding:
![9](https://user-images.githubusercontent.com/93901857/195382267-034a879a-262b-491a-a180-27299ce435f3.jpg)

### Encoded Dataset:
![10](https://user-images.githubusercontent.com/93901857/195382309-31784377-121b-44d5-80da-c7f9148e92d1.jpg)

### Data Scaling using MinMaxScaler:
![11](https://user-images.githubusercontent.com/93901857/195382320-9325484a-3e53-471c-8d49-9452641f6473.jpg)

### Data Scaling using StandardScaler:
![12](https://user-images.githubusercontent.com/93901857/195383596-a0420c14-a061-4e19-a4fa-188c8961a575.jpg)

### Data Scaling using MaxAbsScaler:
![13](https://user-images.githubusercontent.com/93901857/195383620-f25d94b3-d990-4887-8b6f-23edfef8ea05.jpg)

### Data Scaling using RobustScaler:
![14](https://user-images.githubusercontent.com/93901857/195383629-b6947a61-676e-48f8-9dd7-92c7d9221ba3.jpg)



## Titanic.csv

### Initial Dataset:
![15](https://user-images.githubusercontent.com/93901857/195383636-f5507529-1e09-4af8-95e2-b09f8772292c.jpg)

### Data cleaning before encoding:
![16](https://user-images.githubusercontent.com/93901857/195382330-0813e2ff-74b5-4984-adf2-96491b29ee9c.jpg)

### Cleaned Dataset:
![17](https://user-images.githubusercontent.com/93901857/195382342-bb480c51-c916-4c9e-8dbe-9d74438729e9.jpg)

### Binary Encoding:
![18](https://user-images.githubusercontent.com/93901857/195382347-62bed691-2370-4490-845a-0d541c1e4936.jpg)

### Encoded Dataset:
![19](https://user-images.githubusercontent.com/93901857/195382353-7b19b5e6-9a07-4358-abb6-487ef3544a02.jpg)

### Data Scaling using MinMaxScaler:
![20](https://user-images.githubusercontent.com/93901857/195382366-43002bba-9fe4-4fa2-bae6-69b835c28812.jpg)

### Data Scaling using StandardScaler:
![21](https://user-images.githubusercontent.com/93901857/195382384-cb002403-84aa-4e93-a275-c0a3ec056746.jpg)

### Data Scaling using MaxAbsScaler:
![22](https://user-images.githubusercontent.com/93901857/195382391-2acfbd1d-0313-40c2-a733-83daab81fbbc.jpg)

### Data Scaling using RobustScaler:
![23](https://user-images.githubusercontent.com/93901857/195382403-265c3263-ac2d-4027-97f5-2c8233d544ff.jpg)


# RESULT 
Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.



