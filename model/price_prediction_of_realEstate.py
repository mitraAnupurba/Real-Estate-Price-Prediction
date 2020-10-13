#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)


## data cleaning begins here :


df1 = pd.read_csv(r"C:\Users\Admin\Desktop\pandas\Bengaluru_House_Data.csv")
print("============================================================================")
print('the data frame created is : ',df1)
print("the dimensions of the initial dataframe is : ",df1.shape)
print("============================================================================")


# In[4]:
print("============================================================================")
print("printing the data frame again :",df1)
print("============================================================================")

# In[3]:


grp = df1.groupby('area_type')['area_type'].agg('count')
print("the number of rows based on area type is : ",grp)

# In[7]:


df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')
print("============================================================================")
print("the data frame after deopping the above columns is : ",df2)
print("============================================================================")

# In[8]: function to get the number of NaN values under each column in every row.

print("============================================================================")
print("the number of NaN values under each column on all the rows is : ",df2.isnull().sum())
print("============================================================================")


#dropping the NaN values under location columns :
df3 = df2.dropna(subset=['location'])

#checking if nan values have been dropped correctly :
print("============================================================================")
print("now the number of NaN values under the location columns : ",df3.isnull().sum())
print("============================================================================")

print("============================================================================")
print("checking the number of nan values under the bath column : ")
print(df3['bath'].isnull().sum())
df3['bath'].isnull().sum()
print("============================================================================")

# we will fill the NaN values in the bath column with median value of the bath column.
print("============================================================================")
print("the median of bath column is : ",df3['bath'].median())
print("============================================================================")

df4 = df3.fillna({
    'bath' : 2.0
})

print("the output after filling up the bath column with the median value :",df4.isnull().sum())

#for the location and size columns, we will drop the NaN values :
df5 = df4.dropna(subset=['location','size'])
print("============================================================================")
print("after droping the NaN values for location and size column is :",df5)
print("============================================================================")
print("now the NaN values are : ",df5.isnull().sum())
print(df5.shape)
print("============================================================================")

#checking all the different values that are present in the size column, there could be 4 bedroom and 4 BHK
# which are the same values. thus we must clean the data.

print("all the unique values in the size column are : ",df5['size'].unique())

#we are creating a new column in the dataframe to store only the number of rooms :
df5['BHK'] = df5['size'].apply(lambda x: int(x.split(' ')[0]))

#the new data frame is :
print("============================================================================")
print("the cleaned data frame : ",df5.head())
print("============================================================================")

#now checking all the unique values in the newly created BHK column :
print("the unique values in the BHK column is : ",df5['BHK'].unique())


# In[34]:


df5[df5['BHK'] > 20]


# cleaning the total_sqft column :

print("============================================================================")
print("all the unique values in the total_sqft column is : ",df5['total_sqft'].unique())
print("============================================================================")

# function to convert all the values in the total_sqft column to float
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

print("============================================================================")
print(df5[~df5['total_sqft'].apply(is_float)])
print("============================================================================")

# function to convert the range type values in the total_sqft column to float
# by taking average of the end values of the range .

def convert_sqft_range_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try : 
        return float(x)
    except: 
        return True
    

#checking if the function works correctly :
print("============================================================================")
print(convert_sqft_range_to_num('2134'))
print(convert_sqft_range_to_num('2100-2300'))
print("============================================================================")



# In[41]:


df6 = df5.copy()
df6['total_sqft'] = df6['total_sqft'].apply(convert_sqft_range_to_num)
print("============================================================================")
print("dataframe after taking care of the range type values in the total_sqft column : ",df6)
print("============================================================================")


#creating a deep copy fof df6 in df7 :
df7 = df6.copy()
print("============================================================================")
print("the deep copy of df6 in df7 is : ",df7)
print("============================================================================")


## feature engineering begins here :

#adding a new feature : price_per_sqft and price_per_sqft_lacks:
df7['price_per_sqft'] = (df7['price(Lacks)'].div(df7['total_sqft']))
print("============================================================================")
print("the price per sqft is : ",df7)
print("============================================================================")

#converting price per sqft to price per sqft in lacks  by multiplying the erlier value by 100000

df8 = df7.copy()
df8['price_per_sqft_lacks'] = df7['price_per_sqft'].mul(100000)
print("============================================================================")
print("the price per sqft is : ",df8)
print("============================================================================")




df8 = df7.copy()
df8['price_per_sqft_lacks'] = df7['price_per_sqft'].mul(100000)
df8


# dropping the price_per_sqft column as we dont need it :
df8 = df8.drop(['price_per_sqft'],axis='columns')
print("============================================================================")
print("the price per sqft is : ",df8)
print("============================================================================")

#there are a lot of different categpries under the locatoion column, ut
#some of these categories have only one or two values.
#we will put all such categories into one banner : "OTHER "

# checking all the unique values under the location column :
print("============================================================================")
print("all the unique values in the location column are :",df8['location'].unique())
print("============================================================================")


#we are removing all the extra white spaces, etc from the location column using strip() function :
df8['location'] = df8['location'].apply(lambda x : x.strip())


#storing the df8 dataframe to a csv file for backup :
df8.to_csv(r"C:\Users\Admin\Desktop\pandas\realEstate_price.csv")









location_stats = df8.groupby('location')['location'].agg('count')
print("============================================================================")
print("getting the location_stat variable value : ",location_stats)
print("============================================================================")
print("getting the type of location_stat variable : ",type(location_stats))
print("============================================================================")

#sorting the location_stats in descending order for checking what are the highest and lowest value:
print("============================================================================")
print("location_stats sorted : ",location_stats.sort_values(ascending=False))
print("============================================================================")

# creating a new series type varible to store the locations whose coukt is less than 10"
location_stats_less_than_10 = location_stats[location_stats<=10]
print("============================================================================")
print("the locations where there are less than 10 houses are : ",location_stats_less_than_10)
print("============================================================================")


#for all the values in the location column whose values are in the location_stats_less_than_10
#variable, put them in the other category
df8.location = df8.location.apply(lambda x : 'other' if x in location_stats_less_than_10 else x)

print("============================================================================")
print("now the number of unique values in the location column are : ",df8.location.unique())
print(len(df8.location.unique()))
print("============================================================================")

print("============================================================================")
print("the data frame now is : ",df8)
print("============================================================================")

#if the sqft per room is less than 300, it is most likely an anomaly
#we are checking for all such anomalies using this code:
print("============================================================================")
print("anomalies : no of cases where per sqft size is < 300",df8[(df8.total_sqft/df8.BHK) < 300])
print("============================================================================")

# filtering out such anomalies :
print("============================================================================")
print("the dimensions of the resulting dataframe before removing such anomalies are : ",df8.shape)
df9 = df8[~((df8.total_sqft/df8.BHK) < 300)]
print("the dimensions of the resulting dataframe after removing such anomalies are : ",df9.shape)
print("============================================================================")

print("============================================================================")
# minimum per sqft price :
print("minimum per sqft price",df9.price_per_sqft_lacks.min())

#maximum per sqft price :
print("maximum per sqft price",df9.price_per_sqft_lacks.max())


# standard seviation of per sqft :
print("standard deviation of per sqft price is : ",df9.price_per_sqft_lacks.std())
print("============================================================================")



# function for removing the outliers in orice per sqft column based on the standard deviation :
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft_lacks)
        st =  np.std(subdf.price_per_sqft_lacks)
        reduced_df = subdf[(subdf.price_per_sqft_lacks>(m-st)) & (subdf.price_per_sqft_lacks<(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

df10 = remove_pps_outliers(df9)
print("============================================================================")
print("the dimensions after removing the outliers : ",df10.shape)
print("finally cleaned data : ",df10)
print("============================================================================")

#writing the final value to a excel file for backup :
#we are not writing the index values as we will use this fine in future procedures:
df10.to_csv(r"C:\Users\Admin\Desktop\pandas\after_df10.csv",index=False)

bhk = df10.BHK.unique()
print("the different bhk values are :", bhk)

#function to plot the outliers using scatter plot:
def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.BHK == 2)]
    bhk3 = df[(df.location == location) & (df.BHK == 3)]
    bhk4 = df[(df.location == location) & (df.BHK == 4)]
    bhk5 = df[(df.location == location) & (df.BHK == 5)]
    bhk6 = df[(df.location == location) & (df.BHK == 6)]
    bhk7 = df[(df.location == location) & (df.BHK == 7)]
    bhk8 = df[(df.location == location) & (df.BHK == 8)]
    bhk9 = df[(df.location == location) & (df.BHK == 9)]
    bhk10 = df[(df.location == location) & (df.BHK == 10)]
    bhk11 = df[(df.location == location) & (df.BHK == 11)]
    bhk12 = df[(df.location == location) & (df.BHK == 12)]
    bhk13 = df[(df.location == location) & (df.BHK == 13)]
    bhk16 = df[(df.location == location) & (df.BHK == 16)]
    matplotlib.rcParams["figure.figsize"] = (15, 10)
    plt.scatter(bhk2.total_sqft, (bhk2['price(Lacks)']), color='b', label='2 BHK')
    plt.scatter(bhk3.total_sqft, (bhk3['price(Lacks)']), color='g', marker='+', label='3 BHK')
    plt.scatter(bhk4.total_sqft, (bhk4['price(Lacks)']), color='red',  label='4 BHK')
    plt.scatter(bhk5.total_sqft, (bhk5['price(Lacks)']), color='black', label='5 BHK')
    plt.scatter(bhk6.total_sqft, (bhk6['price(Lacks)']), color='yellow', label='6 BHK')
    plt.scatter(bhk7.total_sqft, (bhk7['price(Lacks)']), color='cyan', label='7 BHK')
    plt.scatter(bhk8.total_sqft, (bhk8['price(Lacks)']), color='magenta', label='8 BHK')
    plt.scatter(bhk9.total_sqft, (bhk9['price(Lacks)']), color='purple', label='9 BHK')
    plt.xlabel('total square feet area')
    plt.ylabel('price')
    plt.title(location)
    plt.legend()
    plt.show()


#plot_scatter_chart(df10, "other")




grp = df10.groupby('location')
locations = df10.location.unique()

#for the sake of data visualisation to identify the outliers, we are plotting a scatter plot
#for all the different locations.but only

#for loca in locations:
#    plot_scatter_chart(df10, loca)

#function to plot a histogram to see the spread of our data in a range
def plot_histogram(df):
    bins = [5000, 10000, 15000, 20000, 25000, 30000, 35000]
    matplotlib.rcParams["figure.figsize"] = (20, 10)
    plt.hist(df.price_per_sqft_lacks, rwidth=0.8)
    plt.xlabel('price per sqft')
    plt.ylabel('count')
    plt.title("histogram for number of total sqft vs price")
    plt.show()


plot_histogram(df10)

#function to plot an histogram to see what number of bathrooms most of our properties have :
def plot_histogram_for_bath(df,bins):
    matplotlib.rcParams["figure.figsize"] = (20,10)
    plt.hist(df.bath,rwidth=0.8)
    plt.xlabel('number of baths')
    plt.ylabel('count')
    plt.title("histogram for bath versus total sqft")
    plt.show()


bins = [2,4,6,8,10,12,14,16]
plot_histogram_for_bath(df10,bins)

#we are filtering out the bathroom feature:
# if any house has number of baths > number of bedrooms+2, we are fiktering it out.
df11 = df10[df10.bath < df10.BHK+2]

#storing df11 for backup:
df11.to_csv(r"C:\Users\Admin\Desktop\pandas\after_df11_dataCleaning_done.csv")

## TRAINING THE ML MODEL USING SKLEARN:

#we are dropping these 2 columns because they are unecessary features
df12 = df11.drop(['size','price_per_sqft_lacks'],axis='columns')
print("the shape of the data frame now after dropping the size,price_per_sqft_lacks: ",df12.shape)

#creating dummies df to replace the location column :
dummies_df = pd.get_dummies(df12.location)
print("the dummies df is : ",dummies_df)

# we can add the dummies data frame to the normal dataframe and dropping the location column :
df13 = pd.concat([df12,dummies_df.drop('other',axis='columns')],axis='columns')
print("the dummies data frame  is : ",df13.head(3))

#dropping the location column :
df14 = df13.drop('location',axis='columns')
print("the data frame after dropping the location column is : ",df14.head(3))
print("the shape of data frame object is ", df14.shape)

#data frame of independent variables :
x_df = df14.drop('price(Lacks)',axis='columns')
print("============================================================================")
print("the data frame of independent variables is : ")
print("============================================================================")
x_df.head(3)
df13.rename(columns={"price(Lacks)": "price"},inplace = True)
print("============================================================================")
print("df 13 after renaming the price column is : ")

print(df13.head(3))
print("============================================================================")

#dataframe of dependent variables :
y_df = df13.price
print("============================================================================")
print("the data frame of dependent variable price is : ")
print(y_df.head(3))
print("============================================================================")


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x_df,y_df,test_size = 0.2,random_state=10)

#we are using linear regression model to predict the outputs:
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,Y_train)
print("============================================================================")
print("the accuracy score of this model is : ",lr_clf.score(X_test,Y_test))
print("============================================================================")


#we will use K-cross validation to check the accuracy of out model in various shuffle splits :
from sklearn.model_selection import *

cv = ShuffleSplit(n_splits=10,test_size=0.2,random_state=0)
print("the cross validation scores are : ",cross_val_score(LinearRegression(),x_df,y_df,cv=cv))

#we will compare our given model with two other algorithms :
#lasso and decision tree, and provide the output in a dataframe:
from sklearn.linear_model import *
from sklearn.tree import *

#function to find the best algorithm for our price prediction :
def find_best_algo_to_determine_price_using_gridsearchcv(X, Y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['mse', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }

    }

    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(x_df, y_df)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
    df.to_csv(r"C:\Users\Admin\Desktop\pandas\acc_scores.csv",)
    return df


find_best_algo_to_determine_price_using_gridsearchcv(x_df, y_df)

#function to predict prics:
def predict_price(location, total_sqft, bath, bhk):
    loc_index = np.where(x_df.columns == location)[0][0]

    x = np.zeros(len(x_df.columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]

import pickle
with open(r'bangalore_realestate_price_prediction.pickle', 'wb') as f:
    pickle.dump(lr_clf,f)

import json
columns = {
    'data_columns' : [col.lower() for col in x_df.columns]
}
with open(r'bangalore_realestate_price_prediction.json', 'w') as f:
    f.write(json.dumps(columns))


user_loc = str(input("enter your preferred location to buy plot"))
user_sqft = int(input("enter the size of house you need "))
user_bath =int(input("the number of bathrooms needed are :"))
user_bhk = int(input("enter the number of rooms :"))
print("your predicted cost could be : ",predict_price(user_loc,user_sqft,user_bath,user_bhk))