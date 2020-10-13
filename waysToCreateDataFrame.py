import pandas as pd

# creating data frames using a excel file:
# we need to install the xlrd package to get excel support
# command to install : $conda install -c anaconda xlrd or $pip install xlrd

#

#from CSV file :
df = pd.read_csv('F:\personal_information\Online Teaching\Psatticus_onlineTeaching\Content_prep_offer\dataexport_20200902T055215.csv')
print("The data frame is : ")
print(df)

#from excel file
df = pd.read_excel('F:\personal_information\covid_data\owid-covid-data.xlsx')
print("The data frame is : ")
print(df)

# creating data frames using a tuple set, each tuple is a row
weather_data = [('1/1/2017',32,6,'rainy'),
                ('1/2/2017',25,8,'snowy'),
                ('1/3/2017',34,5,'sunny')]
df = pd.DataFrame(weather_data,columns=['date','tempr','wind speed','event'])
print(df)

## creating data frames using a list of dictioneries :
weather_data = [
    {'day':'1/1/2020' ,'temperature' : 36},
    {'day':'1/2/2020' ,'temperature' : 25},
    {'day':'1/3/2020' ,'temperature' : 30}]

df = pd.DataFrame(weather_data)
print(df)