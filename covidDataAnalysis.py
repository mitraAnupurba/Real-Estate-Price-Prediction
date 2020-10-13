import pandas as pd
df = pd.read_csv('F:\personal_information\covid_data\covid_19_india.csv')

#to print only the columns
print("the columns are :")
print(df.columns)

#to print only the information under the cured column
print("the information under cured column is ")
print(df.Cured)

#to print information under these specific columns, not all
print("data under three specific columns, namely serial number, state and date are : specific columns are :")
print(df[['Sno','Date','State/UnionTerritory']])

#statistical operations on data:
print("the maximum number of deaths are :",df['Deaths'].max())
print("the maximum number of deaths are :",df['Deaths'].min())

#All the statistical information inside a dataset : count, mean, standard deviation, etc:
print("the statistical values are :",df.describe())

#conditionally printng a row:
print("The maximum number of dates are :",df[df.Deaths == df['Deaths'].max()])

#conditionally printing a row and a column:
print("the row and column of the maximum number of deaths are :")
print(df[['Sno','State/UnionTerritory','Deaths']][df.Deaths == df['Deaths'].max()])

#printing the index value of the data set
print(df.index)

#detting a different attrivbute as an index value temporarily
df.set_index('Sno')

#detting a different attrivbute as an index value permanently
df.set_index('Sno', inplace=True)
print("the modified data frame is :")
print(df)


#location of the row with the index value 4000
print("the information in row number 4000 is :",df.loc[4000])

#reset the index to the pervious value
df.reset_index(inplace=True)
df.set_index('Sno',inplace=True)
