import pandas
import matplotlib.pyplot as plt
import numpy as np

''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path1.
'''
dataset_2 = pandas.read_csv('NYC_Bicycle_Counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
#print(dataset_2.to_string()) #This line will print out your data

# converting the columns to NumPy arrays
brooklyn = dataset_2['Brooklyn Bridge'].to_numpy()
manhattan = dataset_2['Manhattan Bridge'].to_numpy()
queensboro = dataset_2['Queensboro Bridge'].to_numpy()
williamsburg = dataset_2['Williamsburg Bridge'].to_numpy()

bSum = np.sum(brooklyn) # number of bikers
mSum = np.sum(manhattan) # number of bikers
qSum = np.sum(queensboro) # number of bikers
wSum = np.sum(williamsburg) # number of bikers

sensorDict = {"Brooklyn Bridge": bSum, "Manhattan Bridge": mSum, "Queensboro Bridge": qSum, "Williamsburg Bridge": wSum}
keys = list(sensorDict.keys())
values = list(sensorDict.values())
sorted_index = np.argsort(values)
print(sorted_index)
sorted_dict = {keys[i]: values[i] for i in sorted_index}


print("The sensors should be installed on:")
for i in range(3, 0, -1):
    print(f'- the {keys[sorted_index[i]]} ({values[sorted_index[i]]} bikers)')