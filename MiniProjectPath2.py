import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix



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
dataset_2['Total']  = pandas.to_numeric(dataset_2['Total'].replace(',','', regex=True))
dataset_2['High Temp']  = pandas.to_numeric(dataset_2['High Temp'].replace(',','', regex=True))
dataset_2['Low Temp']  = pandas.to_numeric(dataset_2['Low Temp'].replace(',','', regex=True))
dataset_2['Precipitation']  = pandas.to_numeric(dataset_2['Precipitation'].replace(',','', regex=True))
#print(dataset_2.to_string()) #This line will print out your data

# Part 2
# Function that normalizes testing set according to mean and std of training set
#
# Input
# --------------------
# X_test : Testing data, a numpy array with shape (n_samples, n_features)
# trn_mean : Mean of each column in training set, float
# trn_std : Standard deviation of each column in training set, float
#
# Output
# --------------------
# X : The normalized version of the feature matrix, X_test
def normalize_test(X_test, trn_mean, trn_std):

    '''
    Fill in your code here
    '''
    
    X = (X_test - trn_mean) / trn_std
    
    #CHECKED

    return X


# Function that normalizes features in training set to zero mean and unit variance.
#
# Input
# --------------------
# X_train : Training data, a numpy array with shape (n_samples, n_features),
#         where n_samples is the number of samples in the training data, and 
#         n_features is the number of features in the data
#
# Outut
# --------------------
# X : The normalized version of the feature matrix, a numpy array
# trn_mean : The mean of each column in the training set, float
# trn_std : The std dev of each column in the training set, float
def normalize_train(X_train):
    if np.any(X_train) == False:
        return(np.array([]), np.array([]), np.array([]))
    trn_mean = []
    trn_std = []
  
    trn_mean = np.mean(X_train, axis=0)
    trn_std = np.std(X_train, axis=0)
    X = (X_train - trn_mean) / trn_std
    
    #CHECKED
    return X, trn_mean, trn_std


# converting the columns to NumPy arrays
brooklyn = dataset_2['Brooklyn Bridge'].to_numpy()
manhattan = dataset_2['Manhattan Bridge'].to_numpy()
queensboro = dataset_2['Queensboro Bridge'].to_numpy()
williamsburg = dataset_2['Williamsburg Bridge'].to_numpy()
total = dataset_2['Total'].to_numpy()
high = dataset_2['High Temp'].to_numpy()
low = dataset_2['Low Temp'].to_numpy()
precip = dataset_2['Precipitation'].to_numpy()



# Problem 1: Sensors

bSum = np.sum(brooklyn) # number of bikers
mSum = np.sum(manhattan) # number of bikers
qSum = np.sum(queensboro) # number of bikers
wSum = np.sum(williamsburg) # number of bikers

sensorDict = {"Brooklyn Bridge": bSum, "Manhattan Bridge": mSum, "Queensboro Bridge": qSum, "Williamsburg Bridge": wSum}
keys = list(sensorDict.keys())
values = list(sensorDict.values())
sorted_index = np.argsort(values)
sorted_dict = {keys[i]: values[i] for i in sorted_index}


print("The sensors should be installed on:")
for i in range(3, 0, -1):
    print(f'- the {keys[sorted_index[i]]} ({values[sorted_index[i]]} bikers)')
    
    


# Problem 2: Weather
length = len(brooklyn) # how many data points we have for each column
trLength = round(length * 0.8) # length of training data (80/20 split)

tstLength = length - trLength # length of testing data


trHigh = high[0:trLength] # training data for high temp
tstHigh = high[trLength::] # testing for high temp

trLow = low[0:trLength] # training data for low temp
tstLow = low[trLength::] # testing data for low temp

trPrecip = precip[0:trLength] # training data for precipitation
tstPrecip = precip[trLength::] # testing data for precipitation

trTotal = total[0:trLength] # training data for total bikers
tstTotal = total[trLength::] # testing data for total bikers




# building training feature matrix
trainX = np.column_stack((trHigh, trLow, trPrecip))

normX, meanX, stdX = normalize_train(trainX)

# linear regression
model_lin = LinearRegression(fit_intercept = True)
model_lin.fit(normX, trTotal) # training model
coef = model_lin.coef_
intercept = model_lin.intercept_




# building testing feature matrix
testX = np.column_stack((tstHigh, tstLow, tstPrecip))

normTest = normalize_test(testX, meanX, stdX)

predicted_Bikers = model_lin.predict(normTest)

r2 = round(r2_score(tstTotal, predicted_Bikers), 3)

print(f'The r^2 score is: {r2}')

# Plot outputs
x = range(0, tstLength)
plt.scatter(x, tstTotal, color='black', label='test data')
plt.scatter(x, predicted_Bikers, marker='x', color='red', label='predicted data')
plt.xlabel('Days Since September 18')
plt.ylabel('Number of Bikers')
plt.title(f"Does Weather Influence the Number of Bikers on Bridges in NYC?\nr^2 = {r2}")
plt.legend()
plt.show()



# Problem 3
k = 5

arr = np.mod(np.arange(williamsburg.size) + k, 7)

Total = dataset_2['Total'].to_numpy()

new = arr.reshape(-1, 1)

newTot = Total.reshape(-1, 1)

new.ravel()

newTot.ravel()



X_train, X_test, y_train, y_test = train_test_split(new, newTot, test_size=0.2)



model = MLPClassifier()

X_train = np.ravel(X_train)

y_train = np.ravel(y_train)

#print(X_train)

X_train = X_train.reshape(-1, 1)

y_train = y_train.reshape(-1, 1)



model.fit(X_train, y_train)

# ^ causing warnings

predictions = model.predict(X_test)



mse = mean_squared_error(y_test, predictions)




print("Mean Squared Error for Part 3:", mse)

x = range(0, len(X_test))
plt.scatter(x, y_test, color='black', label='test data')
plt.scatter(x, predictions, marker='x', color='blue', label='predicted data')
plt.xlabel('Days Since September 18')
plt.ylabel('Number of Bikers')
plt.title(f"Does Weather Influence the Number of Bikers on Bridges in NYC?\nMSE = {round(mse, 3)}")
plt.legend()
plt.show()



# now add up days of week from 0-6 on all bridges for each day.

#Sunday = 0

#Sat - 6






