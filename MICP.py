import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# loading dataset
insurance_dataset = pd.read_csv('./Dataset/insurance.csv')

# shape of ds
print(insurance_dataset.head())

# information about the dataset
# print(insurance_dataset.info())

# categorical features
# sex 
# smoker
# region

# chking for missing values
# print(insurance_dataset.isnull().sum())


# Data Analysis
# statistical Measures of the data
# print(insurance_dataset.describe())

# distribution of age value
# sns.set_theme()
# plt.figure(figsize=(6,6))
# sns.displot(insurance_dataset['age'])
# plt.title('Age Distribution')
# plt.show()

# Gender column
# sexDis = insurance_dataset['sex'].value_counts()
# print(sexDis)

# BMI Distribution
# Normal BMI for a person is 18.5 to 24.9 
# if less than 18.5 is less weight and over than 24.9 is over-weighted
# sns.set_theme()
# plt.figure(figsize=(6,6))
# sns.displot(insurance_dataset['bmi'])
# plt.title('BMI Distribution')
# plt.show()


# Children slot
# plt.figure(figsize=(6,6))
# sns.countplot(x='children', data=insurance_dataset)
# plt.title('Children')
# plt.show()
# childDis = insurance_dataset['children'].value_counts()
# print(childDis)

# smoker
# plt.figure(figsize=(6,6))
# sns.countplot(x='smoker', data=insurance_dataset)
# plt.title('Smoker')
# plt.show()
# smokerDis = insurance_dataset['smoker'].value_counts()
# print(smokerDis)

# region column
# plt.figure(figsize=(6,6))
# sns.countplot(x='region', data=insurance_dataset)
# plt.title('Region')
# plt.show()
# RegDis = insurance_dataset['region'].value_counts()
# print(RegDis)

# Charges
# plt.figure(figsize=(6,6))
# sns.displot(insurance_dataset['expenses'])
# plt.title('Expense Charges')
# plt.show()
# ExpenseDis = insurance_dataset['expenses'].value_counts()
# print(ExpenseDis)

# Data Pre-Processing
# Encoding the categorical features

# encoding sex column
# insurance_dataset.replace({'sex':{
#     'male':0, 
#     'female':1
# }}, inplace=True)

# encoding smoker
# insurance_dataset.replace({'smoker':{
#     'yes':0,
#     'no':1
# }}, inplace=True)

# encoding region
# insurance_dataset.replace({'region':{
#     'southeast':0,
#     'southwest':1,
#     'northeast':2,
#     'northwest':3
# }}, inplace=True)

# print(insurance_dataset.head())

# Spliting the Features nd Target
# X = insurance_dataset.drop(columns='expenses', axis=1)
# Y = insurance_dataset['expenses']

# print(Y)

# Splting the data into training and testing data
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
# print(X.shape, X_train.shape, X_test.shape)

# Model Training
# Loding the Linear Regression Model
# regressor = LinearRegression()
# regressor.fit(X_train, Y_train)


# Model Evaluations
# Prediction on training data
# training_data_prediction = regressor.predict(X_train)

# R square value
# r2_train = metrics.r2_score(Y_train, training_data_prediction)
# print('Training accuracy: ',r2_train)

# Prediction on testing data
# testing_data_prediction = regressor.predict(X_test)
# r2_test = metrics.r2_score(Y_test, testing_data_prediction)
# print('Testing accuracy: ',r2_test)

# Building Predictive System
# input_data = (31, 1, 25.74, 0, 1, 0)

# changing input_data into numpy array
# input_data_as_numpy_array = np.asarray(input_data)
# print(input_data_as_numpy_array)

# input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
# print(input_data_reshaped)

# predict = regressor.predict(input_data_reshaped)
# print('The insurance cost is USD: ',predict[0])