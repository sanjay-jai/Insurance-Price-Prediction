import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def Cross_Entropy(age, sex, bmi, child, smoker, region):
    # loading dataset
    insurance_dataset = pd.read_csv('./Dataset/insurance.csv')

    # Encoding categorical variables
    insurance_dataset.replace({'sex': {'male': 0, 'female': 1},
                            'smoker': {'yes': 0, 'no': 1},
                            'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

    # Splitting the Features and Target
    X = insurance_dataset.drop(columns='expenses', axis=1)
    Y = insurance_dataset['expenses']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Adding a column of ones to account for the bias term
    X_scaled = np.column_stack((np.ones(len(X_scaled)), X_scaled))

    # Splitting the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=2)

    # Implementing Gradient Descent for Linear Regression
    def gradient_descent(X, Y, theta, alpha, num_iterations):
        m = len(Y)
        cost_history = []
        for i in range(num_iterations):
            # Hypothesis
            h = X.dot(theta)
            # Error
            error = h - Y
            # Gradient calculation
            gradient = X.T.dot(error) / m
            # Updating parameters
            theta -= alpha * gradient
            # Cost calculation
            cost = np.sum((error ** 2)) / (2 * m)
            cost_history.append(cost)
        return theta, cost_history

    # Initializing parameters and hyperparameters
    theta = np.zeros(X_train.shape[1])
    alpha = 0.01
    num_iterations = 1000

    # Training the model using Gradient Descent
    theta, cost_history = gradient_descent(X_train, Y_train, theta, alpha, num_iterations)

    # Plotting the cost function over iterations
    plt.plot(range(1, num_iterations + 1), cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function Over Iterations')
    plt.show()

    # Making predictions on the testing data
    def predict(X, theta):
        return X.dot(theta)

    Y_pred_train = predict(X_train, theta)
    Y_pred_test = predict(X_test, theta)

    # Evaluating the model
    def r2_score(Y_true, Y_pred):
        ssr = np.sum((Y_pred - Y_true) ** 2)
        sst = np.sum((Y_true - np.mean(Y_true)) ** 2)
        return 1 - (ssr / sst)

    print('Training accuracy (R2 score):', r2_score(Y_train, Y_pred_train))
    print('Testing accuracy (R2 score):', r2_score(Y_test, Y_pred_test))

    # Building Predictive System
    input_datas = tuple([age, sex, bmi, child, smoker, region])

    # Feature scaling for input data
    input_data_scaled = scaler.transform(np.array(input_datas).reshape(1, -1))
    # Adding bias term
    input_data_scaled = np.append(1, input_data_scaled)

    # Making prediction
    predicted_cost = predict(input_data_scaled, theta)
    # print('The insurance cost is USD:', predicted_cost)
    return predicted_cost