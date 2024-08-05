#!/usr/bin/env python3

import pickle
import pandas as pd
import numpy as np
import sklearn 
from sklearn import linear_model

if __name__ == "__main__":
    data = pd.read_csv("phone_data.csv")

    predict = "price_range"
    #List for all the features corresponding with their integer values --> Data did not need to be encoded 
    classes = ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"]
    #Dropping the target/label column from the data so just the feauture/input columns remain 
    features = np.array(data.drop([predict], axis=1))
    #Getting the label column only 
    target = np.array(data[predict])

    #Seperating/Splitting the phone_data so that 10% is used for test the model and the remaining 90% is used to train the model
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(features, target, test_size=0.1)
    
    best = 0
    #This loop is training the model training, trying to get the best accuracy, in this case 91,922% accuracy was achieved 
    #This model is then saved in a pickle
    # for x in range(30):
    #     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(features, target, test_size=0.1)
    #     linear = linear_model.LinearRegression()
    #     linear.fit(x_train, y_train)
    
    #     accurracy = linear.score(x_train, y_train)
    #     if accurracy > best:
    #         with open("phonemodel.pickle", "wb") as mypickle:
    #             pickle.dump(linear, mypickle)
    
    #Getting the model from the file  
    mypickle = open("phonemodel.pickle", "rb")
    linear = pickle.load(mypickle)
    print("--> Final Accurracy: ", linear.score(x_train, y_train))
    #Testing the model on data it has not seen 
    prediction = linear.predict(x_test)
    #Results
    for x in range(len(prediction)):
        print("Prediction: ", classes[int(prediction[x])], "\nActual Price: ", classes[y_test[x]])
        print("-" * 40)
    