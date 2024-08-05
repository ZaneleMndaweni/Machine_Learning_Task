Firstly, I got the data from Kaggle  Mobile Price Classification (kaggle.com)
My code aims to train a linear regression model to predict phone price ranges from low cost to a very high cost based on the various features/inputs (dua_sim, int_memory, etc.).
Then ultimately test how well the model works.
I did this by first reading the data from the csv file into my program using the pandas library. 
I then separated the data into separate arrays for the various features columns and the label (Price range) column using the numpy library.
I then split the data into training and testing sets. 
90% was used to train the model and the remaining 10% was used to test how well the model worked. 
To find the model with the best accuracy I had a loop that ran 30 times, training the model. 
The model with the best accuracy was then saved into a binary file. 
I then commented this piece of code once the best model was found.
The model with the best accuracy was then loaded in from the binary file and used to predict the price ranges using the test data (data, the model had never seen). 
I then printed the actual prices along with the models’ predictions. The current model has an accuracy of 91.922%.
