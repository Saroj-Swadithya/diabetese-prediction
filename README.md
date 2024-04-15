# Diabetes Prediction
Hey there! This is a Python machine learning project that deals with a diabetes prediction.

### Methodology
1. The diabetes dataset is loaded using the Pandas library. 
2. Data preprocessing is performed to handle missing values by replacing zeros with NaN using the copy method.
3. The missing values are imputed with either the mean or median of the respective feature using the fillna method with inplace=True.
4. Standard scaling is applied to normalize the features using the StandardScaler from sklearn.preprocessing.
5. The data is split into training and testing sets using the train_test_split function from sklearn.model_selection. The Random Forest classifier is chosen as the model for prediction, and an instance of RandomForestClassifier is created with a specified number of estimators.
6. The model is then fitted or trained on the training data using the fit method.
7. The accuracy of the model is evaluated on the testing data using the accuracy_score function from sklearn.metrics. The feature importance of the Random Forest model is obtained using the feature_importances_ attribute, which indicates the weightage of each feature in the model building phase.
8. The trained model is then saved using pickle.dump function to serialize it into a binary file.
9. The next step is to create a machine learning model using the RandomForestClassifier from sklearn.ensemble module. 
10. Then fit the model on the training data and makes predictions on the testing data. The accuracy of the model is evaluated using the accuracy_score and classification_report functions.
11. To predict the outcome for new input, the saved model is loaded using pickle.load function, and the predict method is used to make predictions. Additionally, the script allows the user to input a number, which is used as an index to retrieve the corresponding data from the original dataset. The predicted outcome is then printed based on the user input or the new input.

### Things to Note
1. The code has been currently tested only with available data set from PIMS.
2. Its not a large data set.
