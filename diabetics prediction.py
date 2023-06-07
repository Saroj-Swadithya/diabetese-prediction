import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# reading the dataset
diabetes_df = pd.read_csv('C:\\Users\\saroj\\OneDrive\\Documents\\diabetes.csv')

# counting the zeros with the NAN values
diabetes_df_copy = diabetes_df.copy(deep=True)
diabetes_df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_df_copy[
    ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

# inplace=True: data will be modified without returning a copy of the data or the original data
diabetes_df_copy['Glucose'].fillna(diabetes_df_copy['Glucose'].mean(), inplace=True)
diabetes_df_copy['BloodPressure'].fillna(diabetes_df_copy['BloodPressure'].mean(), inplace=True)
diabetes_df_copy['SkinThickness'].fillna(diabetes_df_copy['SkinThickness'].median(), inplace=True)
diabetes_df_copy['Insulin'].fillna(diabetes_df_copy['Insulin'].median(), inplace=True)
diabetes_df_copy['BMI'].fillna(diabetes_df_copy['BMI'].median(), inplace=True)

# standard scaling, help our ML model to give a better result
sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(diabetes_df_copy.drop(["Outcome"], axis=1), ),
                 columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                          'DiabetesPedigreeFunction', 'Age'])

y = diabetes_df_copy.Outcome

# Model Building7yyyyyyy: Splitting the dataset
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']
# split the data into training and testing data using the train_test_split function
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

# Building the model using RandomForest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)

# check the accuracy of the model on the training dataset
rfc_train = rfc.predict(X_train)
from sklearn import metrics

print("Accuracy_Score of training dataset =", format(metrics.accuracy_score(y_train, rfc_train)))

# Getting the accuracy score for Random Forest
from sklearn import metrics

predictions = rfc.predict(X_test)
print("Accuracy_Score of rfc test data=", format(metrics.accuracy_score(y_test, predictions)))

# shows that how much weightage each feature provides in the model building phase
rfc.feature_importances_

# Saving Model – Random Forest
import pickle

# Using the dump() function to save the model using pickle
saved_model = pickle.dumps(rfc)
# loading that saved model
rfc_from_pickle = pickle.loads(saved_model)
# After loading that model, use this to make predictions
rfc_from_pickle.predict(X_test)

# Predicting outcome by values
if rfc.predict([[1, 148, 72, 36, 0, 33.6, 0.627, 40]]) == 1:
    print("\nperson is diabetic")
else:
    print("\nperson is not diabetic")

# Output from the dataset
a = int(input("\nEnter a number: "))
if y[a + 1] == 1:
    print("person is diabetic")
else:
    print("person is not diabetic")
