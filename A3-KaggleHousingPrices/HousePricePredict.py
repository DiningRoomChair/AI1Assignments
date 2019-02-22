import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

# get and view training and test data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print("Train - ", train.shape)
print("Test - ", test.shape)
print(train.head())
# view plotted data for training ideas
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
# use np.log() to normalize the data for the right skew
target = np.log(train.SalePrice)
# get the columns with numbers and check their types
numeric_columns = train.select_dtypes(include=[np.number])
print(numeric_columns.dtypes)
# remove the outliers of largest magnitude
train = train[train['GarageArea'] < 1150]
# check how many nulls are in each column
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
# consider the non-numeric features and display details of columns
categoricals = train.select_dtypes(exclude=[np.number])
# use pd.get_dummies() to create a new column called enc_street
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(test.Street, drop_first=True)
 # Pave and Grvl values converted into 1 and 0
train.enc_street.value_counts()
# make SaleCondition as a new feature
def encode(x): return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
y = np.log(train.SalePrice)
# exclude ID from features and run the regression
X = data.drop(['SalePrice', 'Id'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
# make the predictions
predictions = model.predict(X_test)
final_submission = pd.DataFrame()
final_submission['Id'] = test.Id
# select the features from the test data for the model as we did above.
edits = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()
# reexponentiate the logged data
pred = np.exp(model.predict(edits))
# set predictions and export to kaggle
final_submission['SalePrice'] = pred
final_submission.to_csv('submission.csv', index=False)
