from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns

from pre_processing import *

lm = LinearRegression()


"""
Setting X and y training variables for linear regression model
"""

X = train_set.drop(['SalePrice','Id'],axis = 1)
test_set.drop(['Id'],axis=1,inplace =True)
y = np.log(train_set['SalePrice'])

"""
Splitting the training data set into training and test data set to get an idea
of the accuracy of our model
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

"""
Training the linear regression model
"""
lm.fit(X_train,y_train)
#print(lm)
"""
Predicting the survived column for test set
"""
predictions = lm.predict(X_test)
#print(predictions)

"""
Calculating accuracy of the model 
"""
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
score = (lm.score(X_test,y_test))
print("Model score is %g" %score)
#print(X_train)
sns.scatterplot(predictions,y_test,alpha=.75)

#print('\n',test_set.info())
#print('\n',X.info())
#print(X.columns & test_set.columns)

"""
Using the entire train.csv file as the training dataset to predict survived column in the test.csv file
"""
X_test_final = test_set
lm_final = LinearRegression()
lm_final.fit(X,y)   #Training the model with the completetraining set
predictions_final = lm_final.predict(X_test_final)

predictions_final = np.exp(predictions_final)
#print(predictions_final)
X_test_final['SalePrice'] = predictions_final
X_test_final.to_csv('Results.csv',index=False)


plt.show()
