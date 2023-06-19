

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

data = pd.read_csv('Real estate.csv')

data.head()

data.info()

data.describe()

data.isnull().sum().sort_values(ascending = False)

data =data.drop(['No'],axis = 1)

plt.figure(figsize =(10,10))
sns.heatmap(data.corr() , annot = True , cbar = True , fmt = '.1f' ,cmap = 'Greens',annot_kws ={'size':8})
data.rename(columns = {'X1 transaction date' :'transaction_date', 'X2 house age':'houseage',
       'X3 distance to the nearest MRT station':'distance_to_the_nearest_MRT_station',
       'X4 number of convenience stores':'number_of_convenience_stores', 'X5 latitude':'latitude', 'X6 longitude':'longitude',
       'Y house price of unit area':'house_price_of_unit_area'} ,inplace = True)

x = data.drop(['house_price_of_unit_area'] , axis = 1)
y = data['house_price_of_unit_area']

x.head()

from sklearn.model_selection import train_test_split

#from sklearn.linear_model import SGDClassifier

from sklearn import metrics

x_train ,x_test ,y_train , y_test = train_test_split(x , y ,test_size = 0.2 ,random_state = 2)

print(x.shape , x_train.shape , x_test.shape)

from sklearn import svm
model = svm.SVR()

model.fit(x_train , y_train)

train_prediction = model.predict(x_train)

print(train_prediction)

#R squared Error
score_1 = metrics.r2_score(y_train , train_prediction)
# mean Absolute Error
score_2  = metrics.mean_absolute_error(y_train ,train_prediction)


print('R square Error :' , score_1)
print('mean square error :' , score_2)

plt.scatter(y_train , train_prediction)
plt.xlabel('Actual Prices')
plt.ylabel('predicted prices')
plt.title('Actual vs predicted prices')
plt.show()

test_prediction = model.predict(x_test)

#R squared Error
score_11 = metrics.r2_score(y_test , test_prediction)
# mean Absolute Error
score_22  = metrics.mean_absolute_error(y_test ,test_prediction)


print('R square Error :' , score_11)
print('mean square error :' , score_22)

model.predict([[2012.917,32.0	,84.87882	,10	,24.98298	,121.54024]])

pickle.dump(model , open('model.pkl', 'wb'))