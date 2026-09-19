import pandas as pd
from matplotlib import pyplot as  plt
data = {
    'hours_studied':[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,6.0],
    'passed':[0,0,0,0,1,1,1,1,1,1]
}
df = pd.DataFrame(data)

print(df.head())

plt.scatter(df.hours_studied,df.passed,marker='+',color='red')

plt.xlabel('hour studied')
plt.ylabel('passed')
plt.show()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(
    df[['hours_studied']],
    df.passed,
    train_size=0.8,
    random_state=42
)

print(x_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(x_train,y_train)

y_predicted = model.predict(x_test)

print('predict values')
print(y_predicted)

print('prbabilities:')
print(model.predict_proba(x_test))

print('model accuracy:')
print(model.score(x_test,y_test))

print('coefficient:')
print(model.coef_)

print('intercept')
print(model.intercept_)

import math

def sigmoid(x):
    return 1/(1+math.exp(-x))

def prediction_function(hours):
     z = model.coef_[0][0]*hours +model.intercept_[0]

     y = sigmoid(z)

     return y
hours = 2
print(prediction_function(hours))


