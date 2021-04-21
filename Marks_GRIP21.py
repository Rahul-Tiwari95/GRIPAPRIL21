# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 11:40:17 2021

@author: rahul
"""

# Supervised Machine Learning TASK 2
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

scores=pd.read_csv("E:\\Data Science\\Data Sheet\\hours_scores.csv")

plt.hist(scores.hr)
plt.boxplot(scores.hr,0,"rs",0)

plt.hist(scores.skr)
plt.boxplot(scores.skr)

plt.plot(scores.hr,scores.skr,"bo");plt.xlabel("Hours Studied");plt.ylabel("Scores Secured")



scores.hr.corr(scores.skr) # # correlation value between X and Y
np.corrcoef(scores.hr,scores.skr)


#splitting dataset into train and test datasets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(scores[["hr"]],scores.skr,test_size=0.3,random_state=42)

from sklearn.linear_model import LinearRegression
#initializing the model
linreg= LinearRegression()
#fitting training data to the model
linreg.fit(X_train,Y_train)
linreg.intercept_
linreg.coef_

pickle.dump(linreg,open('model.pkl','wb'))

#prediction
y_pred=linreg.predict(X_test)

#for 9.25 hours

linreg.predict([[9.25]])


#creating Data Frame with actual and predicted 
test_pred_df=pd.DataFrame({'actual':Y_test,'predict':np.round(y_pred,2)})
test_pred_df.plot(kind="bar",color=['b','g'])



from sklearn import metrics

mse=metrics.mean_squared_error(Y_test,y_pred)
rmse=round(np.sqrt(mse),2)
print("RMSE: ",rmse)

# Visualization of regresion line over the scatter plot of Scores and Hours
# For visualization we need to import matplotlib.pyplot
sns.regplot(x=scores['hr'],y=scores['skr'],data=scores)
plt.title('Relationship between Hours to study and Scores obtained')
plt.xlabel('Hours')
plt.ylabel('Scores')

