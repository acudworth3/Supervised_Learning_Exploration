import preprocess as prp
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score  #probably don't need this
from sklearn.ensemble import AdaBoostRegressor


#TODO get some kind of model that makes predictions
#write helper functions for error ROC etc.
#write helper for cross validation
#identify how to tune model

#get data
ab_obj = prp.ab_data()
#clean data select target/features
ab_obj.clean() #TODO decide if this should be done automatically
# ab_obj.target = ['price']
# ab_obj.features = ['host_id','price']
ab_obj.init_model_data()
X = ab_obj.X
Y = ab_obj.Y



#TODO rewrite this
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.1,random_state=105) #TODO move to helper function

DTree=DecisionTreeRegressor(min_samples_leaf=.0001)
DTree.fit(x_train,y_train)
y_predict=DTree.predict(x_test)

r2_score(y_test,y_predict)
print(r2_score(y_test,y_predict))


# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=4)
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=105)
regr_1.fit(x_train,y_train)
regr_2.fit(x_train,y_train)
y_1 = regr_1.predict(x_test)
y_2 = regr_2.predict(x_test)
print(r2_score(y_test,y_2))
