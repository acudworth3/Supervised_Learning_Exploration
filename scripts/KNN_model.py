import preprocess as prp
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve
import pandas as pd
import numpy as np
import plot_learning_curve as plcv
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV

#get data
ab_obj = prp.ab_data()
#clean data select target/features
ab_obj.clean() #TODO decide if this should be done automatically
# ab_obj.target = ['calculated_host_listings_count']
# ab_obj.features = ['host_id','price']


ab_obj.init_model_data()
X = ab_obj.X
Y = ab_obj.Y

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.1,random_state=105) #TODO move to helper function


#create model
KNN = KNeighborsClassifier()
KNN.fit(x_train,np.ravel(y_train))
y_predict=KNN.predict(x_test)
score = metrics.accuracy_score(y_test,y_predict)
print("single fit: ",score)

#single learning curve
# # train_sizes, train_scores, valid_scores = learning_curve(KNeighborsClassifier(n_neighbors=3), X, np.ravel(Y), train_sizes=np.linspace(int(np.array(X).shape[0]*0.9),np.array(X).shape[0],5,dtype=int), cv=5)
# train_sizes, train_scores, valid_scores = learning_curve(KNeighborsClassifier(n_neighbors=3), X, np.ravel(Y), train_sizes=np.linspace(int(np.array(X).shape[0]*0.1),np.array(X).shape[0]*0.79,5,dtype=int), cv=5)
#
# print(train_sizes)

#grid search
# param_dict = KNN.get_params()
#

parameters = {'n_neighbors':[1,2,3,4], 'leaf_size':[30,35,40]}
# clf = GridSearchCV(KNN, parameters,refit=True)
# clf.fit(x_train, y_train)
# results = pd.DataFrame(clf.cv_results_)
#clf.best_estimator_ gives best estimator
#note onehot encoder slows things down alot

clf = make_pipeline(preprocessing.Normalizer(),
                    GridSearchCV(KNeighborsClassifier(),
                                 param_grid=parameters,
                                 cv=2,
                                 refit=True,n_jobs=1))

print("grid complete:")

clf.fit(x_train,y_train)
print("pipeline complete:",clf.score(x_test,y_test))


# #plot learning curve
# chart_obj = plcv.plot_learning_curve(clf.best_estimator_, 'test', X, Y, axes=None, ylim=None, cv=None,
#                         n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5))
# chart_obj.show()