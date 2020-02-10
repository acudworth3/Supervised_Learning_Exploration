import preprocess as prp
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm

#get data
ab_obj = prp.ab_data(n=5000)
#clean data select target/features
ab_obj.clean() #TODO decide if this should be done automatically
# ab_obj.target = ['price']
# ab_obj.features = ['host_id','price']
ab_obj.init_model_data()
X = ab_obj.X
Y = ab_obj.Y

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.1,random_state=105) #TODO move to helper function


SVM_model = svm.SVC(kernel='linear')
SVM_model.fit(x_train,y_train)
y_predict=SVM_model.predict(x_test)
score = metrics.accuracy_score(y_test,y_predict)
print(score)