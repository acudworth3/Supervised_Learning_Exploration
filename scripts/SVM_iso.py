import preprocess as prp
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm

import preprocess as prp
from sklearn import metrics
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import dump, load
from sklearn.model_selection import GridSearchCV


pkr_data = prp.pkr_data()
pkr_data.clean()
pkr_data.target = 'hand'
pkr_data.features = pkr_data.all.columns[pkr_data.all.columns != pkr_data.target]

pkr_data.init_model_data(target =['hand'],features = ['suit1','card1','suit2','card2','suit3','card3','suit4','card4'])

#ab data
ab_data = prp.ab_data(n=100)
# ab_data.encode = ['room_type']
ab_data.clean()
ab_data.target = 'room_type'
ab_data.features = ab_data.all.columns[ab_data.all.columns != ab_data.target]

ab_data.init_model_data(target=ab_data.target,features=ab_data.features)
print("Models Initiated")


#validate ab
SVM_model = svm.SVC(kernel='linear',verbose=True)
SVM_model.fit(ab_data.x_train,ab_data.y_train)
ab_data.y_predict=SVM_model.predict(ab_data.x_test)
y_prob = SVM_model.predict_proba(ab_data.x_test)
ab_acc_score = metrics.accuracy_score(ab_data.y_test,ab_data.y_predict)
ab_roc_score = metrics.roc_auc_score(ab_data.y_test,y_prob,multi_class='ovr',average='macro',max_fpr=1.0) #for ab data


#validate pkr
# SVM_model = svm.SVC(kernel='linear',verbose=True)
# SVM_model.fit(pkr_data.x_train,pkr_data.y_train)
# pkr_data.y_predict=SVM_model.predict(pkr_data.x_test)
# # y_prob = SVM_model.predict_proba(pkr_data.x_test)
# pkr_acc_score = metrics.accuracy_score(pkr_data.y_test,pkr_data.y_predict)
# pkr_roc_score = metrics.roc_auc_score(pkr_data.y_test,pkr_data.y_predict,multi_class='ovr',average='macro',max_fpr=1.0) #for pkr data
#
# print("pkr roc",pkr_roc_score)
# print("pkr acc",pkr_acc_score)
marker = 1

# print(score)
#
params = {'C':[0.1,0.5,1.0], 'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'degree':[1,2,3,4],'decision_function_shape':['ovr']}
#C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr'

def SVM_gridSearch(data_obj, params):
    # TODO add progress bar

    # clf_grid = make_pipeline(preprocessing.StandardScaler(),
    #                          GridSearchCV(svm.SVC(verbose=True),
    #                                       param_grid=params,
    #                                       cv=5,
    #                                       refit=True, n_jobs=3, verbose=1))

    clf_grid = make_pipeline(preprocessing.StandardScaler(),
                             GridSearchCV(svm.SVC(verbose=True),
                                          param_grid=params,
                                          cv=5,
                                          refit=True, n_jobs=3, verbose=1,scoring='roc_auc_ovr'))
    clf_grid.fit(data_obj.x_train, data_obj.y_train)

    # y_prob = clf_grid.predict_proba(data_obj.x_test)
    # y_predict = clf_grid.predict(data_obj.x_test)

    # score = clf_grid.score(data_obj.x_test, data_obj.y_test)
    # score = metrics.roc_auc_score(data_obj.y_test,y_prob,multi_class='ovr',average='macro',max_fpr=1.0) #for ab data
    # score = metrics.roc_auc_score(data_obj.y_test, y_predict, multi_class='ovr', average='macro',max_fpr=1.0)  # for pkr data
    score = 0000
    print("grid search:", score)
    dump(clf_grid, 'clf_SVM_grid' + str(data_obj.title)+'_final_' + str(round(score, 4)) + '.joblib')
    # dump(clf_grid, 'clf_SVM_grid' + str(data_obj.title) + '_final_roc_' + str(round(score, 4)) + '.joblib')


# SVM_gridSearch(pkr_data, params)
# SVM_gridSearch(ab_data,params)


# dump(clf_oneh_grid, 'clf_oneh_grid.joblib'+datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
print("no errors")
