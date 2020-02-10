import preprocess as prp
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import dump, load
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_auc_ovr
from sklearn.model_selection import GridSearchCV



#pkr_data
#TODO move into preprocess
pkr_data = prp.pkr_data()
pkr_data.clean()
pkr_data.target = 'hand'
pkr_data.features = pkr_data.all.columns[pkr_data.all.columns != pkr_data.target]

pkr_data.init_model_data(target =['hand'],features = ['suit1','card1','suit2','card2','suit3','card3','suit4','card4'])

#ab data
ab_data = prp.ab_data()
# ab_data.encode = ['room_type']
ab_data.clean()
ab_data.target = 'room_type'
ab_data.features = ab_data.all.columns[ab_data.all.columns != ab_data.target]

ab_data.init_model_data(target=ab_data.target,features=ab_data.features)
print("Models Initiated")
#validate classifier ab_data
# KNN = KNeighborsClassifier()
# KNN.fit(ab_data.x_train,ab_data.y_train)
# y_prob=KNN.predict_proba(ab_data.x_test)
# y_predict = KNN.predict(ab_data.x_test)


KNN = KNeighborsClassifier()
KNN.fit(pkr_data.x_train,pkr_data.y_train)
y_prob=KNN.predict_proba(pkr_data.x_test)
y_predict = KNN.predict(pkr_data.x_test)

#https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
# macro_roc_auc_ovo = metrics.roc_auc_score(ab_data.y_test, y_prob, multi_class="ovo",
#                                   average="macro")
# weighted_roc_auc_ovo = metrics.roc_auc_score(ab_data.y_test, y_prob, multi_class="ovo",
#                                      average="weighted")
# macro_roc_auc_ovr = metrics.roc_auc_score(ab_data.y_test, y_prob, multi_class="ovr",
#                                   average="macro")
# weighted_roc_auc_ovr = metrics.roc_auc_score(ab_data.y_test, y_prob, multi_class="ovr",
#                                      average="weighted")
# print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
#       "(weighted by prevalence)"
#       .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
# print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
#       "(weighted by prevalence)"
#       .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))

# score = metrics.roc_auc_score(ab_data.y_test,y_prob,multi_class='ovr',average='macro',max_fpr=1.0)
# acc_score = metrics.accuracy_score(ab_data.y_test,y_predict)

score = metrics.roc_auc_score(pkr_data.y_test,y_predict,multi_class='ovr',average='macro',max_fpr=1.0)
acc_score = metrics.accuracy_score(pkr_data.y_test,y_predict)

print("roc single fit: ",score)
print("accuracy score: ",acc_score)


    

# params = {'n_neighbors':[1,2,3,4,5,15,50],'weights':['uniform','distance'],'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}
params = {'n_neighbors':[1,2,3,4,5,15,50],'weights':['uniform','distance'],'algorithm':['auto']}

# params = {'n_neighbors':[1,2],'weights':['uniform','distance'],}

# roc_scorer = make_scorer(roc_auc,needs_proba=True)
def KNN_gridSearch(data_obj,params):

    #TODO add progress bar
    
    clf_grid = make_pipeline(preprocessing.StandardScaler(),
                                        GridSearchCV(KNeighborsClassifier(),
                                             param_grid=params,
                                             cv=5,
                                             refit=True,n_jobs=2,verbose=1,scoring='roc_auc_ovr'))

    # clf_grid = make_pipeline(preprocessing.StandardScaler(),
    #                                     GridSearchCV(KNeighborsClassifier(),
    #                                          param_grid=params,
    #                                          cv=5,
    #                                          refit=True,n_jobs=2,verbose=1))

    clf_grid.fit(data_obj.x_train,data_obj.y_train)
    # y_prob = clf_grid.predict_proba(data_obj.x_test)
    y_predict = KNN.predict(pkr_data.x_test)
    # marker = 1
    #
    # score = metrics.roc_auc_score(data_obj.y_test,y_prob,multi_class='ovr',average='macro',max_fpr=1.0) #for ab data
    score = metrics.roc_auc_score(pkr_data.y_test, y_predict, multi_class='ovr', average='macro', max_fpr=1.0)
    # score = clf_grid.score(data_obj.x_test, data_obj.y_test)
    print("grid search:",score)
    dump(clf_grid, 'clf_KNN_grid'+str(data_obj.title)+'final_score_rocxxxxxxx.joblib')




# KNN_gridSearch(ab_data,params)
KNN_gridSearch(pkr_data,params)


# dump(clf_oneh_grid, 'clf_oneh_grid.joblib'+datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
print("no errors")


