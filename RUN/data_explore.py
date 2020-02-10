import preprocess as prp
from sklearn.dummy import DummyClassifier
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


#read data
#dummy learners
    #plot
    #adjust scoring (can't do it)
#apply scalars

def dummy_plot(data_obj):
    dummy_vals = {'stratified':0,'uniform':0,'most_frequent':0,'prior':0}
    for stratg in dummy_vals.keys():
        dclf = DummyClassifier(strategy=stratg,random_state=data_obj.rand_seed)
        dclf.fit(data_obj.x_train,data_obj.y_train)
        dummy_vals[stratg] = dclf.score(data_obj.x_test,data_obj.y_test)

    #https://stackoverflow.com/questions/16010869/plot-a-bar-using-matplotlib-using-a-dictionary
    plt.bar(range(len(dummy_vals)), list(dummy_vals.values()), align='center')
    plt.xticks(range(len(dummy_vals)), list(dummy_vals.keys()))
    plt.title(data_obj.title+' dummy eval '+datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    return plt, dummy_vals

# ab_data.init_model_data()

#poker data
pkr_data = prp.pkr_data()
pkr_data.clean()
cols = pkr_data.head.columns
pkr_data.init_model_data(target =['hand'],features = ['suit1','card1','suit2','card2','suit3','card3','suit4','card4','suit5','card5'])

#ab data
ab_data = prp.ab_data()
ab_data.drop_cols = ['id', 'name','host_id','host_name','last_review','reviews_per_month','latitude','longitude', 'calculated_host_listings_count','neighbourhood','neighbourhood_group']
ab_data.clean()
ab_data.target = 'room_type'
ab_data.features = ab_data.all.columns[ab_data.all.columns != ab_data.target]

ab_data.init_model_data(target=ab_data.target,features=ab_data.features)



#preprocessing
#pkr
#normalize the numbers
#Normalize the target
#Onehot encode the suite
# cat_enc = preprocessing.OneHotEncoder(categories=['suit1'])

#Categorizing

# def cat_enc_columns(data_obj,cols = []):
#     cat_enc = preprocessing.LabelBinarizer()
#     for col in cols:
#         col_vals = data_obj.all[col]
#         data_obj.all = pd.concat([data_obj.all, pd.DataFrame(cat_enc.fit_transform(col_vals))], axis=1)
#         data_obj.all.drop([col],axis=1)
#         marker = 1
#
# cat_enc_columns(ab_data,cols=['neighbourhood','neighbourhood_group'])
# data = cat_enc.fit_transform(pkr_data)
#alternate is pd.getdummies['suit1']
# col_vals = pkr_data.all['suit1']
# pd.concat([pkr_data.all,pd.DataFrame(cat_enc.fit_transform(col_vals))],axis=1)
# # pd.concat([pkr_data.all,pd.DataFrame(cat_enc.fit_transform(pkr_data.all.suit1.values))],axis=1)
#
#
# #scaling
# scal_enc = preprocessing.Normalizer().fit(pkr_data.all)



KNN = KNeighborsClassifier()
KNN.fit(ab_data.x_train,np.ravel(ab_data.y_train))
y_predict=KNN.predict(ab_data.x_test)
score = metrics.accuracy_score(ab_data.y_test,y_predict)
print("single fit: ",score)

parameters = {'n_neighbors':[2,3,4,5], 'leaf_size':[30,35,40]}

clf = make_pipeline(preprocessing.StandardScaler(),
                    GridSearchCV(KNeighborsClassifier(),
                                 param_grid=parameters,
                                 cv=2,
                                 refit=True,n_jobs=1))

print("grid complete:")

clf.fit(ab_data.x_train,ab_data.y_train)
print("pipeline complete:",clf.score(ab_data.x_test,ab_data.y_test))

#DUMMY work
# pkr_dclf = dummy_plot(pkr_data)
# pkr_dclf.show()
# ab_dclf = dummy_plot(ab_data)
# ab_dclf.show()
#
print("no errors")
