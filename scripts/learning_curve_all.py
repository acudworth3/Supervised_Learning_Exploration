from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit
import preprocess as prp
from sklearn import metrics
import numpy as np
from joblib import dump, load
from models import generic_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from seaborn import distplot
from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import GridSearchCV



#pkr_data
#TODO move into preprocess
pkr_data = prp.pkr_data()
pkr_data.clean()
pkr_data.init_model_data(target =['hand'],features = ['suit1','card1','suit2','card2','suit3','card3','suit4','card4'])#

#ab data
ab_data = prp.ab_data()
ab_data.clean()
ab_data.target = 'room_type'
ab_data.features = ab_data.all.columns[ab_data.all.columns != ab_data.target]
ab_data.init_model_data(target=ab_data.target,features=ab_data.features)



# x_vars=list(ab_data.features)
# x_vars.append(ab_data.target)
# ab_corell_data = ab_data.all

# ab_corell_plt_obj = sns.pairplot(ab_corell_data,x_vars=ab_data.features,y_vars=ab_data.target,hue='room_type',diag_kind='hist')
# plt.suptitle('Airbnb Feature vs Target Grouping')
# plt.subplots_adjust(top=0.9)
# ab_corell_plt_obj.savefig('plt_ab_correl.png')
# plt.legend('top')
# plt.close()
#
# #Pkr
# pk_corell_data = pkr_data.all
# pk_corell_plt_obj = sns.pairplot(pk_corell_data,x_vars=pkr_data.features,y_vars=pkr_data.target,hue='hand',diag_kind='hist')
# plt.title("this is ugly make it a box plot")
# pk_corell_plt_obj.savefig('plt_pk_corell')
# plt.close()
# plt.show()
# le

#
# loaded_clf_ab = load('clf_DT_ada_gridabnb_final_0.8894.joblib')
ab_model_dict = {'Boosted DT (AirBnb Data)':'clf_DT_ada_gridabnb_final_0.8894.joblib',
                    'DT (AirBnb Data)':'clf_DT_gridabnb_final_0.8277.joblib',
                    'KNN (AirBnb Data)':'clf_KNN_gridabnbfinal_score_0.8137797810688989.joblib',
                    'NN (AirBnb Data)':'clf_NN_abnb_final_0.8273.joblib'}

pk_model_dict = {'Boosted DT (Poker Data)':'clf_DT_ada_gridpkr data_final_0.9722.joblib',
                    'DT (Poker Data)':'clf_DT_gridpkr data_final_0.9722.joblib',
                    'KNN (Poker Data)':'clf_KNN_gridpkr datafinal_score_0.9718535103150487.joblib',
                    'NN (Poker Data)':'clf_NN_pkr data_final_0.9728.joblib',
                    'SVM (Poker Data)':'clf_SVM_gridpkr data_final_0.9722.joblib'}

# ab_model_dict = {'SVM -Linear (AirBnb Data (sample))':'SVM_ab_linear.joblib'}
# pk_model_dict = {'SVM -Linear (Poker Data)':'SVM_pkr_linear.joblib'}

for title in pk_model_dict.keys():

    model = pk_model_dict[title]
    loaded_clf_pk = load(model)
    clf_pk = loaded_clf_pk.named_steps.gridsearchcv.best_estimator_
    # clf_pk = load(pk_model_dict[title])
    pkr_pass = prp.pkr_data()
    pkr_accur_grid_model = generic_model(pkr_pass,clf_pk,title)
    pkr_accur_grid_model.make_plots(roc=False,prec_rec=False,cnf_mtr=False)
    print("completed: ",title)


for title in ab_model_dict.keys():

    model = ab_model_dict[title]
    loaded_clf_ab = load(model)
    clf_ab = loaded_clf_ab.named_steps.gridsearchcv.best_estimator_

    clf_ab = load(ab_model_dict[title])
    ab_pass = prp.ab_data()
    ab_accur_grid_model = generic_model(ab_pass,clf_ab,title)
    ab_accur_grid_model.make_plots(roc=False,prec_rec=False,cnf_mtr=False)

    print("completed: ",title)

