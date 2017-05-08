import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import KFold, cross_val_score
import xgboost
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score,log_loss
import pickle
import itertools
import sys
from plot import Plot
from parameterTuning import Tuning
from featureSelection import dataAnalysis
import warnings
warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",DeprecationWarning)
import logging
import os

logging.basicConfig(format='%(asctime)s %(message)s',filename=os.getcwd()+"\\main.log")
logging.setlevel(logging.DEBUG)
logger = logging.getLogger(__name__)

def getDummies(data,features) :
    ddata = pd.DataFrame()
    for feature in features :
        dcol = pd.get_dummies(data[feature],prefix=feature)
        ddata = pd.concat([ddata,dcol,],axis=1)
    return ddata

def cross_validation_score(clf_xgb,X_train,y_train) :
    kf = KFold(len(X_train),n_folds=5)
    scores=[]
    logloss = []
    i=0
    f = open("metrics.txt",'wb')
    params = clf_xgb.get_xgb_params()
    f.write("Metrics for model with params \n"+ str(params)+"\n\n")
    f.write("FOLD \t ACCURACY \t LOG_LOSS \t\n")
    for tr_index,ts_index in kf :
        i=i+1
        X1_tr,y1_tr = X_train.iloc[tr_index],y_train.iloc[tr_index]
        X1_ts,y1_ts = X_train.iloc[ts_index],y_train.iloc[ts_index]
        print "Fold ", i
        start = time.time()
        clf_xgb.fit(X1_tr,y1_tr)
        end = time.time()
        print "Time taken to build model : ",end-start
        y_pred = clf_xgb.predict(X1_ts)
        print "Calculating Metrics\n"
        acc = accuracy_score(y1_ts,y_pred)
        loss = log_loss(y1_ts,y_pred)
        print "accuracy for fold ",i," : ",acc
        print "logloss for fold ", i," : ",loss 
        scores.append(acc)
        logloss.append(loss)
        try :
            f.write(str(i)+" \t "+str(acc)+" \t "+str(loss)+" \t \n")
        except Exception as e :
            print e
    f.write("\nOver-all accuracy : "+str(np.mean(scores)))
    f.close()

if __name__ == '__main__' :
    if len(sys.argv) > 1 :
        data_path = sys.argv[1]
    else :
        print "Usage : python main.py <path_to_data>"
        raise Exception("Please specify the path to data.csv")

    data = pd.read_csv(data_path)
    analyser = dataAnalysis(data)
    data,features = analyser.analyse(savePlot=False,showPlot=False)
    ddata = getDummies(data,features)
    train_data = ddata[data['shot_made_flag'].notnull()]
    test_data = ddata[data['shot_made_flag'].isnull()]
    '''
    Parameters are tuned using GridSearchCV with params :
    max_depth=[6,7,8], min_child_weight=[3.5,4,4.5], colsample_bytree=[0.6,0.65,0.7]
    subsample = range(0.6,0.7,0.2)(5 values), reg_alpha=[3,4,5] with learning_rate=0.012 and n_estimators=1000.
    Best Params obtained after performing GridSearchCV are : {'reg_alpha': 5, 'colsample_bytree': 0.6,'min_child_weight': 4, 'subsample': 0.62}
    '''
    
    try :
        clf_xgb = pickle.load(open("xgb.pickle.dat", "rb"))
        logger.info("Loaded trained model from xgb.pickle.dat")
        logger.info(clf_xgb.get_xgb_params())
    except Exception as e :
        clf_xgb = XGBClassifier(max_depth=7,min_child_weight=4,learning_rate=0.012, n_estimators=1000, subsample=0.62, colsample_bytree=0.6,reg_alpha=4.5,seed=1,silent=1)
        logger.info(clf_xgb.get_xgb_params())
        y_train = data.shot_made_flag[data['shot_made_flag'].notnull()]
        logger.info("Fitting xgb classifier")
        print "Fitting xgb classifier"
        start = time.time()
        clf_xgb.fit(train_data,y_train)
        end = time.time()
        time_taken = str(end-start)
        logger.info("Model trained in %s seconds",'time_taken')
        print "Model trained in ",end-start,"seconds"
        logger.info("Dumping model to xgb.pickle.data for future use")
        print "Dumping model to xgb.pickle.data for future use"
        f = open("xgb.pickle.dat","wb")
        pickle.dump(clf_xgb,f)
        f.close()
    
    logger.info("Evaluating model performance on test data")
    result = clf_xgb.predict_proba(test_data)[:,1]
    shot_id = data[data.shot_made_flag.isnull()]['shot_id']
    result_data = pd.DataFrame({'shot_id' : shot_id,'shot_made_flag':result})
    result_data.sort_values('shot_id',inplace=True)
    result_data.to_csv('result_xgb_final.csv',index=False)
    logger.info("Results saved to result_xgb_final.csv")
    print "Results saved to result_xgb_final.csv"
    
    

