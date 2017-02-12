# KOBE-SHOT-SELECTION

### Understanding Data

The given data contains the location and circumstances of every basket attempted by Kobe Bryant during his 20-year career.Our task is to predict whether Kobe was successful with his shot or not, this is represented by shot_made_flag with 1 meaning shot is successful and 0 meaning missed shot.Data consists a total of 30,697 records with 25 columns consisting of different fields and out of these 30,697 records 5000 records are given to us for testing i.e. with blank shot_made_flag.25 columns of data are fields such as action_type, combined_shot_type, game_event_id, game_id,lat, loc_x, loc_y, lon, minutes_remaining, period, playoffs, season , seconds_remaining, shot_distance, shot_made_flag, shot_type, shot_zone_area, shot_zone_basic, shot_zone_range, team_id, team_name, game_date, matchup, opponent, shot_id. Of these all fields only some of them are important to decide whether the shot is successful or not. Selection of fields important in predicting the shot(Feature Engineering) is documented in "featureSelection.py" along with some plots in "Plots" folder and is self-explanatory.


### Building Model with XGBoost 

I choose XGBoost(eXtreme Gradient Boosting) algorithm. It works on the principle of ensemble, which combines the prediction of multiple trees together. It’s a highly sophisticated algorithm, an advanced implementation of gradient boosting algorithm and is powerful enough to deal with all sorts of irregularities(variance & bias) in data. The implementation of the model supports the features of the scikit-learn (Python) and R implementations, with new additions like regularization.

#### Tuning parameters 
XGBoost algorithm has 3 types of parameters : General, Booster, Learning Task parameters
Parameter tuning is done using [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) with following values.
max_depth=[6,7,8], min_child_weight=[3.5,4,4.5], colsample_bytree=[0.6,0.65,0.7],subsample = range(0.6,0.7,0.2)(5 values), reg_alpha=[3,4,5] with learning_rate=0.012 and n_estimators=1000.
Best params for our model after performing GridSearchCV are : {'reg_alpha': 5, 'colsample_bytree': 0.6,'min_child_weight': 4, 'subsample': 0.62} leading to an [log_loss](https://www.kaggle.com/wiki/LogarithmicLoss) of 0.60098 and over-all cross-validation score(accuracy) with 5-folds being 68.11%

### How to run the code 

#### Pre-requisites 
1. Numpy         :  pip install numpy 
2. Pandas        : pip install pandas  
3. Scipy         : Download appropriate wheel matching your python version and os from                                                                             [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy) and type pip install name_of_wheel_file. Need to install numpy+mkl package first.
4. xgboost       : [Follow this link](http://www.picnet.com.au/blogs/guido/post/2016/09/22/xgboost-windows-x64-binaries-for-download/)
5. scikit-learn : pip install scikit-learn

### Usage 
> python main.py path-to-data-file

### Module Details
* featureSelection.py contains information about discarded and considered fields from data with appropriate plots in folder named 'Plots'.
* plot.py contains plotting functions which are used by featureSelection.py
* parameterTuning.py implements GridSearchCV with specified params and returns the best_params
* Metrics.txt contains accuracy for each fold with final parameters
* xgb.pickle.dat contains the trained model which can be loaded directly instead of training model again.(Total time taken to train the model: 1100s)


    
