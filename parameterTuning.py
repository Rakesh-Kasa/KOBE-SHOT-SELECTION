import logging
from sklearn.model_selection import GridSearchCV

logging.basicConfig(filename="gridsearch.log")
logger = logging.getLogger(__name__)
class Tuning() :
    def __init__(self,model,params) :
        self.model = model
        self.params = params

    def tuneModel(self,X_train,y_train) :
        clf_grid = GridSearchCV(self.model, self.params, scoring="neg_log_loss",cv=5,verbose=1)
        grid_result = clf_grid.fit(X_train,y_train)
        logging.info("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
                
