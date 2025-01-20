import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
    
class RFReg:
    def __init__(self):
        self.param_grid = {
            'n_estimators' : [int(x) for x in np.linspace(start=100, stop=1200, num=12)],
            'criterion': ['mse', 'squared_error','absolute_error', 'friendman_mse'],
            'max_features' : [1.0, 'sqrt', 'log2', None],
            'max_depth' : [int(x) for x in np.linspace(5, 30, num=6)],
            'min_samples_split' : [2, 5, 10, 15, 20],
            'min_samples_leaf' : [1, 2, 5, 10], 
            'bootstrap': [False],
            'warm_start':[False]
            }
        self.rf_model=None
    
    def model(self,x, y):
        RF_reg = RandomForestRegressor()
        self.rf_model = RandomizedSearchCV(RF_reg, self.param_grid, scoring='neg_mean_squared_error', 
                            cv=5, n_iter=100, random_state=43, n_jobs=-1)
        self.rf_model.fit(x,y)
        
    def pred_model(self, x):
       return self.rf_model.predict(x)
