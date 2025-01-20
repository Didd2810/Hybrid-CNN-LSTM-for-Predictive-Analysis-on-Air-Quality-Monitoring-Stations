from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

class LassoReg:
    def __init__(self):
        self.param_grid = {
            'alpha' : [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20],
            'fit_intercept': [True, False],
            'max_iter': [5000, 10000, 50000],  
            'tol': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
            'selection': ['cyclic', 'random']
            }
        self.lasso_model=None
    
    def model(self,x, y):
        lasso = Lasso()
        self.lasso_model = GridSearchCV(lasso, self.param_grid, cv=5)
        self.lasso_model.fit(x,y)
        
    def pred_model(self, x):
       return self.lasso_model.predict(x)