from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

class RidgeReg:
    def __init__(self):
        self.param_grid = {
            'alpha' : [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20],
            'fit_intercept': [True, False],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'],
            'random_state': [42], 
            'max_iter': [1000, 2000, 3000],  
            'tol': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]  
            }
        self.ridge_model = None
        
    def model(self,x, y):
        ridge = Ridge()
        self.ridge_model = GridSearchCV(ridge, self.param_grid, cv=5)
        self.ridge_model.fit(x,y)
        
    def pred_model(self, x):
       return self.ridge_model.predict(x)


