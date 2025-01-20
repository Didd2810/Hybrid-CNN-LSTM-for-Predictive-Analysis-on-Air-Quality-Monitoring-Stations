import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error

class LSVM:
    def __init__(self):
        self.param_grid = {'kernel': ['linear', 'poly', 'rbf'],
                           'C': [0.1, 1, 10],
                           'degree': [2,3],
                           'gamma': ['scale', 'auto']}
        self.svm_model = SVC()
        
    def BestSearch(self, x_train, y_train):
        grid_search = GridSearchCV(estimator=self.svm_model, param_grid=self.param_grid, cv=5)
        grid_search.fit(x_train, y_train)
        print("Best parameters:", grid_search.best_params_)
        exit(0)