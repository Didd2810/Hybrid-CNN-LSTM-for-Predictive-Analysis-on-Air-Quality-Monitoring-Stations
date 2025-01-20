from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import make_classification

class LSVR:
    def __init__(self):
        self.param_grid = {'kernel': ['linear'], #third kernel prev ha 'rbf'
                           'C': [0.1, 1, 10],
                           'degree': [2,3],
                           'gamma': ['scale', 'auto']}
        self.svr_model = SVR()
        self.best_model = None
        
    def BestSearch(self, x_train, y_train, x_test, y_test):
        grid_search = GridSearchCV(estimator=self.svr_model, param_grid=self.param_grid, cv=5)
        grid_search.fit(x_train, y_train)
        #print(f'Best parameters: {grid_search.best_params_}')
        #print(grid_search.best_estimator_)
        self.best_model = grid_search.best_estimator_
        #y_pred = self.best_model.predict(x_test)

        #mse = mean_squared_error(y_test, y_pred)
        #print("Mean Squared Error:", mse)
        
    def pred_model(self, x):
        return self.best_model.predict(x)