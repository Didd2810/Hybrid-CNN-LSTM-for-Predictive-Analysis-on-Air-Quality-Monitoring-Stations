import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import aqi
from LinearPreProcessing import PreProcessor
from PCATrial import PCA_Test
#from AQICalculation import AQI_calc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from Metrics import Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.model_selection import GridSearchCV

class DT:
    def __init__(self):
        '''
        self.x_train =  None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.tree=None
        self.in_features=None
        self.target=None
        self.target_gas = None
        '''
        self.param_grid = {
            'splitter': ['best', 'random'],
            'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
            'min_samples_leaf': list(range(1,6)),
            'min_weight_fraction_leaf': [i/10 for i in range(1, 5)],
            'max_features': [1.0],
            'max_leaf_nodes': [None, 10, 20, 30, 40, 50, 60, 70]
        }
        self.tree_model=None
        
        
    def model(self, x_train, y_train):
        
        self.tree =  DecisionTreeRegressor(criterion='friedman_mse')
        self.tree_model = GridSearchCV(self.tree, self.param_grid, cv=5)
        self.tree_model.fit(x_train, y_train)
        self.tree_model.best_params_
        
    def predict(self,value):
        return self.tree_model.predict(value)
        
        
    def tree_graph(self, gas_data):
        # Create DOT data
        #gas_data.drop(columns=[self.target_gas], inplace=True)
        #print(str(i) for i in range(gas_data.shape[1]))
        dot_data = export_graphviz(self.tree, out_file=None, 
                                feature_names=[str(i) for i in range(gas_data.shape[1])],  
                                class_names=self.target.index)
        
        graph = pydotplus.graph_from_dot_data(dot_data)
        # Show graph
        Image(graph.create_png())   
        exit(0)


