from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from LinearSVR import LSVR
from NonLinearSVR import NLSVR
from DTRegressor import DT
from KNNRegressor import KNNR
from RidgeReg import RidgeReg
from LassoReg import LassoReg
from RandomForestReg import RFReg
from XGBRegressor import XGBoost
from Metrics import Metrics

class ModelSelection:
    def __init__(self):
        self.type=None
        self.model=None
        self.x_train=None
        self.y_train=None
        
    def select_model(self, input):
        #self.type=input
        if 'LR' in input or 'lr' in input:
            if 'M' in input or 'm' in input:
                self.type = 'MLR'
            else: self.type = 'SLR'
            self.model = LinearRegression()
            #self.model.fit(x_train, y_train)
        elif input=='LSVR' or input=='Linear SVR' or input=='lsvr':
            self.type='LSVR'
            self.model = LSVR()
           #self.model.BestSearch(x_train, y_train, x_test, y_test)
        elif input=='NLSVR' or input=='Non-Linear SVR' or input=='nlsvr':
            self.type='NLSVR'
            self.model = NLSVR()
        elif input=='KNN' or input=='knn':
            self.type='KNN'
            self.model = KNNR()
        elif input=='DT' or input=='dt':
            self.type='DT'
            self.model=DT()
        elif input=='Ridge' or input=='ridge':
            self.type='Ridge'
            self.model=RidgeReg()
        elif input=='Lasso' or input=='lasso':
            self.type='Lasso'
            self.model=LassoReg()
        elif input=='RFR' or input=='RF Regressor' or input=='rfr':
            self.type='RFR'
            self.model=RFReg()
        elif input=='XGBR' or input=='xgbr' or input=='XGB':
            self.type='XGBR'
            self.model=XGBoost()
        else:
            self.model=None
        return self.model
        
    def fit(self, x_train, y_train, x_test, y_test, features, target, target_gas):
        self.x_train, self.y_train = x_train, y_train
        if self.type == 'MLR':
            self.model.fit(x_train, y_train)
        elif self.type == 'LSVR' or self.type == 'NLSVR':
            self.model.BestSearch(x_train, y_train, x_test, y_test)
        elif self.type == 'Ridge' or self.type=='Lasso' or self.type=='RFR' or self.type=='DT' or self.type=='XGBR':
            self.model.model(x_train,y_train)
 
    def types(self):
        return self.type
    
    def predict(self, next_value):
        y_pred = 1
        if self.type == 'MLR' or self.type=='DT':
            y_pred = self.model.predict([next_value])
        elif self.type == 'LSVR' or self.type=='NLSVR' or self.type=='Ridge' or self.type=='Lasso' or self.type=='RFR' or self.type=='XGBR':
            y_pred = self.model.pred_model([next_value])
        elif self.type == 'KNN':
            y_pred = self.model.pred_model([next_value], self.x_train, self.y_train)
        
        return y_pred