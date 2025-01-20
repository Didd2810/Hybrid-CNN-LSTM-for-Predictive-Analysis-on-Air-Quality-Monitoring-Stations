import numpy as np
import pandas as pd
#import matpotlib as matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Qt5Agg')
import seaborn as sns
#import aqi
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

class Metrics:
    def __init__(self):
        #metric_names = ['R-score', 'R^2 score', 'MAE', 'MSE', 'RMSE']
        #metric_vals = []
        self.r = {}
        self.r_2 = {}
        self.mae = {}
        self.mse = {}
        self.rmse ={}
        self.r_pc = [[]]
        self.r_2_pc = []
        self.mae_pc = []
        self.mse_pc = []
        self.rmse_pc = []
        self.coefficients=None
        self.intercept=None
        self.metrics_res=None
        
    def calc_stats(self, model):
        self.coefficients = model.coef_
        self.intercept = model.intercept_
        
    def calc_daily_metrics(self, actual_vals, pred_vals, i, num, gas):
      
        if i>num:
            r,_ = pearsonr(actual_vals, pred_vals)
            self.r[gas].append(r)
            self.r_2[gas].append(r2_score(actual_vals, pred_vals))
            self.mae[gas].append(mean_absolute_error(actual_vals, pred_vals))
            self.mse[gas].append(mean_squared_error(actual_vals, pred_vals))
            self.rmse[gas].append(np.sqrt(mean_squared_error(actual_vals, pred_vals)))
            
        else:
            self.r[gas]=[]
            self.r_2[gas]=[]
            self.mae[gas]=[]
            self.mse[gas]=[]
            self.rmse[gas]=[]
            self.r[gas].append(0)
            self.r_2[gas].append(0)
            self.mae[gas].append(0)
            self.mse[gas].append(0)
            self.rmse[gas].append(0)
        
    def calc_overall_metrics(self, actual_vals, pred_vals, gas, gases, num, model_type):
        self.r[gas],_ = pearsonr(actual_vals, pred_vals)
        self.r_2[gas] = r2_score(actual_vals, pred_vals)
        self.mae[gas] = mean_absolute_error(actual_vals, pred_vals)
        self.mse[gas] = mean_squared_error(actual_vals, pred_vals)
        self.rmse[gas] = np.sqrt(self.mse[gas])
        
        metrics = [self.r[gas]*100, self.r_2[gas]*100, self.mae[gas], self.mse[gas], self.rmse[gas]]
        #return metrics
        #print(f'\n{model_type}: Metrics for PC_Num = {len(gases.columns)-1-num}\n')
        self.metrics_res = ({'Gas': gas, 'R': self.r[gas], 'R2': self.r_2[gas], 'MAE': self.mae[gas], 'MSE': self.mse[gas], 'RMSE': self.rmse[gas]})    
        #print(self.metrics_res)
        '''
        print(f'\nMetrics for {model_type}')
        metric_names = ['R-score', 'R^2 score', 'MAE', 'MSE' ,'RMSE']
        print(f'{metric_names[0]} for {gas}: {self.r[gas]*100}%')
        print(f'{metric_names[1]} for {gas}: {self.r_2[gas]*100}%')
        print(f'{metric_names[2]} for {gas}: {self.mae[gas]}')
        print(f'{metric_names[3]} for {gas}: {self.mse[gas]}')
        print(f'{metric_names[4]} for {gas}: {self.rmse[gas]}')
        '''
    def graph_PC(self, col_size, num):
        plt.figure(figsize=(10,6))
        plt.title(f'netrics for each PC columns')
        #plt.xticks(range(col_size), label = str(i) for i in range(col_size) )
    
    def graph_values(self, actual_vals, pred_vals, days, target_gas, model_type):
        plt.figure(figsize=(10, 8))
        print(actual_vals)
        plt.xticks(range(len(actual_vals)), labels=[str(i) if i%1==0 else str("") for i in range(len(days))])
        plt.plot(days, actual_vals, label='Actual')
        plt.plot(days, pred_vals, label='Predicted')
        plt.title(f'{model_type}: Actual vs Predicted for {target_gas}')
        plt.xlabel('Day')
        plt.ylabel('Gas Concentration')
        plt.legend()    
        plt.show()
        
    def graph_metrics(self, actual_vals, pred_vals, target_gas, days, model_type):
        metrics_list = {'R': self.r[target_gas], 'R^2': self.r_2[target_gas], 'MAE':self.mae[target_gas], 'MSE': self.mse[target_gas], 'RMSE': self.rmse[target_gas]}
        for name, value in metrics_list.items():
            plt.figure(figsize=(10, 6))
            plt.title(f'{model_type}: Daily {name}-score Values for {target_gas}')
            #plt.xticks(range(len(actual_vals)), labels=[str(i) if i%1==0 else str("") for i in range(len(days))])
            print(days)
            print(value)
            plt.plot(days, value)
            #plt.plot(days, self.r_2[target_gas], label = 'R^2-score')
            #plt.plot(days, self.mae[target_gas], label = 'MAE-score')
            #plt.plot(days, self.mse[target_gas], label = 'MSE-score')
            #plt.plot(days, self.rmse[target_gas], label = 'RMSE-score')
            plt.xlabel('Day')
            plt.ylabel('Value')
            #plt.legend()
            plt.show
        
    def print_metrics(self, target_gas, gases, num, model_type):
        
        #metric_vals = [self.r[target_gas]*100, self.r_2[target_gas]*100, self.mae[target_gas], self.mse[target_gas], self.rmse[target_gas]]
        print(f'{model_type}: Metrics for PC_Num = {len(gases.columns)-1-num}\n')
        metric_names = ['R-score', 'R^2 score', 'MAE', 'MSE' ,'RMSE']
        print(f'{metric_names[0]} for {target_gas}: {self.r[target_gas]*100}%')
        print(f'{metric_names[1]} for {target_gas}: {self.r_2[target_gas]*100}%')
        print(f'{metric_names[2]} for {target_gas}: {self.mae[target_gas]}')
        print(f'{metric_names[3]} for {target_gas}: {self.mse[target_gas]}')
        print(f'{metric_names[4]} for {target_gas}: {self.rmse[target_gas]}')
        
    def bar_metrics(self, target_gas):
        metric_vals = [self.r[target_gas]*100, self.r_2[target_gas]*100, self.mae[target_gas], self.rmse[target_gas]]
        metric_names = ['R-score', 'R^2 score', 'MAE', 'RMSE']
        ax = plt.figure(figsize=(10, 6))
        plt.bar(metric_names, metric_vals)
        plt.title(f'Overall Metrics for {target_gas}')
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.legend
        for i in ax.patches:
            plt.text(i.get_width()+0.2, i.get_y()+0.5, str(round((i.get_width()), 2)),fontsize = 10, fontweight ='bold', color ='grey')
        plt.show()
        
        
    