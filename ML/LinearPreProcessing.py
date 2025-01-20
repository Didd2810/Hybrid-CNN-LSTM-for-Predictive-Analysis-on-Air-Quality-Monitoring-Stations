import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from prettytable import PrettyTable 
from statsmodels.stats.outliers_influence import variance_inflation_factor

class PreProcessor:
    def __init__(self):
        self.data=None
        self.total_count = 0
        self.gas_data=None
        self.target_gas=None
        self.time_data=None
        self.target_gas_columns = ['CO', 'NO2', 'O3', 'SO2', 'PM2_5', 'PM10']
        #self.target_gas_columns = ['CO', 'NO2', 'O3', 'PM2_5', 'PM10']
        self.std_data=None
        self.mean_vals=None
        self.std_vals=None
        self.correlation_matrix=None
        self.myTable=None
        self.min_vals=None
        self.q1=None
        self.q2=None
        self.q3=None
        
    def PreProcessing(self, file_path):
        #self.data = self.convert_to_df(file_path)
        self.data = pd.read_csv(file_path)
        #self.data.drop(['so2', 'nh3', 'no'], axis=1, inplace=True)
        #self.data.columns = [col.capitalize() for col in self.data.columns]
        self.data.columns = self.data.columns.str.upper()
        #print(self.data.columns)
        self.total_count = self.data.size
        if self.data.columns[0]=='date' or self.data.columns[0][0]=='D':
            self.data[self.data.columns[0]] = pd.to_datetime(self.data[self.data.columns[0]])
            self.time_data = self.data[self.data.columns[0]]
            self.gas_data = self.data.drop(columns=self.data.columns[0])
            self.target_gas = self.data[self.target_gas_columns]
            
    def convert_to_df(self, filepath):
        data = []
        dataset = None
        with open(filepath, 'r') as file:
            dataset = json.load(file)
        #dataset = json.dumps(json_data)
        for _, dates in dataset.items():
            #print(dates)
            for date, timestamps in dates.items():
                for time, gas_values in timestamps.items():
                    if 'AQI' in gas_values:
                        del gas_values['AQI']
                    row = {'Date': date + ' ' + time}  # Combine date and time
                    row.update(gas_values)  # Add gas values
                    data.append(row)
        df = pd.DataFrame(data, columns=['Date', 'CO', 'NO2', 'O3', 'SO2', 'PM10', 'PM2_5'])
        #df = df[cols]
        #print(df)
        df['Date'] = pd.to_datetime(df['Date'], format='%d_%m_%Y %H:%M')
        #df.set_index('Date', inplace=True)
        return df
    
    def KNN(self):
        imputer = KNNImputer(n_neighbors=10)
        if self.total_null_count() or self.total_zero_count():
            print('Imputing values...')
            self.gas_data.replace(0, None, inplace=True)
            self.gas_data = pd.DataFrame(imputer.fit_transform(self.gas_data), columns=self.gas_data.columns)
            
    def Time_Interpolate(self):
        if self.total_null_count() or self.total_zero_count():
            print('Interpolating values...')
            self.gas_data.replace(0,None,inplace=True)
            self.gas_data = pd.DataFrame(self.gas_data, index=self.data['date'])
            self.gas_data.index = pd.date_range(start=self.gas_data.index[0], periods=len(self.gas_data), freq='H')
            print(self.gas_data.index.freq)
            self.gas_data = self.gas_data.interpolate(method='time')
        
    def total_null_count(self):
        missing_count =  self.data.isnull().sum()
        total_missing = missing_count.sum()
        percent_missing = (total_missing/self.total_count) * 100
        if total_missing!=0:
            print(f'Total Number of Missing Values: {total_missing}')
            print(f'Percentage of Missing values: {percent_missing:.1f}%')
        return total_missing
    
    def total_zero_count(self):
        zero_count = (self.data == 0).sum()
        total_zero = zero_count.sum()
        percent_zero = (total_zero/self.total_count)*100
        if total_zero!=0:
            print(f'Total Number of Zero Values: {total_zero}')
            print(f'Percentage of Missing values: {percent_zero:.1f}%')
        return total_zero
            
    def column_null_count(self):
        for gas in self.data:
            col_data = self.data[gas]
            missing_count =  col_data.isnull().sum()
            total_count = col_data.size
            total_missing = missing_count.sum()
            percent_missing = (total_missing/total_count) * 100
            print(f'Total Number of Missing Values for {gas}: {total_missing}')
            print(f'Percentage of Missing values for {gas}: {percent_missing:.1f}%')
    '''
    def standardise_data(self, gas_data):
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(gas_data)
        standardized_df = pd.DataFrame(standardized_data, columns=self.target_gas_columns)
        return standardized_df
     '''      
    def calc_stats(self):  
        
        print(self.gas_data.describe())
        for gas in self.target_gas_columns:
            box_gas =  self.gas_data.drop(columns=gas)
            sns.boxplot(data=box_gas)
            '''
            self.mean_vals=np.mean(self.gas_data, axis=0)
            self.std_vals=np.std(self.gas_data, axis=0)
            self.min_vals=np.min(self.gas_data,axis=0)
            self.max_vals=np.max(self.gas_data,axis=0)
            quartiles = np.percentile(self.gas_data, [25, 50, 75])
            self.q1, self.q2 , self.q3 = quartiles
            '''
     
    def std_col (self, data):
        mean = np.mean(data)
        std_dev = np.std(data)
        std_col = (data-mean)/std_dev
        return std_col 
    
    def standardize_data(self):
        self.mean_vals=np.mean(self.gas_data, axis=0)
        self.std_vals=np.std(self.gas_data, axis=0)
        self.std_data = (self.gas_data - self.mean_vals)/self.std_vals
        #print(self.std_data)
        #return self.std_data
        '''
        for gas in self.gas_data:
            scaler = StandardScaler()
            standardized_gas = scaler.fit_transform(self.gas_data[gas])
            std_gas = pd.DataFrame(standardized_gas)
        self.std_data.append(std_gas)
        '''
    def destandardize_data(self, gas):
        return (self.std_data[gas]*self.std_vals[gas]) + self.mean_vals[gas]
    
    def destandardize_value(self,value,gas):
        return (value*self.std_vals[gas]) + self.mean_vals[gas]
    
    def destd_PC(self, cols, PC):
        mean_vals = np.mean(self.gas_data[cols], axis=0)
        std_vals = np.std(self.gas_data[cols], axis=0)
        PC = pd.DataFrame(PC, columns=cols)
        return (PC*std_vals) + mean_vals
    
    def vif_func(self):
        vif_data = {}
        vif_data_res = {}
        print(self.gas_data.columns)
        for gas in self.target_gas:
            input_feat = self.gas_data.drop(columns=gas)
            vif_data[f"VIF for {gas}"] = [variance_inflation_factor(input_feat, i) for i in range(input_feat.shape[1])]
            vif_data_res[gas] = vif_data
        return vif_data
        '''
        for self.target_gas, vif_data in vif_data_res.items():
            print("VIF results for", self.target_gas)
            print(vif_data)
        '''
        
    def Corr_Analysis(self):
        self.correlation_matrix={}
        #for gas in self.target_gas:
        #corr_data = self.std_data.drop(columns=gas)
        corr_data = self.std_data
        #self.correlation_matrix[gas] = corr_data.corr()
        self.correlation_matrix = corr_data.corr()
        #print(f'corr matrix for {gas}: {self.correlation_matrix[gas]}')
        #print('correlation matrix\n{self.correlation_matrix}')
        plt.figure(figsize=(10, 8))
        #sns.heatmap(self.correlation_matrix[gas], annot=True, cmap='coolwarm', fmt ='.2f')
        sns.heatmap(self.correlation_matrix, annot=True, cmap='coolwarm', fmt ='.2f', annot_kws={"size": 10})
        #plt.title(f'Correlation Matrix for target gas: {gas}')
        plt.title(f'Correlation Matrix')
        plt.show()
            
            
    
    def add_rows(self, header, row):
        headers=[]
        headers.append(header)
        for i in range(len(row)):
            headers.append(row[i])
        if header=='Stats':
            self.myTable = PrettyTable(headers)
        else:
            self.myTable.add_row(headers)
        
    def pair_plot(self):
        plt.figure(figsize=(10, 6)) 
        sns.pairplot(self.gas_data)
        plt.show
    
   
        
        
        