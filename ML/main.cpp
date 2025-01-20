import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as matplotlib
import seaborn as sns
from LinearPreProcessing import PreProcessor
from PCATrial import PCA_Test
from AQICalculation import AQI_calc
from LinearSVR import LSVR
#from sklearn.svm import LinearSVR
from DTRegressor import DT
from KNNRegressor import KNNR
from ModelSelect import ModelSelection
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from Metrics import Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

The dataset is taken from the esp-32 as shown in the first line but a fixed sample dataset will be used for this example
#file_path = 'esp-32-data.json' 
file_path = 'delhi_aqi.csv'
#data = pd.read_csv(file_path)

prep = PreProcessor() 
#metrics = Metrics()
#Converts to pandas dataframe, checks for date column and creates separate columns for later use
prep.PreProcessing(file_path) 

'''
#Checks for missing values in data
if prep.total_null_count()!=0:
    #Prints missing value count and percentage for both total and seperate column data
    prep.total_null_count()
    prep.column_null_count()
    #Uses K-Nearest Neighbours to fill in missing values with nearest neighbours set at 3
    prep.KNN()
'''
data = prep.gas_data

#exit(0)
#prep.KNN()
#prep.Time_Interpolate()
#Post filling values, extracting columns
data = prep.data
gas_data = prep.gas_data
target_gases = prep.target_gas
time_data = prep.time_data

#prep.pair_plot()

#Calculates stats of data including mean, std_dev, and the various quartiles
prep.calc_stats()

#Standardizes data
prep.standardize_data()
#Initiates PCA class
pca = PCA_Test()
#pca.tabulate_stats(gas_data, prep)

'''
Prelim tests to ensure data is suitable for PCA
Bartlett's Test:
    Null Hypothesis: 
        Correlation matrix is an identity matrix ergo no relationships b/w variables
    If p_value from test is below the signifance level of 0.05 then 
    null hypthesis can be rejected and PCA can be conducted
    KMo: asseses sampling adequacy, if above 0.6 then suitable
'''
prep.Corr_Analysis()
pca.Bartlett_Test(prep.correlation_matrix, gas_data.shape[0])
#exit(0)
pca.KMO_Test(gas_data)
final_data=[]

cov_matrix = np.corrcoef(prep.std_data, rowvar=False)
plt.figure(figsize=(10, 8))
sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', fmt ='.2f')
plt.title(f'Correlation Matrix')
plt.show()

cov_matrix = np.cov(prep.std_data, rowvar=False)
plt.figure(figsize=(10, 8))
sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', fmt ='.2f')
plt.title(f'Covariance Matrix')
plt.show()
#exit(0)
if pca.p_value<0.05 :
   #prep.standardize_data()
   #prep.Corr_Analysis()
   #exit(0)
   #kaiser_comps = pca.Kaiser_Criterion(prep.correlation_matrix, target_gases)
   #print(f'Kaiser Components: {kaiser_comps}')
   #pca.PCA_Fn(prep.std_data, target_gases, kaiser_comps)
   pc = pca.cal_PCA(prep.std_data, target_gases)
   #print(pca.eigenvalues)
   pca.cov_matrix
   pca.Cond_Num()
   #print(prep.vif_func())
   pca.Scree_Plot()
   #pca.CumSum()
   #final_data.append(pca.final_data)
 

else:
    final_data={}
    for gas in target_gases:
        final_data[gas]=target_gases[gas]
    



actual_aqi_vals = {}
pred_aqi_vals = {}
actual_values = {}
pred_values = {}

PC_metrics = {}

time_window = 6
pc_num = len(gas_data.columns)-2

'''
def pc_num_gen(gas):
    if gas=='co' or gas =='no2' or gas=='o3' or gas=='so2':
        return 5
'''
ms = ModelSelection()
#model = ms.select_model(input('Choose a model: '))
#model_name = ms.type

#for j in range(pc_num, -1, -1):
#for j in range(1):
j=5
model_name = 'A'
start_from =  24*6*5
end_at = 24*6*6
while (model_name!=None):
    model = ms.select_model(input('\nChoose a model: '))
    model_name = ms.type
    pc = pca.cal_PCA(prep.std_data, target_gases)
    pc = pca.filter_PC(j)
    metrics = Metrics()
    metrics_res = []
    #pc = pca.filter_PC(j)
    for target_gas in target_gases:
        in_features = pc[target_gas]
        out_target = prep.std_col(gas_data[target_gas])
        if (model_name == 'SLR'):
            out_target = prep.std_col(gas_data[target_gas])
            in_features = out_target
        x_train, x_test, y_train, y_test = train_test_split(in_features, out_target, test_size=0.3, random_state=43)
        ms.fit(x_train, y_train, x_test, y_test, in_features, out_target, target_gas)
        print('\nTarget gas: ' + target_gas + '\n')
        
        actual_vals = []
        pred_vals = []
        #mlr_vals = []
        #lsvr_vals = []
        #knn_vals=[]
        
        if model_name == 'KNN':
            model.model(gas_data.drop(columns=target_gas), gas_data[target_gas])
        #knn = KNNR()
        #knn.model(gas_data.drop(columns=target_gas), gas_data[target_gas])
        
        
        for i in range(start_from,end_at):
            
            features = in_features[:i+1]
            
            target = out_target.iloc[:i+1]
            
          
            next_val = in_features[i+1]
            y_pred = ms.predict(next_val)
            out_target = prep.destandardize_data(target_gas)
            y_actual = out_target.iloc[i+1]
            y_pred = prep.destandardize_value(y_pred, target_gas)
            if model_name == 'MLR':
                metrics.calc_stats(model)
            #coefficients = MLR_model.coef_
            #intercept = MLR_model.intercept_
           
            actual_vals = np.concatenate((actual_vals, np.array([y_actual])))
            if (model_name == 'SLR'):
                pred_vals = np.concatenate((pred_vals, np.array([y_pred])))
            else:
                pred_vals = np.concatenate((pred_vals, np.array(y_pred)))
            
            metrics.calc_daily_metrics(actual_vals, pred_vals, i, start_from, target_gas)
            #metrics.calc_daily_metrics(actual_vals, lsvr_vals, i, target_gas)

        actual_vals = np.array(actual_vals) 
        pred_vals = np.array(pred_vals)
        
        actual_values[target_gas] = actual_vals
        pred_values[target_gas] = pred_vals
        hours = range(len(actual_vals))
        days = [hour / (24*6 + 1) for hour in hours]
        metrics.graph_values(actual_vals, pred_vals, days, target_gas, model_name)
        
        metrics.graph_metrics(actual_vals, pred_vals, target_gas, days, model_name)
        metrics=Metrics()
        #calc_overall_metrics(self, actual_vals, pred_vals, target_gas, gases, num, model_type)
        metrics.calc_overall_metrics(actual_vals, pred_vals, target_gas, gas_data, j, model_name)
        
        metrics_res.append(metrics.metrics_res)
        metrics.bar_metrics(target_gas)
    metrics = pd.DataFrame(metrics_res)
    print(f'\nMetrics Results for {model_name}:')
    print(metrics)
    aqi = AQI_calc()
    for gas in target_gases:
        if len(actual_values[gas]) >= 24 and len(pred_values[gas]) >= 24:
            
            #actual_aqi_vals.append(sub_aqi(i, actual_values[i]))
            actual_aqi_vals[gas] = aqi.sub_aqi(gas, actual_values[gas])
            #pred_aqi_vals.append(sub_aqi(i, pred_values[i]))
            pred_aqi_vals[gas] = aqi.sub_aqi(gas, pred_values[gas])
            
            if gas=='CO' or gas=='NO2' or gas=='PM2_5' or gas=='PM10':
                actual_aqi_vals[gas] = actual_aqi_vals[gas][0] 
                pred_aqi_vals[gas] = pred_aqi_vals[gas][0]

    actual_aqi_values = []
    pred_aqi_values = []
    #i is row and j is column

    for i in range(len(actual_aqi_vals['CO'])):
        actual_aqi_value=[]
        pred_aqi_value=[]
        for gas in target_gases:
            actual_aqi_value.append(actual_aqi_vals[gas][i])
            pred_aqi_value.append(pred_aqi_vals[gas][i])
            
        actual_aqi = aqi.calc_aqi(actual_aqi_value)
            #print(f'AQI value: {aqi}')
        pred_aqi = aqi.calc_aqi(pred_aqi_value)
        
        actual_aqi_values.append(actual_aqi)
        pred_aqi_values.append(pred_aqi)
        



    plt.figure(figsize=(10, 6))
    actual_aqi_values = np.array(actual_aqi_values)
    actual_aqi_values = actual_aqi_values.reshape(actual_aqi_values.shape[-1])
    pred_aqi_values = np.array(pred_aqi_values)
    pred_aqi_values = pred_aqi_values.reshape(pred_aqi_values.shape[-1])
    hours = range(len(actual_vals))
    days = [hour / (24*6 + 1) for hour in hours]
    plt.xticks(range(len(actual_vals)), labels=[str(i) if i%1==0 else str("") for i in range(len(days))])
    plt.plot(days, actual_aqi_values, label='Actual AQI')
    plt.plot(days, pred_aqi_values, label='Predicted AQI')
    #plt.xlim(30, 35)
    plt.title('Actual vs Predicted AQI')
    plt.xlabel('Day')
    plt.ylabel('AQI')
    plt.legend()
    plt.show()

    i=0
    for gas in target_gases:
        plt.figure(figsize=(10, 6))
        actual_aqi_vals[gas] = np.array(actual_aqi_vals[gas])
        actual_aqi_vals[gas] = actual_aqi_vals[gas].reshape(actual_aqi_vals[gas].shape[-1])
        pred_aqi_vals[gas] = np.array(pred_aqi_vals[gas])
        pred_aqi_vals[gas] = pred_aqi_vals[gas].reshape(pred_aqi_vals[gas].shape[-1])
        plt.xticks(range(len(actual_vals)), labels=[str(i) if i%1==0 else str("") for i in range(len(days))])
        plt.plot(days, actual_aqi_vals[gas], label='Actual AQI')
        plt.plot(days, pred_aqi_vals[gas], label='Predicted AQI')
        #plt.xlim(30, 35)
        i=i+1
        plt.title(f'Actual vs Predicted AQI for {gas}')
        plt.xlabel('Day')
        plt.ylabel('AQI')
        plt.legend()
        plt.show()


    #plt.figure(figsize=(10,6))
    categories = []

    for i in range(len(actual_aqi_values)):
        categories.append(aqi.calc_aqi_category(actual_aqi_values[i]))
        
    color_code = {}
    colors = ['g', 'lime', 'y', 'orange', 'r', 'maroon' ]
    for i in range(len(aqi.aqi_categories)):
        color_code[aqi.aqi_categories[i][2]] = colors[i]
        
    #plt.figure(figsize=(8,4),dpi=200)
    #plt.bar()
