import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from LinearPreProcessing import PreProcessor
#from docx import Document
from scipy.stats import chi2

class PCA_Test:
    def __inti__(self):
        self.data_set=None
        self.p_value=None
        self.kmo_model=None
        self.pca=None
        self.explained_variance_ratio=None
        self.final_data=None
        self.eigenvalues=None
        self.eigenvectors=None
        self.evr=None
        self.target_gases=None
        self.pc=None
        self.mod_data=None
        self.size=None
        self.cov_matrix = None
        
        
    #def gas_data (data):
     #   self.data_set=data
        
    #data = pd.read_csv("Delhi_aqi.csv")

    #X = data.drop(columns=['date'])  # Assuming 'date' is not a feature
    #y = data['o3']  # Assuming 'o3' is the target variable

    #Checks whether the variables are intercorrelated or not. 
    #Null hypothesis, p_value>0.05 so uncorrelated because it is an identity matrix. Otherwise, null hypothesis rejected and alternate hypothesis accepted
    def Bartlett_Test(self, corr_matrix, num):
        #self.data_set=data
        p = corr_matrix.shape[0]
        chi_square = -(num - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(corr_matrix))
        df = p * (p - 1) / 2
        self.p_value = 1 - chi2.cdf(chi_square, df)
        #return chi_square_value, p_value
        #chi_square, self.p_value = calculate_bartlett_sphericity(corr_matrix)
        print(f'chi square value: {chi_square}')
        # Has to be below significance level of 0.05 so null hypothesis is rejected
        print(f'p value: {self.p_value}') 
        
    #Measures sampling adequacy. 
    def KMO_Test(self, dataset):
        kmo_all, self.kmo_model = calculate_kmo(dataset)
        print(f'KMO model: {self.kmo_model}')
   
    #scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(X)
    
    def cov_matrix_func(self, gas):
        self.cov_matrix =  {}
        self.cov_matrix[gas] = np.cov(self.mod_data[gas], rowvar=False)
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.cov_matrix[gas], annot=True, cmap='coolwarm', fmt ='.2f')
        plt.title(f'Covariance matrix for target gas: {gas}')
        plt.show()
    
    def cal_PCA(self, data, target_gases):
        cov_matrix={}
        self.eigenvalues={}
        self.eigenvectors={}
        self.pc= {}
        self.mod_data={}
        self.size={}
        pc_num={}
        trans_data={}
        self.target_gases=target_gases
        for gas in target_gases:
            #target_data = data[gas]
            self.mod_data[gas] = data.drop(columns=gas)
            
            self.cov_matrix_func(gas)
    
            self.eigenvalues[gas], self.eigenvectors[gas] = np.linalg.eig(self.cov_matrix[gas])
            pc_num[gas] = self.cov_matrix[gas].shape[1]
            #print(self.eigenvectors[gas])
            #print(pc_num[gas])
            
            self.pc[gas] = self.eigenvectors[gas][:,:pc_num[gas]]
            #print(self.pc[gas].shape[0])
            #print(self.eigenvectors['co'].shape[0])
            
            trans_data[gas] = np.dot(self.mod_data[gas], self.pc[gas])
            self.size[gas] = self.eigenvalues[gas].shape[0]
        #print(f'shape of pc {pc.shape[0]}, {pc.shape[1]}\nshape of trans {trans.shape[0]}, {trans.shape[1]}')
        return trans_data
    
    def Cond_Num(self):
        cond_num = {}
        #If condition number > 5 : muliticollinearity
        for gas in self.target_gases:
            cond_num[gas] = max(self.eigenvalues[gas])/min(self.eigenvalues[gas])
            print(f'CN for {gas}: {cond_num[gas]}')
            
    
    def filter_PC(self, num):
        final_pc={}
        trans_data={}
        temp=self.eigenvectors
        final_pc=temp
        for gas in self.target_gases:
           mod_ev = self.eigenvalues[gas]
          
           for i in range(self.size[gas]-num):
               min_pc = np.argmin(mod_ev)
               
               final_pc[gas] = np.delete(final_pc[gas], min_pc, axis=0)
               mod_ev = np.delete(mod_ev, min_pc, axis=0)
               
          
           if num==0:
               final_pc=self.eigenvectors
           
           final_pc[gas] = self.eigenvectors[gas][:,:self.eigenvectors[gas].shape[1]]
           trans_data[gas] = np.dot(self.mod_data[gas], final_pc[gas].T)
          
        return trans_data
    
    def CumSum(self):
        cum_var = np.cumsum(self.explained_variance_ratio_)
        
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, len(cum_var)+1), cum_var, marker='o', linestyle='-')
        plt.title('Cumulative Explained Variance')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        plt.show()
        threshold = 0.90
        num_components = np.argmax(cum_var >= threshold) + 1
        print("Number of components to retain:", num_components)
        
    def Kaiser_Criterion(self, corr, target_gases):
        kaiser_comps={}
        for gas in target_gases:
            eigenvalues,_ = np.linalg.eig(corr[gas])
            kaiser_comps[gas] = np.sum(eigenvalues > 1)
            #print("Number of components to retain according to the Kaiser criterion:", Kaiser_comps)
        return kaiser_comps


        
    def Scree_Plot(self):
        #eigensum={}
        for gas in self.target_gases:
            self.evr=0
            eigensum = sum(self.eigenvalues[gas]) #sums all the eigenvalue
            #cum_var = np.cumsum(self.eigenvalues[gas])
            self.evr = self.eigenvalues[gas]/eigensum * 100 
          
            #print(f'EVR for {gas}: {self.evr}')
            cum_var = np.cumsum(self.evr)
            
            #print(f'for {gas} {cum_var}')
            fig, (ax1, ax2) = plt.subplots(2)
            #ax1.figure(figsize=(10, 6))
            ax1.plot(np.arange(1, len(self.evr) + 1), self.evr, marker='o', linestyle='-')
            ax1.set_ylim(0, 100)
            ax1.set_title(f'Scree Plot for {gas}')
            ax1.set_xlabel('Principal Component')
            ax1.set_ylabel('Eigenvalue')
            #plt.ylabel('Explained Variance Ratio (%)')
            #ax2.figure(figsize=(10, 6))
            ax2.plot(np.arange(1, len(self.evr) + 1), cum_var, marker='x', linestyle='-')
            #ax2.plot(np.arange(1, len(self.evr) + 1), )
            ax2.set_ylim(0, 120)
            ax2.set_title(f'Cumulative Explained Variance for {gas}')
            ax2.set_xlabel('Principal Component')
            ax2.set_ylabel('Proportion (%)')
            
            plt.tight_layout()
            plt.show()
            '''
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(1, len(cum_var) + 1), cum_var, marker='o', linestyle='-')
            plt.xlabel('Input Features')
            plt.ylabel('Cumulative V/ariance Ratio (%)')
            plt.show()
            '''
   


    
'''
    def PCA_Fn(self, data, target_gases, n):
        self.eigenvalues=[]
        self.evr_ratio=[]
        
        self.final_data={}
        for gas in target_gases:
            PC_cols=[]
            target_data= data[gas]
            mod_data=data.drop(columns=gas)
            for i in range(n[gas]):
                PC_cols.append('PC'+str(i+1))
            self.pca = PCA(n_components=n[gas]) 
            pca_data = self.pca.fit_transform(mod_data)
           
            X_pca_data = pd.DataFrame(data=pca_data, columns=PC_cols)
            print(X_pca_data)
            #exit(0)
            self.final_data[gas] = pd.concat([X_pca_data, target_data], axis=1)
            self.eigenvalues.append(self.pca.explained_variance_)
            self.evr_ratio.append(self.pca.explained_variance_ratio_)
            #print(f'eigenvalue for {gas}: {self.pca.explained_variance_}')
            #print(f'evr ratio for {gas}: {self.pca.explained_variance_ratio_}')
            vf = self.pca.components_
           # print(f'vf for {gas}: {vf}')
            #print(len(vf[0]))
            
            plt.figure(figsize=(10, 6))
            plt.title(f'{gas}')
            plt.scatter(self.final_data[gas]['PC1'], self.final_data[gas]['PC2'], c=self.final_data[gas][gas], cmap='viridis')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.colorbar(label='O3')
            plt.show()
            
              
    def Scree_lot(self):
        #self.explained_variance_ratio = self.pca.explained_variance_ratio_
        #print(self.explained_variance_ratio)
# Plot the scree plot
        print(self.eigenvalues)
        print(np.cumsum(self.eigenvalues[0]))
        exit(0)
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, len(self.explained_variance_ratio) + 1), self.explained_variance_ratio, marker='o', linestyle='-')
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.xticks(np.arange(1, len(self.explained_variance_ratio) + 1))
        plt.grid(True)
        plt.show()
'''       
    
'''
   
    def tabulate_stats(self, gas_data, prep):
        #myTable=None
        prep.add_rows('Stats', gas_data.columns)
        mean_values = np.array(prep.mean_vals)
        prep.add_rows('Mean', mean_values)
        std_values = np.array(prep.std_vals)
        prep.add_rows('Std_Dev', std_values)
        min_values = np.array(prep.min_vals)
        prep.add_rows('Min', min_values)
        print(prep.myTable)
        doc = Document()
        for row in prep.myTable:
            cells = row.get_string().split("\n")
            row = doc.add_table(rows=1, cols=len(cells))
            row_cells = row.rows[0].cells
            for i, cell in enumerate(cells):
                row_cells[i].text = cell
        doc.save("output.docx")
        exit(0)
        
        
        headers=[]
        headers.append('Stats')
        for i in range(len(self.gas_data.columns)):
            headers.append(self.gas_data.columns[i])
        myTable = PrettyTable([headers])
        mean_values = np.array(self.mean_vals)
        headers=[]
        headers.append('Mean')
        for i in range(len(mean_values)):
            headers.append(mean_values[i])
        print(headers)
        myTable.add_row([headers])
        print(myTable)
'''
       
