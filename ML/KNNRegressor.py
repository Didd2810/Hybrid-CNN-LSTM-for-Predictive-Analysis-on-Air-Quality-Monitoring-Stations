import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

class KNNR:
    def __init__(self):
        #self.knn =  None
        self.max_score_index={}
        self.size=20
        
    def model(self, x, y):
        scores = []
        dist_name = None
        for num in range(1,4):
            self.max_score_index[num] = {}
            scores = []
            for i in range(1, self.size):
                knn = KNeighborsRegressor(n_neighbors=i, p = num)
                score = cross_val_score(knn, x, y, cv=5, scoring="neg_mean_squared_error")
                scores.append(score.mean())
            if num==1:
                dist_name = 'Manhattan'
            elif num==2:
                dist_name = 'Euclidean'
            else: dist_name = 'Minkowski'
            plt.figure(figsize=(10,6))
            print(self.size)
            print(scores)
            plt.plot(range(1, self.size), scores, marker='x')
            plt.title(f'KNN scores value for n neighbours and dist_metric: {dist_name}')
            plt.show
            self.max_score_index[num] = np.argmax(scores)
        
    def pred_model(self, x_val, x_train, y_train):
        #print(self.max_score_index)
        max_key = max(self.max_score_index, key=self.max_score_index.get)
        max_val = self.max_score_index[max_key]
        knn = KNeighborsRegressor(n_neighbors=max_val, p=max_key)
        knn.fit(x_train, y_train)
        return knn.predict(x_val)
        
