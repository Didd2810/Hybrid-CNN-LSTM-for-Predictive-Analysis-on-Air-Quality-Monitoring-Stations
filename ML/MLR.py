import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import aqi
from LinearPreProcessing import PreProcessor
from PCATrial import PCA_Test
#from AQICalculation import AQI_calc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from Metrics import Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

class MLR:
    def Model(self, X_train, y_train, )