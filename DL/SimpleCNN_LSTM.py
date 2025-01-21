import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, MaxPooling1D, Flatten, Dropout, Bidirectional, GRU, Lambda
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class Simple_CNN_LSTM:
    def __init__(self):
        self.x_test=None
        self.x_train=None
        self.y_train=None
        self.gas = None
 
    def model_add(self, x_train, y_train, x_test, y_test):
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
        self.model = Sequential()
        
        self.model.add(TimeDistributed(Conv1D(filters=128, kernel_size=1, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)), input_shape=(1, self.x_train.shape[2], self.x_train.shape[3])))
        
        #model_cnn_lstm.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation=cnn_activations), input_shape=(None, X_train_series_sub.shape[2], X_train_series_sub.shape[3])))
        #self.model.add(TimeDistributed(Conv1D(filters=128, kernel_size=1, activation=self.cnn_activations, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)), input_shape=(1, self.x_train.shape[2], self.x_train.shape[3])))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
        self.model.add(TimeDistributed(Flatten()))
        #expand_dims_layer = Lambda(lambda x: tf.expand_dims(x, axis=1))
        #self.model.add(LSTM(50, activation='relu'))
        
        for _ in range(1):
            self.model.add(Bidirectional(LSTM(1000, activation='relu')))
            self.model.add(Dropout(0.5))
            #self.model.add(Lambda(lambda x: tf.expand_dims(x, axis=-1)))
        #self.model = expand_dims_layer
        #self.model.add(Dense(50))
        self.model.add(Dense(1))
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
        lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
        #sgd = optimizers.SGD(learning_rate=0.01, decay=1e-5, momentum=0.9, nesterov=True) 
        #opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        def rmse(y_true, y_pred):
            return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
        self.model.compile(loss=rmse, optimizer='adam', metrics=[rmse])
        #early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        # Define list of forecasting horizons
        '''
        horizons = [1, 3, 6, 9, 12]  # Example horizons
        rmse_scores = []

        # Perform rolling origin cross-validation for each horizon
        for horizon in horizons:
            print(f"Evaluating for horizon: {horizon}")
            scores = self.rolling_origin_cv(self.model, horizon)
            avg_score = np.mean(scores)
            rmse_scores.append(avg_score)

        # Plot RMSE scores vs. forecasting horizons
        plt.plot(horizons, rmse_scores, marker='o')  
        plt.title(f'RMSE vs. Forecasting Horizon for {self.gas}')
        plt.xlabel('Forecasting Horizon')
        plt.ylabel('RMSE')
        plt.grid(True)
        plt.show()

        # Train the model on the full training data with the best horizon
        best_horizon = horizons[np.argmin(rmse_scores)]
        print(f"Best forecasting horizon: {best_horizon}")
        '''
        history = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=8, verbose=2, callbacks=[lr_callback, early_stopping_callback])
        #self.plot_history(history)
        #return self.model.summary()
    
    def predict(self, x_test):
        return self.model.predict(x_test)
    
    def rolling_origin_cv(self, model, horizon):
        scores = []
        n = len(self.y_train)
        for i in range(n - horizon):
            train_X, train_y = self.x_train[:i+1], self.y_train[:i+1]
            test_X, test_y = self.x_train[i+1:i+1+horizon], self.y_train[i+1:i+1+horizon]
            model.fit(train_X, train_y)
            pred_y = model.predict(test_X)
            score = np.sqrt(np.mean(np.square(test_y - pred_y)))
            scores.append(score)
        return scores
    
    def add_gas(self, gas):
        self.gas = gas
        
    def plot_history(self, history):
        

        plt.plot(history.history['loss'], marker='o') 
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], marker='s')
        plt.title(f'Model loss for {self.gas}')
        plt.grid(True)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()
    