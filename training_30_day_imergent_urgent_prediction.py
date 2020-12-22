

# source: https://mimic.physionet.org/mimictables/chartevents/
# https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/
# https://stackoverflow.com/questions/46308374/what-is-validation-data-used-for-in-a-keras-sequential-model
# https://keras.io/api/models/sequential/
# https://keras.io/api/layers/core_layers/embedding/


import numpy as np
import os
import gc

X = np.load(os.path.join('data', 'save', "X.npy"))
y = np.load(os.path.join('data', 'save', "y.npy"))

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)

embedding_matrix = np.load(os.path.join('data', 'save', "embedding_matrix.npy"))





train_ratio = 0.65
validation_ratio = 0.25

X_train, y_train = X[:int(len(X)*(train_ratio))], y[:int(len(y)*(train_ratio))]
X_val, y_val = X[int(len(X)*(train_ratio)):int(len(X)*(train_ratio+validation_ratio))], y[int(len(y)*(train_ratio)):int(len(y)*(train_ratio+validation_ratio))]
X_test, y_test = X[int(len(X)*(train_ratio+validation_ratio)):], y[int(len(y)*(train_ratio+validation_ratio)):]

del X
gc.collect()
del y
gc.collect()


# Check if stratified
sum(y_train)/train_ratio
sum(y_val)/validation_ratio
sum(y_test)/(1-(train_ratio+validation_ratio))


import pandas as pd
from sklearn.utils import resample

def down_sample_training_set(X_train, y_train, target_feature='y', mode='downsample', size=-1):
    data = pd.DataFrame(X_train)
    data[target_feature] = pd.Series(y_train)
    # shuffle
    data = data.sample(frac=1).reset_index(drop=True)
    
    # Entries of the both minority and majority classes
    value_majority = data[target_feature].value_counts().sort_values(ascending=False).index[0]
    data_majority = data.loc[data[target_feature] == value_majority]
    data_minority = data.loc[data[target_feature] != value_majority]
    
    
    print("data_majority: {0} @ data_minority: {1}".format(len(data_majority), len(data_minority)))
    
    if mode == 'downsample':
        #filters the majority samples down to the size of minority portion
        data_majority_down_sampled = resample(data_majority, 
                                         replace=True,
                                         n_samples=len(data_minority),
                                         random_state=142)
        
        # Combine majority class with upsampled minority class
        data_up_sampled = pd.concat([data_majority_down_sampled, data_minority])
    
    elif mode == 'upsample':
        #populates the minority portion of the samples up to the size of majority portion
        data_minority_up_sampled = resample(data_minority, 
                                         replace=True,
                                         n_samples=len(data_majority),
                                         random_state=142)
        
        # Combine majority class with upsampled minority class
        data_up_sampled = pd.concat([data_majority, data_minority_up_sampled])
    
    elif size != -1:
        #populates the minority portion of the samples up to the size of majority portion
        data_minority_up_sampled = resample(data_minority, 
                                         replace=True,
                                         n_samples=size,
                                         random_state=142)
        
        #filters the majority samples down to the size of minority portion
        data_majority_down_sampled = resample(data_majority, 
                                         replace=True,
                                         n_samples=size,
                                         random_state=142)
        
        
        # Combine majority class with upsampled minority class
        data_up_sampled = pd.concat([data_majority_down_sampled, data_minority_up_sampled])
        
    
    # Display new class counts
    print(data_up_sampled[target_feature].value_counts())
    
    X_up_sampled = np.array(data_up_sampled.drop([target_feature], 1).astype(np.int32))
    y_up_sampled = np.array(data_up_sampled[target_feature]).astype(np.int32)
    
    
    print("X_up_sampled: ",  len(X_up_sampled), "  y_up_sampled: ",  len(y_up_sampled))
    
    return X_up_sampled, y_up_sampled




from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score

def print_evaluation_metrics_for_the_test_set(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = np.vectorize(round) (y_pred)
    
    print("Accuracy Score: {0}".format(accuracy_score(y_test, y_pred)))
    print("Precision Score: {0}".format(precision_score(y_test, y_pred)))
    print("Recall Score: {0}".format(recall_score(y_test, y_pred)))
    print("F-Beta Score: {0}".format(fbeta_score(y_test, y_pred, beta=2.0)))





# X_train, y_train = down_sample_training_set(X_train, y_train)
X_train, y_train = down_sample_training_set(X_train, y_train, mode='midsample', size=2000)
X_val, y_val = down_sample_training_set(X_val, y_val, mode='midsample', size=1000)


# Training And Evaluation of The Model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D



total_words = embedding_matrix.shape[0]
embedding_dim = embedding_matrix.shape[1]
max_length = X_train.shape[1]


# define model
model = Sequential()
model.add(Embedding(total_words, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
# model.fit(X_train, y_train, epochs=2, verbose=2)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=2)
# evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
# print('Test Accuracy: %f' % (acc))
print_evaluation_metrics_for_the_test_set(model, X_test, y_test)







