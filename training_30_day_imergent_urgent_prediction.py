

# source: https://mimic.physionet.org/mimictables/chartevents/
# https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/
# https://stackoverflow.com/questions/46308374/what-is-validation-data-used-for-in-a-keras-sequential-model
# https://keras.io/api/models/sequential/
# https://keras.io/api/layers/core_layers/embedding/
# https://machinelearningmastery.com/how-to-avoid-exploding-gradients-in-neural-networks-with-gradient-clipping/
# http://mipal.snu.ac.kr/images/1/16/Dropout_ACCV2016.pdf



import numpy as np
import os
import gc

X = np.load(os.path.join('data', 'save', "X.npy"))
y = np.load(os.path.join('data', 'save', "y.npy"))


embedding_matrix = np.load(os.path.join('data', 'save', "embedding_matrix.npy"))


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
embedding_matrix = scaler.fit_transform(embedding_matrix)



# shuffle
from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=142)



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
from sklearn.utils import shuffle

def down_sample_training_set(X, y, target_feature='y', mode='downsample', size=-1):
    
    X, y = shuffle(X, y, random_state=142)
    
    data = pd.DataFrame(X)
    data[target_feature] = pd.Series(y)
    
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
    
    X_up_sampled, y_up_sampled = shuffle(X_up_sampled, y_up_sampled, random_state=142)
    
    return X_up_sampled, y_up_sampled



from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score

def print_evaluation_metrics_for_the_test_set(model, X_test, y_test, threshold=0.5, plot=True):
    
    if threshold == 0.5:
        threshold_fun = round
    else:
        threshold_fun = lambda x: 1 if x > threshold else 0
    
    y_pred = model.predict(X_test)
    y_pred = np.vectorize(threshold_fun) (y_pred)
    
    _accuracy_score = accuracy_score(y_test, y_pred)
    _precision_score = precision_score(y_test, y_pred)
    _recall_score = recall_score(y_test, y_pred)
    _fbeta_score = fbeta_score(y_test, y_pred, beta=2.0)
    
    print("Accuracy Score: {0}".format(_accuracy_score))
    print("Precision Score: {0}".format(_precision_score))
    print("Recall Score: {0}".format(_recall_score))
    print("F-Beta Score: {0}".format(_fbeta_score))
    
    scores = (_accuracy_score, _precision_score, _recall_score, _fbeta_score)
    
    if plot:
        plot_confusion_matrix_for_the_model_results(y_pred, y_test, scores)


from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

def plot_confusion_matrix_for_the_model_results(y_pred, y_test, scores):
    
    # label_names=["normal", "urgent"]
    labels=[0, 1]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(cm)
    ax = plt.axes()
    sns.heatmap(cm, annot=True, cmap=plt.cm.Blues, fmt='d', ax=ax)
    ax.set_title("acc: {:.3f},  prec: {:.3f},  recall: {:.3f},  f-beta: {:.3f}\n".format(*scores))
    plt.show()


def plot_model_training_history(history):  
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


X_train, y_train = down_sample_training_set(X_train, y_train, mode='upsample')
# X_train, y_train = down_sample_training_set(X_train, y_train, mode='midsample', size=5000)
X_val, y_val = down_sample_training_set(X_val, y_val, mode='upsample')
# X_train, y_train = down_sample_training_set(X_train, y_train)
# X_val, y_val = down_sample_training_set(X_val, y_val)


# Training And Evaluation of The Model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import metrics



def execute_training():
    total_words = embedding_matrix.shape[0]
    embedding_dim = embedding_matrix.shape[1]
    max_length = X_train.shape[1]
    
    
    # define model
    model = Sequential()
    model.add(Embedding(total_words, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False))
    model.add(Dropout(rate=0.85))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(Dropout(rate=0.1))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    # compile network
    opt = Adam(learning_rate=0.0001)
    # # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[metrics.BinaryCrossentropy()])
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    # fit network
    callbacks = [EarlyStopping(monitor='val_loss')]
    # model.fit(X_train, y_train, epochs=2, verbose=2)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=8, verbose=2
              # , callbacks=callbacks
              )
    # evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    # print('Test Accuracy: %f' % (acc))
    
    
    print_evaluation_metrics_for_the_test_set(model, X_test, y_test, threshold=0.5, plot=True)
    
    plot_model_training_history(history)


execute_training()




# learning rate
# Dropout
# epoch size
# poolsize
# scaling matrix
# shuffle X, y before splitting



