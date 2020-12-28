
# https://stackoverflow.com/questions/42066352/python-regex-to-replace-all-single-word-characters-in-string/42066867


import pandas as pd
import numpy as np
import os
import re
import string
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec


def get_df_diagnosis_service_match():
    df_admissions = pd.read_csv(os.path.join('data', 'med_data', 'ADMISSIONS.csv.gz')).drop(columns=['ROW_ID'])
    df_services = pd.read_csv(os.path.join('data', 'med_data', 'SERVICES.csv.gz')).drop(columns=['ROW_ID'])
    
    df_diagnosis_service_match = pd.merge(df_admissions, df_services,  how='left', left_on=['SUBJECT_ID','HADM_ID'] ,right_on = ['SUBJECT_ID' , 'HADM_ID'])
    df_diagnosis_service_match = df_diagnosis_service_match[['DIAGNOSIS' , 'CURR_SERVICE']].dropna()
    
    
    #non-surgery = 0, surgery = 1
    Services = {"CMED" : 0,"CSURG":1,"DENT":0,"ENT":0,"GU":0,"GYN":0,"MED":0,"NB":0,"NBB":0,"NMED":0,
                "NSURG":1,"OBS":0,"ORTHO":1,"OMED":0,"PSURG":1,"PSYCH":0,"SURG":1,"TRAUM":1,"TSURG":1,"VSURG":1}
    
    
    
    df_diagnosis_service_match["Surgery"] = df_diagnosis_service_match["CURR_SERVICE"].map(Services)
    df_diagnosis_service_match = df_diagnosis_service_match.drop(columns=['CURR_SERVICE'])
    
    return df_diagnosis_service_match




def clean_text(doc):
    
    doc = str(doc)
    
    # re_alpha = re.compile('[^a-zA-Z]')
    # re_print = re.compile('[^%s]' % re.escape(string.printable))
    # re_punc  = re.compile('[%s]' % re.escape(string.punctuation))
    re_letters = re.compile('[^%s]' % re.escape(string.ascii_letters))
    re_slash = '/'
    re_one_letter_word = re.compile(r'\b[a-zA-Z]\b')
    re_large_gap = re.compile(r'\s+')

    
    doc = doc.replace(re_slash, '')
    doc = re_letters.sub(' ', doc)
    doc = re_one_letter_word.sub(' ', doc)
    doc = re_large_gap.sub(' ', doc)
    doc = doc.strip()
    return doc


def get_model(sentences):
    
    sequences = [s.split() for s in sentences]
    
    model = Word2Vec(sequences, min_count=3)
    return model


def save_embedding_matrix(tokenizer, sentences):
    
    model = get_model(sentences)
    
    # Save word embedding_matrix
    word_index = tokenizer.word_index
    # Creating X with word embeddings
    unique_words = len(word_index)
    total_words = unique_words + 1
    skipped_words = 0
    embedding_dim = 100
    embedding_matrix = np.zeros((total_words,embedding_dim))
    
    for word, index in tokenizer.word_index.items():
        try:
            embedding_vector = model[word]
            embedding_matrix[index] = embedding_vector
        except:
            skipped_words += 1
            pass
    
    
    np.save(os.path.join('data', 'save', "embedding_matrix_operative_service_predicition.npy"), embedding_matrix)

    return model, embedding_matrix


def from_tokens_to_vectors(X, num_words):
    result = np.zeros((len(X), num_words))
    for i,s in enumerate(X):
        for n in s:
            result[i,n] = 1
    
    return result
    




def get_data_splited():

    
    data = get_df_diagnosis_service_match()
    data.DIAGNOSIS = data.DIAGNOSIS.apply(clean_text)
    
    
    X = data.DIAGNOSIS.tolist()
    y = data.Surgery.tolist()
    
    X, y = shuffle(X,y)
    
    max_length = max([len(s.split()) for s in X])
    
    num_words=300
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(X)
    
    
    
    X = tokenizer.texts_to_sequences(X)
    
    X = from_tokens_to_vectors(X, num_words)
    
    # model, embedding_matrix = save_embedding_matrix(tokenizer, data.DIAGNOSIS.tolist())
    
    
    # X_train = pad_sequences(X_train, padding='post', maxlen=max_length)
    # X_test = pad_sequences(X_test, padding='post', maxlen=max_length)

    return X, y #, X_train, X_test, y_train, y_test #, model, embedding_matrix, data


# X, y, X_train, X_test, y_train, y_test = get_data_splited()







