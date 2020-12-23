

# source: https://mimic.physionet.org/mimictables/chartevents/
# https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/
# https://stackoverflow.com/questions/46308374/what-is-validation-data-used-for-in-a-keras-sequential-model
# https://keras.io/api/models/sequential/
# https://keras.io/api/layers/core_layers/embedding/


import pandas as pd
import numpy as np
import os
import gc

within_n_days = 30

data_tables = sorted(os.listdir('data'))

df_admission = pd.read_csv(os.path.join('data', data_tables[0]))

df_admission.columns
df_admission.shape

# subject_id for returning subjects (with at least 2 admissions)
df_subjects_with_multiple_admission = pd.DataFrame(np.array(sorted(df_admission.SUBJECT_ID.unique()))[df_admission.sort_values(by=['SUBJECT_ID']).groupby(by=['SUBJECT_ID'])['HADM_ID'].count() > 1], columns=['SUBJECT_ID'])

# data for returning subjects (with at least 2 admissions)
df_admission_with_returning_subjects = pd.merge(df_admission, df_subjects_with_multiple_admission, how='right', on='SUBJECT_ID')
# reset index to use to obtain next to last admission indices
df_admission_with_returning_subjects = df_admission_with_returning_subjects.reset_index()

# data for the last admissions of the returning subjects
df_last_admisson_for_returning_subjects = df_admission_with_returning_subjects.groupby(by='SUBJECT_ID').last().reset_index()
# indices for next to last admission on table df_admission_with_returning_subjects
indices_for_next_to_last_admission = df_last_admisson_for_returning_subjects['index'].apply(lambda x: x-1).tolist()

# dictionary maps SUBJECT_ID to HADM_ID
dict_last_admission_hamdid_by_subject_id = {}
list_of_tuples_of_SUBJECT_ID_and_HADM_ID_for_last_admission = list(df_last_admisson_for_returning_subjects[['SUBJECT_ID', 'HADM_ID']].to_records(index=False))
for SUBJECT_ID, HADM_ID in list_of_tuples_of_SUBJECT_ID_and_HADM_ID_for_last_admission: 
    dict_last_admission_hamdid_by_subject_id[SUBJECT_ID] = HADM_ID



# dataframe for the next to last admissions of the returning subjects
df_one_before_last_admisson_for_returning_subjects  = df_admission_with_returning_subjects[df_admission_with_returning_subjects.index.isin(indices_for_next_to_last_admission)].reset_index(drop=True)
df_one_before_last_admisson_for_returning_subjects = df_one_before_last_admisson_for_returning_subjects[['SUBJECT_ID', 'HADM_ID', 'DISCHTIME']]

# get the subjects whose last admission was urgent or emergant
emergency_urgent_mask = (df_last_admisson_for_returning_subjects.ADMISSION_TYPE == 'EMERGENCY') | (df_last_admisson_for_returning_subjects.ADMISSION_TYPE == 'URGENT')
df_returning_subject_whose_last_admission_is_emergant_urgent = df_last_admisson_for_returning_subjects[emergency_urgent_mask][['SUBJECT_ID', 'HADM_ID', 'ADMITTIME']]

# merge last admission and next to last discharge dates for those whose last admission was urgent or emergant
df_returning_subject_whose_last_admission_is_emergant_urgent_last_admit_and_next_to_last_discharge_time = pd.merge(df_returning_subject_whose_last_admission_is_emergant_urgent, df_one_before_last_admisson_for_returning_subjects, how='left', on=['SUBJECT_ID'])

df_returning_subject_whose_last_admission_is_emergant_urgent_last_admit_and_next_to_last_discharge_time.dtypes

# Convert date string to datetime
df_returning_subject_whose_last_admission_is_emergant_urgent_last_admit_and_next_to_last_discharge_time.ADMITTIME = pd.to_datetime(df_returning_subject_whose_last_admission_is_emergant_urgent_last_admit_and_next_to_last_discharge_time['ADMITTIME'])
df_returning_subject_whose_last_admission_is_emergant_urgent_last_admit_and_next_to_last_discharge_time.DISCHTIME = pd.to_datetime(df_returning_subject_whose_last_admission_is_emergant_urgent_last_admit_and_next_to_last_discharge_time['DISCHTIME'])

difference_between_last_2_admissions = df_returning_subject_whose_last_admission_is_emergant_urgent_last_admit_and_next_to_last_discharge_time.ADMITTIME - df_returning_subject_whose_last_admission_is_emergant_urgent_last_admit_and_next_to_last_discharge_time.DISCHTIME
is_urgent_admission_within_30_days = (difference_between_last_2_admissions <= pd.Timedelta(within_n_days, unit='D'))

# positive subject ids (the ones who paid his/her last visited as urgent/emergant within 30 days time frame from the previous one)
positive_subject_ids = df_returning_subject_whose_last_admission_is_emergant_urgent_last_admit_and_next_to_last_discharge_time[is_urgent_admission_within_30_days].SUBJECT_ID.tolist()



# Save positive_subject_ids and dict_last_admission_hamdid_by_subject_id
import pickle
with open(os.path.join('data', "positive_subject_ids_visited_in_30_days.txt"), "wb") as _fp:
    pickle.dump(positive_subject_ids, _fp)
    
with open(os.path.join('data', "dict_last_admission_hamdid_by_subject_id.txt"), "wb") as _fp:
    pickle.dump(dict_last_admission_hamdid_by_subject_id, _fp)


# Load positive_subject_ids and dict_last_admission_hamdid_by_subject_id
with open(os.path.join('data', "positive_subject_ids_visited_in_30_days.txt"), "rb") as _fp:
    positive_subject_ids = pickle.load(_fp)

with open(os.path.join('data', "dict_last_admission_hamdid_by_subject_id.txt"), "rb") as _fp:
    dict_last_admission_hamdid_by_subject_id = pickle.load(_fp)








df_noteevents = pd.read_csv(os.path.join('data', data_tables[19]))
df_admission = pd.read_csv(os.path.join('data', data_tables[0]))[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME']]
df_admission = df_admission.sort_values(by=['SUBJECT_ID', 'ADMITTIME']).reset_index(drop=True)[['SUBJECT_ID', 'HADM_ID']]

df_noteevents = pd.merge(df_admission, df_noteevents, how='left', on=['SUBJECT_ID', 'HADM_ID'])



df_noteevents.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ROW_ID'].count().mean()


# get an empty data frame which consist of only all Subject IDs
df_subjects_data = pd.DataFrame(df_noteevents.SUBJECT_ID.unique(), columns=['SUBJECT_ID']).sort_values(by=['SUBJECT_ID']).reset_index(drop=True)
# df_subjects_data_tiny = df_subjects_data[:100] 


# merge TEXT data for each SUBJECT_ID
def get_merged_text_for_each_subject_admission_combination(cr_SUBJECT_ID):
    mask1 = (df_noteevents.SUBJECT_ID == cr_SUBJECT_ID) 
    if cr_SUBJECT_ID in dict_last_admission_hamdid_by_subject_id.keys():
        mask2 = df_noteevents.HADM_ID != dict_last_admission_hamdid_by_subject_id[cr_SUBJECT_ID]
    else:
        mask2 = True
    text_merged = " ".join([str(e) for e in df_noteevents[mask1 & mask2].TEXT.tolist()])
    return text_merged

df_subjects_data['merged_text'] = df_subjects_data.SUBJECT_ID.apply(get_merged_text_for_each_subject_admission_combination)


# Save TEXT under the name of 'df_subjects_data_for_urgent_visit_in_30_days_prediction'
# df_subjects_data.to_csv(os.path.join('data', 'df_subjects_data_for_urgent_visit_in_30_days_prediction.csv'), index=False)




# CHECK-POINT
# LOAD df_subjects_data and continue from here on
df_subjects_data = pd.read_csv(os.path.join('data', 'df_subjects_data_for_urgent_visit_in_30_days_prediction.csv.tar.gz'))
df_subjects_data = df_subjects_data.rename(columns={'df_subjects_data_for_urgent_visit_in_30_days_prediction.csv': 'SUBJECT_ID'})

from datetime import datetime
import string
import re
# from keras.preprocessing.text import text_to_word_sequence
# from gensim.models import Word2Vec
# from nltk.tokenize import sent_tokenize, word_tokenize, ToktokTokenizer
# import nltk



def clean_text_and_sequence(text, min_sequance_element_length=1):
    text = str(text)
    re_punc = re.compile('[^%s]' % re.escape(string.printable))
    re_print = re.compile('[%s]' % re.escape(string.punctuation))
    # words = [re_punc.sub('',w) for w in words]
    # words = [re_print.sub('',w) for w in words]
    try:
        text = re_punc.sub('',text)
    except:
        print('EXCEPTION  !!! re_punc.sub('',text): ', text, type(text))
    text = re_print.sub('',text)
    
    words = re.split(' ', text)
    words = [w for w in words if w.isalpha()]
    words = [w.lower() for w in words]
    if min_sequance_element_length > 1:
        words = [x for x in words if len(x) >= min_sequance_element_length]
    
    words = list(set(words))
    
    return words



def get_sequence_list(text_list, min_sequance_element_length=1):
    global sentences
    for i,text in enumerate(text_list):
        if i % 1000 == 0:
            print("Semple No:", i)
            print(datetime.now())
        cr_sequence = clean_text_and_sequence(text, min_sequance_element_length)
        sentences.append(cr_sequence)
        gc.collect()


raw_text_data = df_subjects_data.merged_text.tolist()
del df_subjects_data




# Save sequenced_sentences
import pickle
with open(os.path.join('data', "raw_text_data_30_days.txt"), "wb") as _fp:
    pickle.dump(raw_text_data, _fp)


# Load positive_subject_ids and dict_last_admission_hamdid_by_subject_id
with open(os.path.join('data', "raw_text_data_30_days.txt"), "rb") as _fp:
    raw_text_data = pickle.load(_fp)





sentences = []
get_sequence_list(raw_text_data, 3)





# Save sequenced_sentences
import pickle
with open(os.path.join('data', 'save', "sequenced_sentences_all.txt"), "wb") as _fp:
    pickle.dump(sentences, _fp)


# Load positive_subject_ids and dict_last_admission_hamdid_by_subject_id
with open(os.path.join('data', 'save', "sequenced_sentences_all.txt"), "rb") as _fp:
    sentences = pickle.load(_fp)
    gc.collect()


# TRAIN WORD2VEC MODEL


from gensim.models import Word2Vec

model = Word2Vec(sentences, min_count=3)

print(model)

# access vector for one word
print(model['nurse'])

model.save(os.path.join('models', 'word2vec_model.bin'))

model = Word2Vec.load(os.path.join('models', 'word2vec_model.bin'))



# Create X and y
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=6000)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
word_index = tokenizer.word_index

from collections import Counter

dict_word = Counter()
for wlist in sentences:
    dict_word.update(wlist)

popular_words = [w for i,(w,c) in enumerate(dict_word.most_common(6000)) if len(w) > 3 and i > 60]

# Save encoded_sentences
import pickle
with open(os.path.join('data', 'save', "popular_words.txt"), "wb") as _fp:
    pickle.dump(popular_words, _fp)


# Load encoded_sentences
with open(os.path.join('data', 'save', "popular_words.txt"), "rb") as _fp:
    popular_words = pickle.load(_fp)
    gc.collect()


sentences = [[w for w in s if w in popular_words] for s in sentences]


# Save encoded_sentences
import pickle
with open(os.path.join('data', 'save', "sequences.txt"), "wb") as _fp:
    pickle.dump(sequences, _fp)


# Load encoded_sentences
with open(os.path.join('data', 'save', "sequences.txt"), "rb") as _fp:
    sequences = pickle.load(_fp)
    gc.collect()


# Padding
from keras.preprocessing.sequence import pad_sequences

max_length = max([len(s) for s in sequences])
X = pad_sequences(sequences, maxlen=max_length, padding='post')

np.save(os.path.join('data', 'save', "X.npy"), X)
X = np.load(os.path.join('data', 'save', "X.npy"))


# Creating X with word embeddings
unique_words = len(word_index)
total_words = unique_words + 1
skipped_words = 0
embedding_dim = 100
embedding_matrix = np.zeros((total_words,embedding_dim))

for word, index in tokenizer.word_index.items():
    try:
        embedding_vector = model[word]
    except:
        skipped_words += 1
        pass
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


np.save(os.path.join('data', 'save', "embedding_matrix.npy"), embedding_matrix)
embedding_matrix = np.load(os.path.join('data', 'save', "embedding_matrix.npy"))
embedding_matrix.shape



# Create target variable
df_subjects_data['y'] = df_subjects_data.SUBJECT_ID.apply(lambda x: 1 if x in positive_subject_ids else 0)
df_subjects_data.y.value_counts()
# positive_sample_ratio:
positive_sample_ratio = round(df_subjects_data.y.value_counts()[1]/df_subjects_data.shape[0], 5)
y = df_subjects_data['y'].to_list()
y = np.array(y, dtype=np.int32)

np.save(os.path.join('data', 'save', "y.npy"), y)
y = np.load(os.path.join('data', 'save', "y.npy"))





#from datetime import datetime

# def get_admission_date_from_note_text(text):
#     date_text = (text.split('ssion Date:  [**')[1]).split('**]')[0]
#     return datetime.strptime(date_text, '%Y-%m-%d')


#df_noteevents.TEXT.apply(get_admission_date_from_note_text)

