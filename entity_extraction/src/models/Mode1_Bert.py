# Importing required library

import pandas as pd
import numpy as np
import tensorflow_hub as hub
from bert import tokenization
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score

# Load dataset from .csv file using pandas

dataframe=pd.read_csv(r'C:\Users\Lovely\PycharmProjects\Entity_Extraction_Invoice\entity_extraction\src\data\Required_Dataset3.csv',encoding='utf-8')
print(dataframe.head())

# Data Visualization :  This is in Visualize.py script

# groupby data by sentence id

data=dataframe.groupby(['Sentence'])['Data'].apply(' '.join)
for item in data.index:
    print(item)

print(data)
label=dataframe.groupby(['Sentence'])['Label'].apply(' '.join)
print(label[1])

print("hi before bert")

# Loading Bert pretrained model from tensorflow hub ( i have loaded this particular model as it is 37 MB around light weight)

module_url = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1'
bert_layer = hub.KerasLayer(module_url, trainable=True)

# Accessing vocab file and making it lower case and tokenizing it

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
print('******',vocab_file)
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
print(do_lower_case)
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
print(tokenizer)

#creating empty list  input and output data i.e data & label

all_tokens = []
all_masks = []
all_segments = []

all_tokens1 = []
all_masks1 = []
all_segments1 = []

def bert_encode(texts, tokenizer, max_len=512,type=data):

    text = texts.split(" ")

    text = text[:max_len - 2]
    input_sequence = ["[CLS]"] + text + ["[SEP]"]
    pad_len = max_len - len(input_sequence)

    tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
    pad_masks = [1] * len(input_sequence) + [0] * pad_len
    segment_ids = [0] * max_len
    if type==True:

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
        return np.array(all_tokens), np.array(all_masks) , np.array(all_segments)
    else:
        all_tokens1.append(tokens)
        all_masks1.append(pad_masks)
        all_segments1.append(segment_ids)
        return np.array(all_tokens1), np.array(all_masks1) ,np.array(all_segments)


# Calling bert_encoding function

max_len=100
for item in data.index:

    data_tokens  = bert_encode(data[item], tokenizer, max_len=max_len,type=True)
print(data_tokens)

for item in data.index:
    label_tokens = bert_encode(label[item], tokenizer, max_len=max_len,type=False)
print(label_tokens)

print("end")

def build_model(bert_layer, max_len):

    seq_length = max_len
    input_word_ids = tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32)
    input_mask = tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32)
    input_type_ids = tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32)

    encoder_inputs = dict(input_word_ids = tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32),
    input_mask = tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32),
    input_type_ids = tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32))

    print(bert_layer(encoder_inputs))

    bert_output = bert_layer(encoder_inputs)['sequence_output'][:,0,:]
    print(bert_output)
    input = tf.keras.layers.Dense(64, activation='relu')(bert_output)
    input = tf.keras.layers.Dropout(0.2)(input)
    input = tf.keras.layers.Dense(32, activation='relu')(input)
    input = tf.keras.layers.Dropout(0.3)(input)
    out = tf.keras.layers.Dense(100, activation='softmax')(input)
    print(out)
    model = tf.keras.models.Model(inputs=encoder_inputs, outputs= out)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='categorical_crossentropy', metrics=[tf.keras.metrics.Accuracy()])

    return model

model = build_model(bert_layer,100)
model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2, verbose=1)
print("befor train")
train_history = model.fit(
    data_tokens,label_tokens,
    epochs=5,
    callbacks=[checkpoint, earlystopping],
    validation_split=0.5,
    verbose=1)


# Predicting on test data

test_pred = model.predict([data_tokens[0][1],data_tokens[1][1],data_tokens[2][1]])
print(test_pred)
print(label_tokens)
test_pred=np.argmax(test_pred,axis=-1)

# Model validation

# Checking accuracy score, F1 score, Recall,Precision
acc=accuracy_score(label_tokens[0][1],test_pred)
print(acc)
F1_score=f1_score(label_tokens[0][1],test_pred,average='weighted')
precision=precision_score(label_tokens[0][1],test_pred,average='weighted')
recall=recall_score(label_tokens[0][1],test_pred,average='weighted',zero_division='warn')
print("Recall:",recall)




