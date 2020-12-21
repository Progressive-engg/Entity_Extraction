import pandas as pd
import numpy as np
import re
from tensorflow import keras
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,Model
from keras.layers import Embedding,Dense,LSTM,Input,Bidirectional,Dropout,TimeDistributed
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score

def main():

    data1=pd.read_csv(r'C:\Users\Lovely\PycharmProjects\Entity_Extraction_Invoice\entity_extraction\src\data\Required_Dataset3.csv',encoding='utf-8')
    data1.head(40)

    unique_word=set(data1['Data'])
    #print(len(unique_word))

    unique_word_index={w:i+2  for i,w in enumerate(unique_word)}

    print(len(unique_word_index))
    #print(unique_word_index)

    unique_index_word={i+2:w  for i,w in enumerate(unique_word)}
    #print(unique_index_word)


    unique_tag=set(data1['Label'])
    #print(len(unique_tag))

    unique_tag_index={w:i+2 for i,w in enumerate(unique_tag)}
    unique_index_tag={i+2:w for i,w in enumerate(unique_tag)}
    #print(unique_index_tag)



    def clean(sent):
        lis = []
        z = [item for item in sent['Data']]
        u = [item for item in sent['Label']]
        combined_data = zip(z, u)
        # print(z)
        # print(u)
        for item in combined_data:
            # print(item)
            lis.append(item)
        return lis

    sentence=data1.groupby(['Sentence'],sort=False).apply(clean)

    print(sentence)

    def word_to_number(x):
        lis=[]
        for i in x :
            lis.append(unique_word_index[i[0]])
        return lis

    input_sentence=sentence.apply(word_to_number)


    lis=[]
    for i in range(len(input_sentence)):
        lis.append(input_sentence[i+1])

    input_lis = []


    def tag_to_number(x):
        lis = []
        for i in x:
            # print(i[1])

            lis.append(unique_tag_index[i[1]])

        return lis

    output_sentence=sentence.apply(tag_to_number)

    lis1=[]
    for i in range(len(output_sentence)):
        lis1.append(output_sentence[i+1])



    maxlen_input=100
    input_pad=pad_sequences(lis,maxlen=maxlen_input,padding='post',value=0)
    len(input_pad[0])

    output_pad=pad_sequences(lis1,maxlen=maxlen_input,padding='post',value=1)

    len(output_pad[0])
    vocab_size=len(unique_word_index)+2

    # Model building start 
    
    inputt=Input(shape=(100,))
    emb=Embedding(vocab_size,100,input_length=100)
    emb=emb(inputt)
    emb=Dropout(0.1)(emb)
    lstm=Bidirectional(LSTM(units=100,return_sequences=True,dropout=0.1,activation='relu'))
    out=lstm(emb)
    output=TimeDistributed(Dense(units=13,activation='softmax'))(out)
    model=Model(inputt,output)

    model.summary()

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])


    output_pad=[to_categorical(i,num_classes=13) for i in output_pad]

    # train test split of data
    
    x_train,x_test,y_train,y_test=train_test_split(input_pad,output_pad,test_size=0.2,random_state=0)
    print(x_test.shape)
    print(x_test)

    # fitting the model
    
    model.fit(x_train,np.array(y_train),batch_size=128,epochs=5,validation_data=(x_test,np.array(y_test)))
    y_pred=model.predict(np.array([x_test[1]]))
    np.array([x_test[1]])

    y_pred=np.argmax(y_pred,axis=-1)

    lis = list(y_pred[0])
    for i in range(len(lis)):
        print('word: {} : {}'.format(unique_index_word[x_test[1][i]], unique_index_tag[lis[i]]))

    # Checking accuracy score, F1 score, Recall,Precision

    y_pred=[to_categorical(i,num_classes=13) for i in y_pred]
    acc=accuracy_score(y_test[0],y_pred[0])
    print(acc)
    F1_score=f1_score(y_test[0],y_pred[0],average='weighted')
    precision=precision_score(y_test[0],y_pred[0],average='weighted')
    recall=recall_score(y_test[0],y_pred[0],average='weighted',zero_division='warn')
    print("Recall:",recall)


if __name__ == '__main__':
    print("starting execution")
    main()

