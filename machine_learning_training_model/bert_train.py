import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM,Bidirectional,Input,Dense,Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
from tensorflow.keras.models import Sequential,Model,save_model,load_model
import transformers
from tqdm import tqdm
import re
from tokenizers import BertWordPieceTokenizer
from sklearn.model_selection import StratifiedShuffleSplit
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d','--data',required = True,
                    help = "path to training data")
parser.add_argument('-v','--val',required = False,
                    help = "path to validation data")
parser.add_argument('-s','--savePath',required = True,
                    help = "path to saving directory")

args = vars(parser.parse_args())

''' TPU CONFIGURATION TO REDUCE TRAINING TIME'''

# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


'''basic preproceesing'''
def replace_contraction(text):
    contraction_patterns = [ (r'won\'t', 'will not'), (r'can\'t', 'can not'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),
                             (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), (r'dont', 'do not'), (r'wont', 'will not'),
                             (r'don\'t', 'do not')]
    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
    for (pattern, repl) in patterns:
        (text, count) = re.subn(pattern, repl, text)
    return text

def replace_links(text, filler=' '):
        text = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*',
                      filler, text).strip()
        return text
        
def remove_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def cleanText(text):
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = replace_contraction(text)
    text = replace_links(text, "link")
    text = remove_numbers(text)
    text = re.sub(r'[,!@#$%^&*)(|/><";:.?\'\\}{`]',"",text)
    text = text.lower()
    return text



'''TRAINING CONFIGURATION'''
AUTO = tf.data.experimental.AUTOTUNE

#configuration
EPOCHS = 10
BATCH_SIZE = 16*strategy.num_replicas_in_sync
MAX_LEN = 192

'''Tkenization of sentences in train_data and val_data'''
#creating tokenization step of sentence for bert 
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=192):
    """
    Encoder for encoding the text into sequence of integers for BERT Input
    """
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(length = maxlen)
    all_ids = []
    
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)

def tokenize_sentence(train_data,val_path,max_len):
    # First load the real tokenizer
    tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    # Save the loaded tokenizer locally
    tokenizer.save_pretrained('.')
    # Reload it with the huggingface tokenizers library
    fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
    x_train = fast_encode(train_data.comment_text.astype(str), fast_tokenizer, maxlen = max_len)

    if(len(val_path) != 0):
        val_data = pd.read_csv(val_path)
        x_test = fast_encode(val_data.comment_text.astype(str),fast_tokenizer, maxlen = max_len)
    else:
        x_test = None

    return x_train,x_test

def split_data(x_train,y_train):
    stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index,test_index in stratSplit.split(x_train,y_train):
        x_train,x_val = x_train[train_index],x_train[test_index]
        y_train,y_val = y_train[train_index],y_train[test_index]

    return x_train,x_val,y_train,y_val

'''DATASET LOADER FOR TENSORFLOW MODEL'''

def tf_data_loader(x_train,x_val,y_train,y_val):
    train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
    )

    valid_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_val, y_val))
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(AUTO)
    )

    return train_dataset,valid_dataset

def build_model(transformer, max_len=512):
    """
    function for training the BERT model
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    # lstm_1 = Bidirectional(LSTM(100, dropout=0.3, recurrent_dropout=0.3,return_sequences = True))(sequence_output)
    # lstm_2 = Bidirectional(LSTM(100, dropout=0.3, recurrent_dropout=0.3))(lstm_1)
    out_1 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(cls_token)
    d_1 = Dropout(0.3)(out_1)
    bt = BatchNormalization()(d_1)
    out_2 = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(bt)
    d_2 = Dropout(0.3)(out_2)
    bt_2 = BatchNormalization()(d_2)
    out = Dense(1,activation='sigmoid')(bt_2)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train():
    '''LOADING DATA IN PANDAS DATAFRANE'''
    train_path = str(args['data'])
    train_data = pd.read_csv(train_path)

    val_path = ""
    if args['val'] is not None:
        val_path = str(args['val'])
        val_data = pd.read_csv(val_path)

    ''' We have trained the model on data from jigsaw multilingual toxic comment detection challange so 
        further process will be according to that, you data processing process may differ '''

    #train_data.comment_text = train_data['comment_text'].apply(cleanText)    #we are training on raw sentences so not necessary
    #if you want to do some basic preprocessing uncomment above sentence

    '''TRAINING CONFIGURATION'''
    AUTO = tf.data.experimental.AUTOTUNE

    #configuration
    EPOCHS = 10
    BATCH_SIZE = 16*strategy.num_replicas_in_sync
    MAX_LEN = 192

    x_train,x_test = tokenize_sentence(train_data,val_path,MAX_LEN)
    y_train = train_data.toxic.values

    x_train,x_val,y_train,y_val = split_data(x_train,y_train)

    train_dataset,valid_dataset = tf_data_loader(x_train,x_val,y_train,y_val)

    '''model loading'''
    with strategy.scope():
        transformer_layer = (
            transformers.TFDistilBertModel
            .from_pretrained('distilbert-base-multilingual-cased')     
        )  #here we have used ditilbert model you can use any model like xlnet, bert base etc for more reference visit transformer oficial website
        model = build_model(transformer_layer, max_len=MAX_LEN)
    model.summary()
    

    '''dataset is imbalanced so using classweight'''
    no_of_positive_example = len(train_data[train_data['toxic'] == 1])
    no_of_negative_example = len(train_data[train_data['toxic'] == 0])
    total = len(train_data['toxic'])

    pos_weight = (total/(2*no_of_positive_example))
    neg_weight = (total/(2*no_of_negative_example))
    class_weights = {0 : neg_weight, 1 : pos_weight}

    print(class_weights)

    n_steps = x_train.shape[0] // BATCH_SIZE
    callbacks = ReduceLROnPlateau(patience = 3,factor = 0.5)
    #callbacks = EarlyStopping(patience = 5)
    train_history = model.fit(
        train_dataset,
        steps_per_epoch=n_steps,
        validation_data=valid_dataset,
        epochs=EPOCHS,
        callbacks = callbacks,
        class_weight = class_weights
    )

    acc = train_history.history['accuracy']
    loss = train_history.history['loss']

    val_acc = train_history.history['val_accuracy']
    val_loss = train_history.history['val_loss']
    epochs = range(1,len(val_acc) + 1)
    plt.plot(epochs,acc,'b',label = 'training accuracy')
    plt.plot(epochs,val_acc,'r',label = 'val acc')
    plt.legend()

    plt.figure()
    plt.plot(epochs,loss,'b',label = 'training loss')
    plt.plot(epochs,val_loss,'r',label = 'validation loss')
    plt.legend()

    plt.show()

    model.save_weights(os.path.join(args['savePath'],'mode_weights.h5'))

    #saving in tensorflow js and node js compatible saved_model_format

    export_dir = args['savePath']
    callable = tf.function(model.call)
    concrete_function = callable.get_concrete_function([tf.TensorSpec([None, 192], tf.int32, name="input_ids")])
    tf.saved_model.save(model,export_dir = export_dir,signatures = concrete_function)

    #if training on tpu you may required reload the graph in only cpu format and then convert to saved_model_format

if __name__ == '__main__':
    train()