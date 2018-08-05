from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.feature_extraction.text import  TfidfTransformer
import numpy as np
from tflearn.data_utils import to_categorical, pad_sequences
import os
import pickle
from My_word2vec_Glove import *


def tf_dif(content):
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(content))

    feature_words = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    f = open('tf_idf.txt', 'a')
    for i, sentence in enumerate(content):
        for word in sentence:
            if word in feature_words:
                j = feature_words.index(word)
                f.write(str(weight[i][j]) + ' ')
            else:
                print(str(word))
                f.write(str(0))
        f.write('\n')
    f.close()


def read(tfidf_path):
    f = open(tfidf_path)
    lines = f.readlines()
    res = []
    for line in lines:
        tfidf = map(float, line.strip())
        res.append(tfidf)
    return np.array(res)

def transform_text(X,X1,X2,word2index):
    trainX,validX,testX=[],[],[]
    for sentence in X:
        now=[]
        for word in sentence:
            if word in word2index:
                now.append(word2index[word])
            else:
                now.append(word2index['<UNK>'])
        trainX.append(now)
    for sentence in X1:
        now=[]
        for word in sentence:
            if word in word2index:
                now.append(word2index[word])
            else:
                now.append(word2index['<UNK>'])
        validX.append(now)
    for sentence in X2:
        now=[]
        for word in sentence:
            if word in word2index:
                now.append(word2index[word])
            else:
                now.append(word2index['<UNK>'])
        testX.append(now)
    return trainX,validX,testX



trainX, trainY, validX, validY, testX, testY = read_yelp(path="../dataset", year=2013)
print(trainX[0])
print(validX[0])
# cache_path = '../dataset/cache_vocabulary_label_pik/' + "yelp_2013_glove_eospad_double_200d.5allwords.pik"
# if os.path.exists(cache_path):  # 如果缓存文件存在，则直接读取
#     with open(cache_path, 'rb') as data_f:
#         embeddings, vocabulary_word2index, vocabulary_index2word = pickle.load(data_f, encoding='latin1')
#
# if "<PAD>" not in vocabulary_word2index:
#     bound = np.sqrt(6.0) / np.sqrt(len(vocabulary_word2index))
#     vocabulary_word2index["<PAD>"] = len(vocabulary_word2index)
#     vocabulary_index2word[len(vocabulary_index2word)] = "<PAD>"
#     embeddings = np.append(embeddings, np.random.uniform(-bound, bound,200).astype(np.float32)).reshape(-1,200)
#
# print("<PAD> index:", vocabulary_word2index['<PAD>'])
# vocab_size = len(vocabulary_word2index)
# print("cnn_model.vocab_size:", vocab_size)
#
# print("testX.shape:", np.array(testX).shape)
#
# print("testY.shape:", np.array(testY).shape)
#
# print("trainX[0]:", trainX[0])
#
# print("trainX[1820]:", trainX[1820])
#
# trainX, validX, testX = transform_text(trainX, validX, testX, vocabulary_word2index)
#
# print("testX.shape:", np.array(testX).shape)
#
# print("testY.shape:", np.array(testY).shape)
# print("testX[0]:", testX[0])
# print("testY[0]:", testY[0])
#
# # raw_input("continue")
#
# # 2.Data preprocessing.Sequence padding
# print("start padding & transform to one hot...")
# trainX = pad_sequences(trainX, maxlen=1250,
#                        value=vocabulary_word2index["<PAD>"])  # padding to max length
# validX = pad_sequences(validX, maxlen=1250,
#                        value=vocabulary_word2index["<PAD>"])  # padding to max length
# testX = pad_sequences(testX, maxlen=1250,
#                       value=vocabulary_word2index["<PAD>"])  # padding to max length
# # with open(FLAGS.cache_path, 'w') as data_f: #save data to cache file, so we can use it next time quickly.
# #    pickle.dump((trainX,trainY,testX,testY,vocabulary_index2word),data_f)
# print("trainX[0]:", trainX[0])  # ;print("trainY[0]:", trainY[0])
# # Converting labels to binary vectors
# print("end padding & transform to one hot...")
# content = np.concatenate((trainX, validX, testX), axis=0)
# with  open('word_index.pk', 'wb') as f:
#      pickle.dump(content, f)
#
print("start calculating tfidf score...")
with  open('word_index.pk', 'rb') as f:
     content = pickle.load(f)
len_train, len_valid, len_test = len(trainX), len(validX), len(testX)
print(type(trainX))
print('shape:', np.array(trainX).shape, np.array(validX).shape, np.array(testX).shape)
print('content.shape:', content.shape)
str_content = []
tmp = 0
for i in content:
    tmp += 1
    if tmp % 100000 == 0:
        print (tmp)
    str_content.append(list(map(str, i)))

tf_idf_fea = tf_idf(str_content)
with open('tf_idf.pk', 'wb') as f:
    pickle.dump(tf_idf_fea)
tf_train, tf_valid, tf_test = tf_idf_fea[:len_train], tf_idf_fea[len_train: len_train + len_valid], tf_idf_fea[len_train + len_valid:]
print('tf_train shape', tf_train.shape)
print('tf_valid.shape', tf_valid.shape)
print('tf_test.shape', tf_test.shape)
print('tf_train[0]:', tf_train[0])

print("end tfidf")

