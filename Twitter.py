import pandas as pd 
pd.options.mode.chained_assignment = None
import numpy as np
import nltk 
nltk.download('punkt') 
import scipy
import matplotlib.pyplot as plt 
import gensim
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer 
from nltk import word_tokenize

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.constraints import MaxNorm




from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook

def ingest():
    data = pd.read_csv('/Documents/Twitter Sentiment Analysis/Tweets.csv', encoding='latin-1') 
    data.columns=["Sentiment","ItemID","Date","Blank","SentimentSource","SentimentText"]
    data.drop(['ItemID', 'SentimentSource'], axis=1, inplace=True)
    data = data[data.Sentiment.isnull() == False]
    data['Sentiment'] = data['Sentiment'].map( {4:1, 0:0}) 
    data = data[data['SentimentText'].isnull() == False]
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    print ('dataset loaded with shape', data.shape  )  
    return data

data = ingest()

tokenizer = TweetTokenizer()
def tokenize(tweet):
    try:
        tweet = tweet.lower()
        tokens = tokenizer.tokenize(tweet)
        tokens = list(filter(lambda t: not t.startswith('@'), tokens))
        tokens = list(filter(lambda t: not t.startswith('#'), tokens))
        tokens = list(filter(lambda t: not t.startswith('http'), tokens))
        return tokens
    except:
        return 'NC'
def postprocess(data):
    data['tokens'] = data['SentimentText'].progress_map(tokenize)
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

data = postprocess(data)

LabeledSentence = gensim.models.doc2vec.LabeledSentence 

def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized 
#Splitting for training and testing
x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(1000000).tokens),
                                                    np.array(data.head(1000000).Sentiment), test_size=0.2)
x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST')
data_labellised= labelizeTweets(np.array(data.tokens), 'data')
n=1000000
n_dim = 200
tweet_w2v = Word2Vec(size=n_dim, min_count=10)
tweet_w2v.build_vocab([x.words for x in tqdm(data_labellised)])
tweet_w2v.train([x.words for x in tqdm(data_labellised)],total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)
tweet_w2v['hello']

# defining the chart
plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A map of 10000 word vectors",
                        tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                        x_axis_type=None, y_axis_type=None, min_border=1)

# getting a list of word vectors. limit to 10000. each is of 200 dimensions
word_vectors = [tweet_w2v[w] for w in list(tweet_w2v.wv.vocab.keys())[:5000]]

# dimensionality reduction. converting the vectors to 2d vectors
from sklearn.manifold import TSNE
tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
tsne_w2v = tsne_model.fit_transform(word_vectors)

# putting everything in a dataframe
tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
tsne_df['words'] = list(tweet_w2v.wv.vocab.keys())[:5000]

# plotting. the corresponding word appears when you hover on the data point.
plot_tfidf.scatter(x='x', y='y', source=tsne_df)
hover = plot_tfidf.select(dict(type=HoverTool))
hover.tooltips={"word": "@words"}
show(plot_tfidf)
print ('building tf-idf matrix ...')
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in data_labellised])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print ('vocab size :', len(tfidf))
#Build tweet vector to give input to FFNN
def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word] #combining w2v vectors with tfidf value of words in the tweet.
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec
train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_train))])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_test))])

test_vecs_w2v = scale(test_vecs_w2v)
#Training 3 layered FFNN
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=200))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_vecs_w2v, y_train, epochs=100, batch_size=10000, verbose=2)

# Evaluating accuracy score
score = model.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)
print(model.metrics_names[0],": ",score[0],"\n",model.metrics_names[1],": ",score[1])

def ingesttest():
    testdata = pd.read_csv('/Documents/Twitter Sentiment Analysis/TweetsTest.csv', encoding='latin-1')
    testdata.columns=["Sentiment","ItemID","Date","Blank","SentimentSource","SentimentText"]
    testdata.drop(['ItemID', 'SentimentSource'], axis=1, inplace=True)
    testdata = testdata[testdata.Sentiment.isnull() == False]
    testdata['Sentiment'] = testdata['Sentiment'].map( {4:1, 0:0, 2:1})
    testdata = testdata[testdata['SentimentText'].isnull() == False]
    testdata.reset_index(inplace=True)
    testdata.drop('index', axis=1, inplace=True)
    print ('dataset loaded with shape', testdata.shape  )  
    return testdata

testdata = ingesttest()
testdata = postprocess(testdata)
test_X=np.array(testdata.tokens)
test_y=np.array(testdata.Sentiment)
test_w2v_vecs = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x:x, test_X))])
score = model.evaluate(test_w2v_vecs,test_y, batch_size=128, verbose=2)
print(model.metrics_names[0],": ",score[0],"\n",model.metrics_names[1],": ",score[1])

#visualization
result=model.predict_classes(test_w2v_vecs)
countone=0
countzero=0
for i,j in enumerate(result):
    if result[i].item()==0:
        countzero += 1
    if result[i].item()==1:
        countone +=1
        
Positive_Tweets =(countone/len(result))*100
Negative_Tweets =(countzero/len(result))*100
print('Positive Tweets %: ',Positive_Tweets)
print('Negative Tweets %: ',Negative_Tweets)
plt.pie([Positive_Tweets,Negative_Tweets ],
        labels=['Positive Tweets','Negative Tweets'],
        colors=['g','r'],
        startangle=90,
        shadow= True,
        autopct='%1.1f %%')
plt.title('Sentiment Pie Chart')
plt.savefig(' sentiment')