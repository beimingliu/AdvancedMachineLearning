
# Word Embeddings

## Sentiment analysis with word embedings
classify movie reviews into positive and negative. The large movie view dataset (http://ai.stanford.edu/~amaas/data/sentiment/) contains a collection of 50,000 reviews from
IMDB. The dataset contains an even number of positive and negative re-
views.The dataset is divided into training and test sets. The training set is
the same 25,000 labeled reviews. The sentiment classication task consists
of predicting the polarity (positive or negative) of a given text.


```python
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
%matplotlib inline
import pickle
```


```python
!wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
```

    --2018-02-22 22:04:21--  http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    Resolving ai.stanford.edu... 171.64.68.10
    Connecting to ai.stanford.edu|171.64.68.10|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 84125825 (80M) [application/x-gzip]
    Saving to: 'aclImdb_v1.tar.gz'

    aclImdb_v1.tar.gz   100%[===================>]  80.23M  15.3MB/s    in 5.2s    

    2018-02-22 22:04:55 (15.4 MB/s) - 'aclImdb_v1.tar.gz' saved [84125825/84125825]



**unzip the file in the local directory**

`gunzip aclImdb_v1.tar.gz`

`tar -xvf aclImdb_v1.tar`

## First look at the data


```python
PATH='aclImdb/'
names = ['neg','pos']
```


```python
!ls aclImdb/train
```

    labeledBow.feat  pos/ unsupBow.feat urls_pos.txt
    neg/    unsup/      urls_neg.txt  urls_unsup.txt



```python
# source: fastai library
def texts_labels_from_folders(path, folders):
    texts,labels = [],[]
    for idx,label in enumerate(folders):
        for fname in glob(os.path.join(path, label, '*.*')):
            texts.append(open(fname, 'r').read())
            labels.append(idx)
    return texts, np.array(labels).astype(np.int64)
```


```python
%%time
trn,trn_y = texts_labels_from_folders(f'{PATH}train',names)
val,val_y = texts_labels_from_folders(f'{PATH}test',names)
```

    CPU times: user 2.1 s, sys: 719 ms, total: 2.82 s
    Wall time: 2.82 s



```python
len(trn), len(trn_y), len(val), len(val_y)
```




    (25000, 25000, 25000, 25000)



## 1. Use the libary spacy to tokenize your data.


```python
import spacy
import string
import re
from spacy.symbols import ORTH
```


```python
# first time run this
#!python3 -m spacy download en
```


```python
# borrowed from fast.ai (https://github.com/fastai/fastai/blob/master/fastai/nlp.py)

re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
def sub_br(x): return re_br.sub("\n", x)

my_tok = spacy.load('en')
def spacy_tok(x):
    #x = x
    return [tok.text.lower() for tok in my_tok.tokenizer(sub_br(x))]
```


```python
trn[2]
```




    "This film lacked something I couldn't put my finger on at first: charisma on the part of the leading actress. This inevitably translated to lack of chemistry when she shared the screen with her leading man. Even the romantic scenes came across as being merely the actors at play. It could very well have been the director who miscalculated what he needed from the actors. I just don't know.<br /><br />But could it have been the screenplay? Just exactly who was the chef in love with? He seemed more enamored of his culinary skills and restaurant, and ultimately of himself and his youthful exploits, than of anybody or anything else. He never convinced me he was in love with the princess.<br /><br />I was disappointed in this movie. But, don't forget it was nominated for an Oscar, so judge for yourself."




```python
print(spacy_tok(trn[2]))
```

    ['this', 'film', 'lacked', 'something', 'i', 'could', "n't", 'put', 'my', 'finger', 'on', 'at', 'first', ':', 'charisma', 'on', 'the', 'part', 'of', 'the', 'leading', 'actress', '.', 'this', 'inevitably', 'translated', 'to', 'lack', 'of', 'chemistry', 'when', 'she', 'shared', 'the', 'screen', 'with', 'her', 'leading', 'man', '.', 'even', 'the', 'romantic', 'scenes', 'came', 'across', 'as', 'being', 'merely', 'the', 'actors', 'at', 'play', '.', 'it', 'could', 'very', 'well', 'have', 'been', 'the', 'director', 'who', 'miscalculated', 'what', 'he', 'needed', 'from', 'the', 'actors', '.', 'i', 'just', 'do', "n't", 'know', '.', '\n\n', 'but', 'could', 'it', 'have', 'been', 'the', 'screenplay', '?', 'just', 'exactly', 'who', 'was', 'the', 'chef', 'in', 'love', 'with', '?', 'he', 'seemed', 'more', 'enamored', 'of', 'his', 'culinary', 'skills', 'and', 'restaurant', ',', 'and', 'ultimately', 'of', 'himself', 'and', 'his', 'youthful', 'exploits', ',', 'than', 'of', 'anybody', 'or', 'anything', 'else', '.', 'he', 'never', 'convinced', 'me', 'he', 'was', 'in', 'love', 'with', 'the', 'princess', '.', '\n\n', 'i', 'was', 'disappointed', 'in', 'this', 'movie', '.', 'but', ',', 'do', "n't", 'forget', 'it', 'was', 'nominated', 'for', 'an', 'oscar', ',', 'so', 'judge', 'for', 'yourself', '.']


## Glove

## 2. Download embedding vectors from https://nlp.stanford.edu/projects/glove/.

## 3. Read the 300 dimensional Glove embeddings into a dictionary.


```python
!ls
```

    [34maclImdb[m[m           embeddings.p      gloves.p          hw5.ipynb
    aclImdb_v1.tar    glove.6B.300d.txt hw-embedding.pdf



```python
globe_path = "glove.6B.300d.txt"
```


```python
def load_word_embedings(file =globe_path):
    embeddings = {}
    with open(file, 'r') as infile:
        for line in infile:
            values = line.split()
            embeddings[values[0]] = np.asarray(values[1:], dtype='float32')
    return embeddings
```


```python
%%time
embeddings = load_word_embedings()
```

    CPU times: user 28.1 s, sys: 1.6 s, total: 29.7 s
    Wall time: 30.9 s



```python
# pickle.dump( embeddings, open( "embeddings.p", "wb" ) )
```


```python
%%time
# load gloves from pickle which is much faster
embeddings = pickle.load( open( "embeddings.p", "rb" ) )
```

    CPU times: user 1.68 s, sys: 627 ms, total: 2.31 s
    Wall time: 2.32 s


## 4. Create *average* feature embedding for each sentence. You may want to ignore stopwords.


```python
len(embeddings.keys())
```




    400000




```python
def sentence_features(s, embeddings=embeddings, emb_size=300):
    words = spacy_tok(s)
    words = [w for w in words if w.isalpha() and w in embeddings]
    if len(words) == 0:
        return np.zeros(3 * emb_size)
    M = [embeddings[w] for w in words]
    M = np.array(M)
    v_mean = M.mean(axis=0)
    v_min = M.min(axis=0)
    v_max = M.max(axis=0)
    return np.hstack([v_mean,v_min, v_max])
```


```python
def sentence_features_v2(s, embeddings=embeddings, emb_size=300):
    words = spacy_tok(s) #tokenizer
    words = [w for w in words if w.isalpha() and w in embeddings]
    if len(words) == 0:
        return np.hstack([np.zeros(emb_size)])
    M = np.array([embeddings[w] for w in words])
    return M.mean(axis=0)
```


```python
%%time
# create sentence vectors
x_train = np.array([sentence_features_v2(i) for i in trn])
```

    CPU times: user 39.5 s, sys: 279 ms, total: 39.8 s
    Wall time: 40 s



```python
%%time
x_val = np.array([sentence_features_v2(i) for i in val])
```

    CPU times: user 36.2 s, sys: 234 ms, total: 36.4 s
    Wall time: 36.5 s


## 5. Fit an XGBoost classifier to this data. Report test and training errors.


```python
import xgboost as xgb
```


```python
%%time
d_train = xgb.DMatrix(x_train, label=trn_y)
d_val = xgb.DMatrix(x_val, label=val_y)

xgb_pars = {"min_child_weight": 100, "eta": 0.03, "max_depth": 8,
            "subsample": 0.5, "silent" : 1, "colsample_bytree": 0.4,
            "nthread": 8,
            "eval_metric": "logloss", "objective": "binary:logistic"}

watchlist = [(d_train, 'train'), (d_val, 'valid')]

bst = xgb.train(xgb_pars, d_train, 2000, watchlist, early_stopping_rounds=100, verbose_eval=100)
```

    [0]	train-logloss:0.687224	valid-logloss:0.687575
    Multiple eval metrics have been passed: 'valid-logloss' will be used for early stopping.

    Will train until valid-logloss hasn't improved in 100 rounds.
    [100]	train-logloss:0.465641	valid-logloss:0.49207
    [200]	train-logloss:0.399465	valid-logloss:0.441601
    [300]	train-logloss:0.363565	valid-logloss:0.416341
    [400]	train-logloss:0.339773	valid-logloss:0.401955
    [500]	train-logloss:0.32184	valid-logloss:0.392844
    [600]	train-logloss:0.307297	valid-logloss:0.386234
    [700]	train-logloss:0.295013	valid-logloss:0.381361
    [800]	train-logloss:0.284476	valid-logloss:0.378237
    [900]	train-logloss:0.275189	valid-logloss:0.37578
    [1000]	train-logloss:0.266685	valid-logloss:0.373537
    [1100]	train-logloss:0.258645	valid-logloss:0.372101
    [1200]	train-logloss:0.251643	valid-logloss:0.371093
    [1300]	train-logloss:0.244961	valid-logloss:0.370306
    [1400]	train-logloss:0.23872	valid-logloss:0.36965
    [1500]	train-logloss:0.232907	valid-logloss:0.369013
    [1600]	train-logloss:0.227371	valid-logloss:0.368874
    Stopping. Best iteration:
    [1524]	train-logloss:0.231591	valid-logloss:0.368852

    CPU times: user 14min 21s, sys: 8.57 s, total: 14min 30s
    Wall time: 1min 55s



```python
y_pred = bst.predict(d_val)
y_pred = [round(i) for i in y_pred]

sum(y_pred == val_y)/len(y_pred)
```




    0.83543999999999996




```python
# logistic regression benchmark
from sklearn.linear_model import LogisticRegression
m = LogisticRegression(C=10, dual=True)
m.fit(x_train, trn_y)
preds = m.predict(x_val)
(preds==val_y).mean()
```




    0.83660000000000001



## 6. Compare previous results to fitting XGBoost to a one-hot encoding
representation of the data with bag of words. Report test and training
errors.


```python
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import CountVectorizer
```


```python
%%time
freq = CountVectorizer()
x_train_ = freq.fit_transform(trn)
x_val_ = freq.transform(val)
```

    CPU times: user 8.26 s, sys: 163 ms, total: 8.42 s
    Wall time: 8.43 s



```python
%%time
d_train = xgb.DMatrix(x_train_, label=trn_y)
d_val = xgb.DMatrix(x_val_, label=val_y)

xgb_pars = {"min_child_weight": 50,
            "eta": 0.05,
            "max_depth": 8,
            #"subsample": 0.5,
            "silent" : 1,
            #"colsample_bytree": 0.4,
            "nthread": 8,
            "eval_metric": "logloss", "objective": "binary:logistic"}

watchlist = [(d_train, 'train'), (d_val, 'valid')]

bst = xgb.train(xgb_pars, d_train, 2000, watchlist, early_stopping_rounds=100, verbose_eval=100)
```

    [0]	train-logloss:0.681089	valid-logloss:0.681198
    Multiple eval metrics have been passed: 'valid-logloss' will be used for early stopping.

    Will train until valid-logloss hasn't improved in 100 rounds.
    [100]	train-logloss:0.411805	valid-logloss:0.432926
    [200]	train-logloss:0.345302	valid-logloss:0.381341
    [300]	train-logloss:0.307416	valid-logloss:0.356062
    [400]	train-logloss:0.282342	valid-logloss:0.341037
    [500]	train-logloss:0.262916	valid-logloss:0.331188
    [600]	train-logloss:0.247591	valid-logloss:0.324449
    [700]	train-logloss:0.235224	valid-logloss:0.319576
    [800]	train-logloss:0.223943	valid-logloss:0.316036
    [900]	train-logloss:0.21466	valid-logloss:0.313344
    [1000]	train-logloss:0.205895	valid-logloss:0.311312
    [1100]	train-logloss:0.198509	valid-logloss:0.309834
    [1200]	train-logloss:0.191528	valid-logloss:0.308789
    [1300]	train-logloss:0.184958	valid-logloss:0.308111
    [1400]	train-logloss:0.178524	valid-logloss:0.307511
    [1500]	train-logloss:0.172735	valid-logloss:0.307153
    [1600]	train-logloss:0.167726	valid-logloss:0.307141
    Stopping. Best iteration:
    [1568]	train-logloss:0.169533	valid-logloss:0.306924

    CPU times: user 57min 33s, sys: 3min 43s, total: 1h 1min 16s
    Wall time: 8min 20s



```python
y_pred = bst.predict(d_val)
y_pred = [round(i) for i in y_pred]

sum(y_pred == val_y)/len(y_pred)
```




    0.86695999999999995




```python
# logistic regression benchmark
from sklearn.linear_model import LogisticRegression
m = LogisticRegression(C=10, dual=True)
m.fit(x_train_, trn_y)
preds = m.predict(x_val_)
(preds==val_y).mean()
```




    0.85940000000000005



## References
* https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb
* https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle/
* https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
