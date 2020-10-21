## Tracking what words are in data
import pandas as pd
import string

q1_col_name = 'question1'
q2_col_name = 'question2'
DATASET_NAME = "./data/filtered.tsv"
df = pd.read_csv(DATASET_NAME, sep='\t')
print(df.head())
df[q1_col_name] = df[q1_col_name].astype('str')
df[q2_col_name] = df[q2_col_name].astype('str')
print(df.dtypes)

bagOfWords = set()
wordToVec = {}

def addToBag(s):
    toks = s.split(' ')
    global wordToVec
    for tok in toks:
        if tok != '':
            bagOfWords.add(tok)
        # if tok == 'futurewhat' : print(s)

df[q1_col_name].apply(addToBag)
df[q2_col_name].apply(addToBag)

print('Total words in dataset: ', len(bagOfWords))

GLOVE_VEC_FILE = './data/glove_6b/glove.6B.300d.txt'

with open(GLOVE_VEC_FILE) as vf:
    i=0
    for line in vf:
        i+=1
        toks = line[:-1].split(' ') # Remove '\n' at end
        # if (len(toks) != 301): print(toks[0], " ", len(toks))
        # if ((toks[0].isupper())): print(toks[0], " ", len(toks))
        # if (len(toks) != 301) or toks[0] in ['?','.','(',')','!',":",'-','+']: print(toks[0], " ", len(toks))
        if(toks[0] in bagOfWords):
            wordToVec[toks[0]] = list(map(lambda x : float(x), toks[1:]))

    print("Total pre-trained words: ", i)
    print("Total words with vectors in df:", len(wordToVec))
    print(len(wordToVec['why']))

import pickle
PICKLE_NAME = "./data/WordToVec.pkl"
STORE_PICKLE = True
if STORE_PICKLE:
    pickle.dump(wordToVec, open(PICKLE_NAME, 'wb'))


'''

Total words in dataset:  83460
Total pre-trained words:  400001
Total words with vectors in df: 59905

'''