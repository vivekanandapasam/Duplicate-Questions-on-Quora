import pandas as pd

INPUT_DATA = './data/quora_duplicate_questions.tsv'
FILTERED_DATA = './data/filtered.tsv' #################  Name of final filtered dataset #################
q1_col_name = 'question1'
q2_col_name = 'question2'

'''
////////////////////////////
    Uncomment 13-30 for creating filtered.tsv
////////////////////////////
'''
df = pd.read_csv(INPUT_DATA,index_col=0,sep='\t')
print(df.tail())
print('\nDropping na...\n')
df.dropna(inplace=True)
print(df.tail())

def isAscii(s):
    # print(s)
    try:
        s[q1_col_name].encode('ascii')
        s[q2_col_name].encode('ascii')
        return True
    except UnicodeEncodeError:
        return False

df2 = df[df.apply(isAscii, axis=1)]

import string
def removePunc(s):
    return str(s).translate(str.maketrans(string.punctuation, ''.join([' ' for _ in string.punctuation]))).lower()

df2[q1_col_name] = df2[q1_col_name].apply(removePunc)
df2[q2_col_name] = df2[q2_col_name].apply(removePunc)

print(df2.tail())
df2.to_csv(FILTERED_DATA, index=False, sep='\t')



'''
////////////////////////////
    Checking stats of filtered
////////////////////////////
'''
df2 = pd.read_csv(FILTERED_DATA, sep='\t')
print(df2.dtypes)
print(df2.head())

print('\n***************************')
print('Length: ', len(df2))
print('Rows with label 0: ', len(df2[df2['is_duplicate'] == 0])/len(df2))
print('Rows with label 1: ', len(df2[df2['is_duplicate'] == 1])/len(df2))
print('***************************')
# print(df2.tail())

'''

***************************
Length:  395545
Rows with label 0:  0.6289853240465687
Rows with label 1:  0.37101467595343135
***************************

'''