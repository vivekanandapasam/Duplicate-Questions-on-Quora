import pickle
LOAD_TRAIN_TEST_PICKLE = True
TRAIN_DATA_PICKLE = './data/data_pickles/train_6b.pkl'
TEST_DATA_PICKLE = './data/data_pickles/test_6b.pkl'



## Loading word to vector dict
print('Loading wordToVec pickle..')
PICKLE_NAME = "./data/WordToVec.pkl"
fi = open(PICKLE_NAME, 'rb')
wordToVec = pickle.load(fi)
fi.close()
print('Loaded wordToVec pickle..')

## Loading train test sets
train_data, test_data = None, None
if LOAD_TRAIN_TEST_PICKLE:
    print('Loading train pickle..')
    fi = open(TRAIN_DATA_PICKLE, 'rb')
    train_data = pickle.load(fi)
    fi.close()
    print('Loaded train pickle..')

    print('Loading test pickle..')
    fi = open(TEST_DATA_PICKLE, 'rb')
    test_data = pickle.load(fi)
    fi.close()
    print('Loaded test pickle..')

else:
    import pandas as pd

    ## Loading dataset
    q1_col_name = 'question1'
    q2_col_name = 'question2'
    is_dup_col_name = 'is_duplicate'
    DATASET_NAME = "./data/filtered.tsv"
    df = pd.read_csv(DATASET_NAME, sep='\t')
    print(df.tail())
    df[q1_col_name] = df[q1_col_name].astype('str')
    df[q2_col_name] = df[q2_col_name].astype('str')
    print(df.dtypes)

    ## Creating input for training
    max_qs_len = 0
    data = []
    for ind in df.index:
        data.append((df[q1_col_name][ind], df[q2_col_name][ind], df[is_dup_col_name][ind]))
        if len(df[q1_col_name][ind].split(' ')) > max_qs_len : max_qs_len = len(df[q1_col_name][ind].split(' '))
        if len(df[q2_col_name][ind].split(' ')) > max_qs_len : max_qs_len = len(df[q2_col_name][ind].split(' '))
    print('Max qs len: ', max_qs_len)

    ## Train-test split
    TEST_SIZE = 0.2
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data, test_size = TEST_SIZE, random_state = 0)
    print('Total data length: ', len(data))
    print('Train data length: ', len(train_data), ' each element ', train_data[0])
    print('Test data length: ', len(test_data), ' each element ', test_data[0])

    ## Saving into pickle
    SAVE_INTO_PICKLE = True
    if SAVE_INTO_PICKLE:
        fi = open(TRAIN_DATA_PICKLE, 'wb')
        pickle.dump(train_data, fi)
        fi.close()
        fi = open(TEST_DATA_PICKLE, 'wb')
        pickle.dump(test_data, fi)
        fi.close()



import torch
from torch import nn
torch.manual_seed(0)

## Creating model
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.LSTM_INPUT = 300
        self.LSTM_OUTPUT = 300
        self.BATCH_SIZE = 1
        self.NN_HIDDEN_SIZE = 128
        self.EPOCHS = 10

        self.lstm = nn.LSTM(self.LSTM_INPUT, self.LSTM_OUTPUT, 1, bias = False, batch_first = True)

        self.hidden = [nn.Linear(self.LSTM_OUTPUT * 2, self.NN_HIDDEN_SIZE), nn.Linear(self.NN_HIDDEN_SIZE, 2)]
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, all_inputs, wordToVec):
        # lstm_hidden = (torch.randn(1, self.BATCH_SIZE, self.LSTM_OUTPUT), torch.randn(1, self.BATCH_SIZE, self.LSTM_OUTPUT))
        ## all_inputs = [q1, q2, is_dup]
        for _ in range(self.EPOCHS):
            loss = 0
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)
            target = torch.LongTensor([[0], [1]])
            for ind, (q1, q2, is_dup) in enumerate(all_inputs):

                ## q1
                q_vec = [wordToVec[tok] for tok in q1.split(' ') if tok in wordToVec.keys()]
                if (len(q_vec)) == 0: continue
                q1_out = self.lstm_train(q_vec)
                del q_vec
                ## q2
                q_vec = [wordToVec[tok] for tok in q2.split(' ') if tok in wordToVec.keys()]
                if (len(q_vec)) == 0: continue
                q2_out = self.lstm_train(q_vec)
                del q_vec

                # print(q1_out.shape)
                # print(q2_out.shape)

                nn_input = torch.cat((q1_out, q2_out))
                x = self.hidden[0](nn_input)
                # print('nn hiden 0 out ', x.shape)
                x = self.sigmoid(x)
                # print('sig out ', x.shape)
                x = self.hidden[1](x)
                x = self.softmax(x)
                # print('final out', x)

                ## Accumulating loss
                loss += criterion(x.view(1, -1), target[is_dup])

                ### Calculate loss for every 100 pairs
                if ind % 100 == 99:
                    print('pair index', ind, 'loss', loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss = 0


    def lstm_train(self, q_vec):
        lstm_hidden = (torch.zeros(1, self.BATCH_SIZE, self.LSTM_OUTPUT), torch.zeros(1, self.BATCH_SIZE, self.LSTM_OUTPUT))
        temp_q_vec = [q_vec]
        # print(len(temp_q_vec[0]))
        out, lstm_hidden = self.lstm(torch.tensor(temp_q_vec), lstm_hidden)
        return out[0][-1]


## Training
model = Network()
model.train()
model.forward(train_data, wordToVec)