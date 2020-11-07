import pickle
LOAD_TRAIN_TEST_PICKLE = True
TRAIN_DATA_PICKLE = './data/data_pickles/train_6b.pkl'
TEST_DATA_PICKLE = './data/data_pickles/test_6b.pkl'

import time

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
    print('train size: ', len(train_data))

    print('Loading test pickle..')
    fi = open(TEST_DATA_PICKLE, 'rb')
    test_data = pickle.load(fi)
    fi.close()
    print('Loaded test pickle..')
    print('test size: ', len(test_data))

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
        self.LSTM_BATCH_SIZE = 1
        self.NN_HIDDEN_SIZE = 128
        self.EPOCHS = 20

        self.lstm = nn.LSTM(self.LSTM_INPUT, self.LSTM_OUTPUT, 1, bias = False, batch_first = True)

        # self.hidden = [nn.Linear(self.LSTM_OUTPUT * 2, self.NN_HIDDEN_SIZE), nn.Linear(self.NN_HIDDEN_SIZE, 2)]
        self.dense = [nn.Linear(self.LSTM_OUTPUT * 4, self.NN_HIDDEN_SIZE), nn.Linear(self.NN_HIDDEN_SIZE, 2)]
        ### actual of lstm outputs + sq_diff + hadamard product + lengths of lstm outputs + sq_euclid_dist
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, all_inputs, test_data, wordToVec, start_epoch, BATCH_SIZE):
        # lstm_hidden = (torch.randn(1, self.LSTM_BATCH_SIZE, self.LSTM_OUTPUT), torch.randn(1, self.LSTM_BATCH_SIZE, self.LSTM_OUTPUT))
        ## all_inputs = [q1, q2, is_dup]
        target = torch.LongTensor([[0], [1]])
        criterion = nn.CrossEntropyLoss()
        CUR_LR = 0.0006
        LR_DECAY = 0.9
        WEIGHT_DECAY = 0.2

        print('Running for config : batch size ', BATCH_SIZE, ' , epochs ', self.EPOCHS, ' lstm hidden size ', self.LSTM_OUTPUT
            , ' nn hidden size ', self.NN_HIDDEN_SIZE, ' for ', len(self.dense), ' layers')
        for epoch_num in range(start_epoch, start_epoch + self.EPOCHS):
            start_time = time.time()
            optimizer = torch.optim.Adam(self.parameters(), lr = CUR_LR, weight_decay = WEIGHT_DECAY)
            loss = 0
            for ind, (q1, q2, is_dup) in enumerate(all_inputs):

                x = self.predict(q1, q2, wordToVec)
                if not torch.is_tensor(x): continue
                # print('final out', x)

                ## Accumulating loss
                loss += criterion(x.view(1, -1), target[is_dup])
                # print('loss ', loss)

                ### Calculate loss for every 100 pairs
                if ind % BATCH_SIZE == BATCH_SIZE-1:
                    print('pair index', ind, 'loss', loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss = 0
                    # print('Current run time: ', time.time() - start_time, ' sec')
                    # self.test(test_data, wordToVec)
                if ind % 10000 == 9999:
                    torch.save(self.state_dict(), './models/model_' + str(ind) + '.pt')
                    print('Saved after time: ', time.time() - start_time, ' sec')

            torch.save(self.state_dict(), './models/b' + str(BATCH_SIZE) + '/model_epoch_' + str(epoch_num)
                + '_' + str(self.NN_HIDDEN_SIZE)+ '_' + str(self.LSTM_OUTPUT) + '.pt')
            print('Model saved! Epoch run time: ', time.time() - start_time, ' sec')
            ## Adding decay to learning rate
            CUR_LR *= LR_DECAY

    ## Given input, get output frm network
    def predict(self, q1, q2, wordToVec):
        ## q1
        q_vec = [wordToVec[tok] for tok in q1.split(' ') if tok in wordToVec.keys()]
        if (len(q_vec)) == 0: return None
        # q1_len = torch.Tensor([len(q_vec)])
        q1_out = self.lstm_train(q_vec)
        del q_vec
        ## q2
        q_vec = [wordToVec[tok] for tok in q2.split(' ') if tok in wordToVec.keys()]
        if (len(q_vec)) == 0: return None
        # q2_len = torch.Tensor(len([q_vec]))
        q2_out = self.lstm_train(q_vec)

        # print(q1_out.shape)
        # print(q1_len)
        # print(q2_out.shape)
        ## Other inputs to dense layers
        sq_diff = (torch.sub(q1_out, q2_out)) ** 2
        # sq_euc_dist = torch.sum(sq_diff).view(-1)
        # print(sq_euc_dist.shape)
        had_prod = torch.mul(q1_out, q2_out)

        # Add difference and other inputs here
        nn_input = torch.cat((q1_out, q2_out, sq_diff, had_prod))
        # nn_input = torch.cat((q1_out, q2_out, q1_len, q2_len, sq_diff, sq_euc_dist, had_prod))
        # nn_input = torch.cat((q1_out, q2_out))
        x = self.dense[0](nn_input)
        # print('nn hiden 0 out ', x.shape)
        x = self.sigmoid(x)
        # print('sig out ', x.shape)
        x = self.dense[1](x)
        x = self.softmax(x)
        return x


    def lstm_train(self, q_vec):
        lstm_hidden = (torch.zeros(1, self.LSTM_BATCH_SIZE, self.LSTM_OUTPUT), torch.zeros(1, self.LSTM_BATCH_SIZE, self.LSTM_OUTPUT))
        temp_q_vec = [q_vec]
        # print(len(temp_q_vec[0]))
        out, lstm_hidden = self.lstm(torch.tensor(temp_q_vec), lstm_hidden)
        return out[0][-1]

    # ## Testing on test data for accuracy
    # def test(self, test_data, wordToVec):
    #     # start_time = time.time()
    #     # print('test len ', len(test_data))
    #     target = torch.LongTensor([[0], [1]])
    #     # criterion = nn.CrossEntropyLoss()
    #     false_pos = 0
    #     false_neg = 0
    #     correct = 0
    #     # loss = 0
    #     with torch.no_grad():
    #         for (q1, q2, is_dup) in (test_data):
    #             x = self.predict(q1, q2, wordToVec)
    #             # loss += criterion(x.view(1, -1), target[is_dup])
    #             if is_dup:
    #                 if(x[0] > x[1]): false_neg += 1
    #                 else : correct += 1
    #             else:
    #                 if(x[0] < x[1]): false_pos += 1
    #                 else : correct += 1

    #     print('********** Testing ***********')
    #     print('Accuracy : ', correct/len(test_data), ' %')
    #     print('False pos : ', false_pos/len(test_data), ' %')
    #     print('False neg : ', false_neg/len(test_data), ' %')
    #     # print('Loss on test data: ', loss.item())
    #     print('******************************')
    #     # print('Test ran for time : ', time.time() - start_time, ' sec')

import sys
import os

BATCH_SIZE = 500
## Training
model = Network()
start_epoch = 0
## Loading saved model
if (len(sys.argv) >= 2) :
    print('Starting with model ' + sys.argv[1], ' at epoch ', sys.argv[2], ' with batchsize ', BATCH_SIZE)
    model.load_state_dict(torch.load(sys.argv[1]))
    start_epoch = int(sys.argv[2])
    BATCH_SIZE = int(sys.argv[3])


## Creating folder for storing models BATCH SIZE
os.system("mkdir models/b" + str(BATCH_SIZE))
model.train()
model.forward(train_data, test_data, wordToVec, start_epoch, BATCH_SIZE)