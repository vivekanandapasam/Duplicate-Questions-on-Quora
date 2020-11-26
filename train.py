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
    def __init__(self, lstm_out_size, nn_hid_size, epochs, batch_size, sec_dense_len):
        super().__init__()

        self.LSTM_INPUT = 300
        self.LSTM_OUTPUT = lstm_out_size
        self.LSTM_BATCH_SIZE = 1
        self.NN_HIDDEN_SIZE = nn_hid_size
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size

        self.lstm = nn.LSTM(self.LSTM_INPUT, self.LSTM_OUTPUT, 1, bias = False, batch_first = True)

        # self.hidden = [nn.Linear(self.LSTM_OUTPUT * 2, self.NN_HIDDEN_SIZE), nn.Linear(self.NN_HIDDEN_SIZE, 2)]
        self.dense = nn.Sequential(nn.Linear(self.LSTM_OUTPUT * 4, self.NN_HIDDEN_SIZE), 
                            # nn.Linear(self.NN_HIDDEN_SIZE, sec_dense_len),
                            # nn.Linear(sec_dense_len, 2)
                            nn.Sigmoid(),
                            nn.Linear(self.NN_HIDDEN_SIZE, 2),
                            nn.Softmax(dim=0)
                    )
        ### actual of lstm outputs + sq_diff + hadamard product + lengths of lstm outputs + sq_euclid_dist
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=0)

    def forward(self, all_inputs, test_data, wordToVec, start_epoch, folder_prefix):
        # lstm_hidden = (torch.randn(1, self.LSTM_BATCH_SIZE, self.LSTM_OUTPUT), 
        # torch.randn(1, self.LSTM_BATCH_SIZE, self.LSTM_OUTPUT))
        ## all_inputs = [q1, q2, is_dup]
        target = torch.LongTensor([[0], [1]])
        criterion = nn.CrossEntropyLoss()
        CUR_LR = 0.006
        LR_DECAY = 0.9
        WEIGHT_DECAY = 0.2

        print('Running for config : batch size ', self.BATCH_SIZE, ' , epochs ', self.EPOCHS, ' lstm hidden size '
        , self.LSTM_OUTPUT, ' nn hidden size ', self.NN_HIDDEN_SIZE, ' for ', len(self.dense), ' layers')
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

                ### Calculate loss for every 100 pairs
                if ind % self.BATCH_SIZE == self.BATCH_SIZE-1:
                    print('pair index', ind, 'loss', loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss = 0
                if ind % 10000 == 9999:
                    torch.save(self.state_dict(), 'models/' + folder_prefix + '/model_' + str(ind) + '.pt')
                    print('Saved after time: ', time.time() - start_time, ' sec')

            torch.save(self.state_dict(), 'models/' + folder_prefix + '/model_epoch_' + str(epoch_num) + '.pt')
            print('Model saved! Epoch run time: ', time.time() - start_time, ' sec')
            ## Adding decay to learning rate
            CUR_LR *= LR_DECAY

    ## Given input, get output frm network
    def predict(self, q1, q2, wordToVec):
        ## q1
        q_vec = [wordToVec[tok] for tok in q1.split(' ') if tok in wordToVec.keys()]
        if (len(q_vec)) == 0: return None
        q1_out = self.lstm_train(q_vec)
        del q_vec

        ## q2
        q_vec = [wordToVec[tok] for tok in q2.split(' ') if tok in wordToVec.keys()]
        if (len(q_vec)) == 0: return None
        # q2_len = torch.Tensor(len([q_vec]))
        q2_out = self.lstm_train(q_vec)

        ## Other inputs to dense layers
        sq_diff = (torch.sub(q1_out, q2_out)) ** 2
        had_prod = torch.mul(q1_out, q2_out)

        # Add difference and other inputs here
        x = torch.cat((q1_out, q2_out, sq_diff, had_prod))
        x = self.dense(x)
        return x


    def lstm_train(self, q_vec):
        lstm_hidden = (torch.zeros(1, self.LSTM_BATCH_SIZE, self.LSTM_OUTPUT), 
                torch.zeros(1, self.LSTM_BATCH_SIZE, self.LSTM_OUTPUT))
        nn.init.xavier_uniform_(lstm_hidden[0])
        nn.init.xavier_uniform_(lstm_hidden[1])
        temp_q_vec = [q_vec]
        # print(len(temp_q_vec[0]))
        out, lstm_hidden = self.lstm(torch.tensor(temp_q_vec), lstm_hidden)
        return out[0][-1]

import sys
import os
import argparse

if __name__ == "__main__":

    ## Take values through command line
    my_parser = argparse.ArgumentParser(description='Passed arguments')
    my_parser.add_argument('-logfile',help='Log file name, doesnt need to include folder name',type=str,default='log.txt')
    my_parser.add_argument('-bs',help='Batch size',type=int,default=500)
    my_parser.add_argument('-se',help='Start epoch',type=int,default=0)
    my_parser.add_argument('-lhs',help='LSTM output size',type=int,default=300)
    my_parser.add_argument('-nnhs',help='NN hidden size',type=int,default=512)
    my_parser.add_argument('-sdl',help='2nd dense layer size',type=int,default=32)
    my_parser.add_argument('-eps',help='Total epochs to run',type=int,default=30)

    args = my_parser.parse_args()
    model = Network(batch_size=args.bs, lstm_out_size=args.lhs, nn_hid_size=args.nnhs, epochs=args.eps, sec_dense_len=args.sdl)
    save_folder_name = str(args.bs) + '_' + str(args.lhs) + '_' + str(args.nnhs) + '_' + str(args.eps) + '_' + str(args.sdl)

    ## Creating folder for storing models BATCH SIZE
    os.system("mkdir models/" + save_folder_name)
    ## Logging all prints into log file
    sys.stdout = open("models/" + save_folder_name + '/' + args.logfile, "a")
    model.train()
    model.forward(train_data, test_data, wordToVec, args.se, save_folder_name)
    sys.stdout.close()