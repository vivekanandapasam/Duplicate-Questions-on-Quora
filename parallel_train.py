import pickle
LOAD_TRAIN_TEST_PICKLE = True
TRAIN_DATA_PICKLE = './data/data_pickles/train_6b.pkl'
TEST_DATA_PICKLE = './data/data_pickles/test_6b.pkl'

import time
import sys

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
import torch.multiprocessing as mp
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

def forward(model, all_inputs, wordToVec, proc_ind, BATCH_SIZE, optimizer):
    # lstm_hidden = (torch.randn(1, self.BATCH_SIZE, self.LSTM_OUTPUT), torch.randn(1, self.BATCH_SIZE, self.LSTM_OUTPUT))
    ## all_inputs = [q1, q2, is_dup]

    loss = 0
    criterion = nn.CrossEntropyLoss()
    target = torch.LongTensor([[0], [1]])
    for ind, (q1, q2, is_dup) in enumerate(all_inputs[BATCH_SIZE*proc_ind : BATCH_SIZE*proc_ind + BATCH_SIZE]):

        ## q1
        q_vec = [wordToVec[tok] for tok in q1.split(' ') if tok in wordToVec.keys()]
        if (len(q_vec)) == 0: continue
        q1_out = lstm_train(model, q_vec)
        del q_vec
        ## q2
        q_vec = [wordToVec[tok] for tok in q2.split(' ') if tok in wordToVec.keys()]
        if (len(q_vec)) == 0: continue
        q2_out = lstm_train(model, q_vec)
        del q_vec

        # print(q1_out.shape)
        # print(q2_out.shape)

        nn_input = torch.cat((q1_out, q2_out))
        x = model.hidden[0](nn_input)
        # print('nn hiden 0 out ', x.shape)
        x = model.sigmoid(x)
        # print('sig out ', x.shape)
        x = model.hidden[1](x)
        x = model.softmax(x)
        # print('final out', x)

        ## Accumulating loss
        loss += criterion(x.view(1, -1), target[is_dup])

    print('pair index', ind, 'loss', loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss = 0
        # if (proc_ind * BATCH_SIZE) % 10000 == 9999 or ind == len(all_inputs) - 1:



def lstm_train(model, q_vec):
    lstm_hidden = (torch.zeros(1, model.BATCH_SIZE, model.LSTM_OUTPUT), torch.zeros(1, model.BATCH_SIZE, model.LSTM_OUTPUT))
    temp_q_vec = [q_vec]
    # print(len(temp_q_vec[0]))
    out, lstm_hidden = model.lstm(torch.tensor(temp_q_vec), lstm_hidden)
    return out[0][-1]

if __name__ == '__main__':
    ## Training
    model = Network()
    model.share_memory()
    model.train()
    BATCH_SIZE = 100
    # print(sys.argv[1])
    for epoch in range(model.EPOCHS):
        start_time = time.time()
        processes = []
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
        # for i in range(int(3)):
        for i in range(int(len(train_data)/100)+1):
            print('Started process ', i)
            p = mp.Process(target=forward,args=(model, train_data, wordToVec, i, BATCH_SIZE, optimizer))
            p.start()
            processes.append(p)

            Num_Parallel = 2 if len(sys.argv) < 2 else int(sys.argv[1]) ## take command line argument if passed
            if i % Num_Parallel == Num_Parallel - 1: ### Run only Num_Parallel processes at a time
                for p in processes:
                    p.join()
                processes.clear()
                print('Batch run time: ', time.time() - start_time, ' sec')

        print('Epoch run time: ', time.time() - start_time, ' sec')
        torch.save(model, f'./models/model_parallel_{epoch}.pt')
        print('Saved after time: ', time.time() - start_time, ' sec')