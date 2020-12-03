import pickle
LOAD_TRAIN_TEST_PICKLE = True
TRAIN_DATA_PICKLE = './data/data_pickles/train_data_6b.pkl'
TEST_DATA_PICKLE = './data/data_pickles/test_data_6b.pkl'
TRAIN_LABEL_PICKLE = './data/data_pickles/train_label_6b.pkl'
TEST_LABEL_PICKLE = './data/data_pickles/test_label_6b.pkl'

import time

## Loading word to vector dict
print('Loading wordToVec pickle..')
PICKLE_NAME = "./data/WordToVec.pkl"
fi = open(PICKLE_NAME, 'rb')
wordToVec = pickle.load(fi)
fi.close()
print('Loaded wordToVec pickle..')

## Loading train test sets
# test_data, test_label = None, None

if __name__ == "__main__":
    if LOAD_TRAIN_TEST_PICKLE:
        print('Loading train data pickle..')
        fi = open(TRAIN_DATA_PICKLE, 'rb')
        train_data = pickle.load(fi)
        fi.close()
        print('Loaded train data pickle..')
        print('train size: ', len(train_data),',',len(train_data[0]))

        print('Loading train label pickle..')
        fi = open(TRAIN_LABEL_PICKLE, 'rb')
        train_label = pickle.load(fi)
        fi.close()
        print('Loaded train label pickle..')
        print('train size: ', len(train_label))

    else:
        import pandas as pd
        from sklearn.model_selection import train_test_split

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
        data_q1 = []
        data_q2 = []
        labels = []
        missed_rows = 0
        for ind in df.index:
            q1_embed = [wordToVec[tok] for tok in df[q1_col_name][ind].split(' ') if tok in wordToVec.keys()]
            q2_embed = [wordToVec[tok] for tok in df[q2_col_name][ind].split(' ') if tok in wordToVec.keys()]
            if len(q1_embed) == 0 or len(q2_embed) == 0:
                missed_rows += 1
                continue
            data_q1.append(q1_embed)
            data_q2.append(q2_embed)
            labels.append(df[is_dup_col_name][ind])
            if len(q1_embed) > max_qs_len : max_qs_len = len(q1_embed)
            if len(q2_embed) > max_qs_len : max_qs_len = len(q2_embed)
        print('Max qs len: ', max_qs_len)
        print('Missed ', missed_rows,' rows')

        ## Train-test split
        TEST_SIZE = 0.2
        train_data_q1, test_data_q1, train_data_q2, test_data_q2, train_label, test_label = train_test_split(
                            data_q1, data_q2, labels, test_size = TEST_SIZE, random_state = 0)
        print('Total data length: ', len(data_q1))
        print('Train data length: ', len(train_data_q1))
        print('Test data length: ', len(test_data_q1))
        train_data = [train_data_q1, train_data_q2]
        test_data = [test_data_q1, test_data_q2]

        ## Saving into pickle
        SAVE_INTO_PICKLE = True
        if SAVE_INTO_PICKLE:
            fi = open(TRAIN_DATA_PICKLE, 'wb')
            pickle.dump(train_data, fi)
            fi.close()
            fi = open(TEST_DATA_PICKLE, 'wb')
            pickle.dump(test_data, fi)
            fi.close()
            fi = open(TRAIN_LABEL_PICKLE, 'wb')
            pickle.dump(train_label, fi)
            fi.close()
            fi = open(TEST_LABEL_PICKLE, 'wb')
            pickle.dump(test_label, fi)
            fi.close()



import torch
from torch import nn
torch.manual_seed(0)
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence, pad_packed_sequence

## Creating model
class Network(nn.Module):
    def __init__(self, lstm_out_size, nn_hid_size, batch_size, sec_dense_len, tog_extra_inps):
        super().__init__()

        self.LSTM_INPUT = 300
        self.LSTM_OUTPUT = lstm_out_size
        self.LSTM_BATCH_SIZE = batch_size
        self.NN_HIDDEN_SIZE = nn_hid_size
        self.BATCH_SIZE = batch_size
        self.NN_INPUT = (self.LSTM_OUTPUT * 4 + 3) if(tog_extra_inps) else self.LSTM_OUTPUT * 4
        self.HAS_EXTRA = tog_extra_inps

        self.lstm = nn.LSTM(self.LSTM_INPUT, self.LSTM_OUTPUT, 1, bias = True, batch_first = True)

        self.dense = nn.Sequential(
                            nn.BatchNorm1d(self.NN_INPUT),
                            nn.Linear(self.NN_INPUT, self.NN_HIDDEN_SIZE)
                            # nn.Linear(self.NN_HIDDEN_SIZE, sec_dense_len),
                            # nn.Linear(sec_dense_len, 2)
                            ,nn.Sigmoid()
                            ,nn.Linear(self.NN_HIDDEN_SIZE, 2)
                    )

    def forward(self, train_data, train_label, start_epoch, total_epochs, folder_prefix, pos_wei_scale):

        # criterion = nn.CrossEntropyLoss(weight=torch.Tensor([class_weight, 1-class_weight]))
        CUR_LR = 0.0006
        # LR_DECAY = 0.9
        WEIGHT_DECAY = 0

        print('Running for config : batch size ', self.BATCH_SIZE, ' , epochs ', total_epochs, ' lstm hidden size '
        , self.LSTM_OUTPUT, ' nn hidden size ', self.NN_HIDDEN_SIZE, ' for ', len(self.dense), ' layers'
        , ' with extra inputs ', self.HAS_EXTRA, ' Weight scale for positive labels = ', pos_wei_scale)

        class_weight = sum(train_label)/len(train_data)
        num_pos = sum(train_label)
        num_in_batch = len(train_data[0])
        criterion = nn.CrossEntropyLoss(
                        # weight=torch.Tensor([class_weight, 1-class_weight])
                        weight=torch.Tensor([num_pos/num_in_batch, (num_in_batch-num_pos)/num_in_batch])
                    )

        ## Changing data into packed sequences
        # seq_train_data_1 = []
        # seq_train_data_2 = []
        # # ind = 0
        # # maxLength = max(max(map(lambda x : len(x[0]), train_data)),max(map(lambda x : len(x[1]), train_data)))
        # empty_token = [0 for _ in  range(self.LSTM_INPUT)]
        # # changing each question into tensor
        # temp_tensors = [list(map(torch.Tensor, train_data[0])), list(map(torch.Tensor, train_data[1]))]
        # del train_data
        # train_data = temp_tensors
        # while ind * self.BATCH_SIZE < len(train_data):
        #     temp1 = train_data[0][ind*self.BATCH_SIZE : (ind+1)*self.BATCH_SIZE]
        #     seq_lens_1 = map(len, temp1)
        #     print(list(seq_lens_1)[:10])
        #     # for x in train_data[ind*self.BATCH_SIZE : (ind+1)*self.BATCH_SIZE]:
        #     #     seq_lens.append(len(x[0]))
        #     #     temp_padded.append(x[0] + [empty_token for _ in range(maxLength - len(x[0]))])
        #     #     break
        #     seq_train_data_1.append(nn.utils.rnn.pack_padded_sequence(
        #                     pad_sequence(torch.Tensor(temp1),batch_first=True,padding_value=empty_token),
        #                     seq_lens_1, batch_first=True))
        #     break
        # return

        for epoch_num in range(start_epoch, start_epoch + total_epochs):

            start_time = time.time()
            optimizer = torch.optim.Adam(self.parameters(), lr = CUR_LR, weight_decay = WEIGHT_DECAY)
            # loss = 0
            ind = 0
            while ind * self.BATCH_SIZE < len(train_data[0]):
                current_batch_q1 = train_data[0][ind*self.BATCH_SIZE:(ind+1)*self.BATCH_SIZE]
                current_batch_q2 = train_data[1][ind*self.BATCH_SIZE:(ind+1)*self.BATCH_SIZE]
                #TODO: remove this
                if ind > 10: break
                x = self.predict(current_batch_q1, current_batch_q2)

                ## Accumulating loss
                batch_labels = torch.LongTensor(train_label[ind*self.BATCH_SIZE:(ind+1)*self.BATCH_SIZE])
                # print(torch.sum(batch_labels),' - ', batch_labels.shape[0])
                loss = criterion(x, batch_labels)
                # del batch_labels
                # print('zzzzzzzzzzzzzzzzzzz                  zzzzzzzzzzzzz')

                print('pair index', (ind+1)*self.BATCH_SIZE, 'loss', loss)
                print('preds-', torch.sum(torch.argmax(x,dim=1)))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                del loss
                # del criterion

                # SAVE_AFTER = int(30000/self.BATCH_SIZE)
                # # if ind % SAVE_AFTER == SAVE_AFTER - 1:
                # #     torch.save(self.state_dict(), 'models/' + folder_prefix + '/model_' + str(ind*self.BATCH_SIZE) + '.pt')
                # #     print('Saved after time: ', time.time() - start_time, ' sec')

                ## Next batch
                ind += 1

            torch.save(self.state_dict(), 'models/' + folder_prefix + '/model_epoch_' + str(epoch_num) + '_we_' + str(pos_wei_scale) + '.pt')
            print('Model saved! Epoch ', epoch_num, ' run time: ', time.time() - start_time, ' sec')
            ## Adding decay to learning rate
            # CUR_LR *= LR_DECAY
            del optimizer

    def predict(self, current_batch_q1, current_batch_q2):
        q1_lens = [(i, len(el)) for i,el in enumerate(current_batch_q1)]
        q2_lens = [(i, len(el)) for i,el in enumerate(current_batch_q2)]

        q1 = pack_sequence(list(map(torch.Tensor, current_batch_q1))
                    , enforce_sorted=False)
        q2 = pack_sequence(list(map(torch.Tensor, current_batch_q2))
                    , enforce_sorted=False)

        q1_out = self.lstm_train(q1)
        # print(q1_out[0][0])
        # print(q1_out[1][0])
        ## q2
        q2_out = self.lstm_train(q2)
        # print(q2_out[0][0])
        # print(q2_out[1][0])

        ########################### Unsort the question outputs to match pairs
        q1_lens.sort(key = lambda x : -1*x[1])
        # print(q1_out.shape)
        # print(q1[0][0][0:8])
        # print(current_batch_q1[q1_lens[0][0]][0][0:8])
        temp = [0]*len(q1_lens)
        # print(q1_lens)
        # print(q1_out.shape)
        for i in range(len(temp)):
            temp[q1_lens[i][0]] = q1_out[i]
        # print(temp[0][:10])
        # q1_out = torch.Tensor(temp)
        del q1_out
        q1_out = torch.stack(temp)
        # print(q1_out.shape)
        # print(q1_out[0][:10])
        # return
        del temp

        q2_lens.sort(key = lambda x : -1*x[1])
        # print(q2[0][0][0:8])
        # print(current_batch_q2[q2_lens[0][0]][0][0:8])
        temp = [0]*len(q2_lens)
        for i in range(len(temp)):
            temp[q2_lens[i][0]] = q2_out[i]
        del q2_out
        q2_out = torch.stack(temp)
        del temp

        ## Other inputs to dense layers
        sq_diff = (torch.sub(q1_out, q2_out)) ** 2
        had_prod = torch.mul(q1_out, q2_out)
        # print(sq_diff[0][0])
        # print(sq_diff[1][0])
        # print(had_prod[0][0])
        # print(had_prod[1][0])
        # print(q1_out[0],q2_out[0],had_prod[0],sep=' ')

        # Add difference and other inputs here
        x = torch.cat((q1_out, q2_out, sq_diff, had_prod), dim=1)
        # if(self.HAS_EXTRA):
        #     temp_q1_lens = torch.Tensor([[len(el)] for el in (current_batch_q1)])
        #     temp_q2_lens = torch.Tensor([[len(el)] for el in (current_batch_q2)])
        #     # print('final - ', temp_q1_lens[0],' - ', len(current_batch_q1[0]))
        #     x = torch.cat((x, temp_q1_lens, temp_q2_lens, torch.sum(sq_diff,dim=1).view(-1,1)), dim=1)
        #     # print('final - ', x.shape)
        # print(x[0][0])
        # print(x[1][0])
        # print(x[0][300])
        # print(x[1][300])
        # print(x[0][600])
        # print(x[1][600])
        # print(x[0][900])
        # print(x[1][900])
        x = self.dense(x)
        # print('final out', x[0])
        # print('final out', x[1])

        # print('q2', q2.data.shape)
        # print('tr', train_data[1][0])
        # print('q2', q2.data[0])
        
        del q1
        del q2
        del q1_lens
        del q2_lens
        return x


    def lstm_train(self, q_vec):
        # temp_q_vec = torch.tensor([q_vec])
        # print(len(temp_q_vec[0]))
        # hid = torch.rand(1, self.BATCH_SIZE, self.LSTM_OUTPUT)
        # cel = torch.rand(1, self.BATCH_SIZE, self.LSTM_OUTPUT)
        # out, lstm_hidden = self.lstm(q_vec, (hid, cel))
        out, lstm_hidden = self.lstm(q_vec)
        # del hid
        # del cel
        return lstm_hidden[0][0]

import sys
import os
import argparse

## Take values through command line
my_parser = argparse.ArgumentParser(description='Passed arguments')
my_parser.add_argument('-logfile',help='Log file name, doesnt need to include folder name',type=str,default='log.txt')
my_parser.add_argument('-bs',help='Batch size',type=int,default=500)
my_parser.add_argument('-se',help='Start epoch',type=int,default=0)
my_parser.add_argument('-lhs',help='LSTM output size',type=int,default=300)
my_parser.add_argument('-nnhs',help='NN hidden size',type=int,default=1000)
my_parser.add_argument('-sdl',help='2nd dense layer size',type=int,default=32)
my_parser.add_argument('-eps',help='Total epochs to run',type=int,default=30)
my_parser.add_argument('-weisc',help='Scale to multiply the loss weight for positive labels',type=float,default=0.9)
my_parser.add_argument('-togex',help='Toggle lenghs and distance inputs',type=bool,default=False)

args = my_parser.parse_args()
print(args.togex)
save_folder_name = str(args.bs) + '_' + str(args.lhs) + '_' + str(args.nnhs) + '_' + str(args.sdl) + ('_' + str(args.togex) if args.togex else '')

model = Network(batch_size=args.bs, lstm_out_size=args.lhs, nn_hid_size=args.nnhs
                        , sec_dense_len=args.sdl
                        , tog_extra_inps=args.togex)


## Creating folder for storing models BATCH SIZE
os.system("mkdir models/" + save_folder_name)
## Logging all prints into log file
# sys.stdout = open("models/" + save_folder_name + '/' + args.logfile, "a")


if __name__ == "__main__":
    model.train()
    model.forward(train_data, train_label, args.se, args.eps, save_folder_name, args.weisc)
    # sys.stdout.close()