from train import Network, TEST_LABEL_PICKLE, TEST_DATA_PICKLE, wordToVec, args, save_folder_name, model
import torch
import time
import pickle
from torch import argmax

print('Loading test data pickle..')
fi = open(TEST_DATA_PICKLE, 'rb')
test_data = pickle.load(fi)
fi.close()
print('Loaded test data pickle..')
print('test size: ', len(test_data),',',len(test_data[0]))

print('Loading test label pickle..')
fi = open(TEST_LABEL_PICKLE, 'rb')
test_label = pickle.load(fi)
fi.close()
print('Loaded test label pickle..')
print('test size: ', len(test_label))

## Testing on test data for accuracy
def test(model, test_data, test_labels):
    start_time = time.time()
    # print('test len ', len(test_data))
    # target = torch.LongTensor([[0], [1]])
    # criterion = nn.CrossEntropyLoss()
    false_pos = 0
    false_neg = 0
    correct = 0
    # loss = 0
    with torch.no_grad():
        ind = 0
        curLoss = 0
        while ind * model.BATCH_SIZE < len(test_data[0]):
            # if ind > 30: break
            current_batch_q1 = test_data[0][ind*model.BATCH_SIZE:(ind+1)*model.BATCH_SIZE]
            current_batch_q2 = test_data[1][ind*model.BATCH_SIZE:(ind+1)*model.BATCH_SIZE]
            #TODO: remove this
            # if ind > 10: break
            x = model.predict(current_batch_q1, current_batch_q2)

            ## Accumulating loss
            batch_labels = torch.IntTensor(test_label[ind*model.BATCH_SIZE:(ind+1)*model.BATCH_SIZE])
            # print(x[:5])
            # print(torch.argmax(x,dim=1)[:5])
            x = torch.argmax(x,dim=1)
            for i in range(len(batch_labels)):
                if((x[i]) == batch_labels[i]):
                    correct += 1
                elif batch_labels[i] == 1:
                    false_neg += 1
                else:
                    false_pos += 1
            # loss = criterion(x.view(1, -1), target[train_label[ind]])
            del batch_labels
            del x

            #### Next batch
            ind += 1


    print("{:.4f}".format(correct/len(test_data[0])), end='\t')
    print("{:.4f}".format(false_pos/len(test_data[0])), end='\t')
    print("{:.4f}".format(false_neg/len(test_data[0])), end='\t')
    print(time.time() - start_time)

import sys

if __name__ == "__main__":

    model.BATCH_SIZE = 1000
    # model_suffix = sys.argv[2]
    num_epochs = args.eps

    print('epoch\tacc\tf_pos\tf_neg\truntime')

    for ep in range(num_epochs):
        model.load_state_dict(torch.load('models/' + save_folder_name + '/model_epoch_' + str(ep) + '_we_' + str(args.weisc) + '.pt'))
        model.eval()
        print(ep, end='\t')
        test(model, test_data, wordToVec)

    sys.stdout.close()