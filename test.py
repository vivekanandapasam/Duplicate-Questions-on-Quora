from train import Network, test_data, train_data, wordToVec
import torch
import time

## Testing on test data for accuracy
def test(model, test_data, wordToVec):
    start_time = time.time()
    # print('test len ', len(test_data))
    target = torch.LongTensor([[0], [1]])
    # criterion = nn.CrossEntropyLoss()
    false_pos = 0
    false_neg = 0
    correct = 0
    # loss = 0
    with torch.no_grad():
        for (q1, q2, is_dup) in (test_data):
            x = model.predict(q1, q2, wordToVec)
            # loss += criterion(x.view(1, -1), target[is_dup])
            if is_dup:
                if(x[0] > x[1]): false_neg += 1
                else : correct += 1
            else:
                if(x[0] < x[1]): false_pos += 1
                else : correct += 1

    print(correct/len(test_data), ',', false_pos/len(test_data), ',', false_neg/len(test_data), ',', time.time() - start_time)

import sys

if __name__ == "__main__":
    model = Network()
    model_folder = sys.argv[1]
    model_suffix = sys.argv[2]
    num_epochs = int(sys.argv[3])

    print('epoch_num,accuracy,false_pos,false_neg,runtime')

    for ep in range(num_epochs):
        model.load_state_dict(torch.load(model_folder + 'model_epoch_' + str(ep) + model_suffix + '.pt'))
        model.eval()
        print(ep,',', end='')
        test(model, test_data, wordToVec)