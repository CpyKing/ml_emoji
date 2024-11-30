from dataloader import CustomImageDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models import MyModel
import torch
from utils import get_seq2cls_table

def get_precision(pred, y):
    pred = torch.argmax(pred, dim=1)
    y = torch.argmax(y, dim=1)
    judge = (pred == y)
    return torch.sum(judge) / pred.shape[0]

if __name__ == '__main__':
    test_dataloader = DataLoader(CustomImageDataset(mode='test'))
    net = MyModel()
    net.load_state_dict(torch.load('save_model.pkl'))
    net.to(device=0)

    net.eval()

    l  = 0
    precision = 0
    cnt = 0
    seq2cls = get_seq2cls_table()
    pred_table = {}
    for X, y in test_dataloader:
        # print(torch.argmax(y[0]))
        X = X.to(device=0)
        out = net(X)
        # print(y[0], seq2cls[str(torch.argmax(out).item())])
        pred_table[y[0]] = seq2cls[str(torch.argmax(out).item())]
        cnt += 1
        if cnt > 50:
            break
    with open('pred.csv', 'w') as f:
        for item in pred_table:
            f.writelines(item + ',' + pred_table[item] + '\n')
