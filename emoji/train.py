from dataloader import CustomImageDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import multiprocessing as mp
from models import MyModel
import os
import torch
import numpy as np
import time
import datetime

def get_precision(pred, y):
    pred = torch.argmax(pred, dim=1)
    y = torch.argmax(y, dim=1)
    judge = (pred == y)
    return torch.sum(judge) / pred.shape[0]

def evaluation(net, val_dataloader, rank=0):
    net.eval()
    with torch.no_grad():
        criterion = torch.nn.CrossEntropyLoss()
        l = 0
        precision = 0
        cnt = 0
        for X, y in val_dataloader:
            X = X.to(device=rank)
            y = y.to(device=rank)
            out = net(X)
            loss = criterion(out, y)
            l += loss

            precision += get_precision(out, y)
            cnt += 1
        print('val loss ', l / cnt)
        print('val precision ', str(precision.item() / cnt * 100) + '%')
    return precision.item() / cnt * 100
    
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8001'
    torch.distributed.init_process_group("gloo", rank = rank, world_size = world_size)
    nccl_group = torch.distributed.new_group(backend='nccl', timeout=datetime.timedelta(0, 3600 * 3))
    return nccl_group

if __name__ == '__main__':
    mp.set_start_method("spawn")
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    

    dataset = CustomImageDataset(data_path='train_scale')
    data_size = len(dataset)
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    split_idx = int(data_size * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    
    nccl_group = setup(rank, world_size)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=300, num_workers=8, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=200, shuffle=True, num_workers=4)
    print(len(train_dataloader))
    net = MyModel()
    net.to(device=rank)
    net = DDP(net, device_ids=[rank], output_device=rank, process_group=nccl_group)

    optimizer = torch.optim.Adam(
            [
                {'params': net.parameters()},
            ],
            lr=0.001,
            weight_decay=0.01
        )
    step_schedule = torch.optim.lr_scheduler.StepLR(step_size=5, gamma=0.75, optimizer=optimizer)
    criterion = torch.nn.CrossEntropyLoss()

    epochs = 50
    max_precision = 0
    for e in range(epochs):
        l  = 0
        precision = 0
        cnt = 0
        
        print('EPOCH ', e)
        start_loading = 0
        end_loading = 0
        net.train()
        torch.distributed.barrier()
        for X, y in train_dataloader:
            # print(torch.argmax(y[0]))
            # end_loading = time.time()
            # print('load data', end_loading - start_loading)
            X = X.to(device=rank)
            y = y.to(device=rank)
            optimizer.zero_grad()
            out = net(X)
            loss = criterion(out, y)
            l += loss
            loss.backward()
            optimizer.step()
            precision += get_precision(out, y)
            cnt += 1
            y = torch.argmax(y, dim=1)
            # print(cnt)
            # print('training time', time.time() - end_loading)
            # start_loading = time.time()
        step_schedule.step()
        print('train loss ', l / cnt)
        print('train precision ', str(precision.item() / cnt * 100) + '%')
        if rank == 0:
            print('rank 0 test')
            val_precision = evaluation(net, val_dataloader, rank)
            if cnt == 0:
                max_precision = val_precision
            if val_precision > max_precision:
                torch.save(net.module.state_dict(), 'save_model.pkl')
                max_precision = val_precision
                print('\n\t\t\t\t\t\t\t save models .....\n')
