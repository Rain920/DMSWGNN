import os
from re import T
import numpy as np
import torch.optim as optim
# from model.DLinear import Model
# from model.PatchTST import Model
# from model.TimeMixer import Model
# from model.iTransformer import Model
# from model.HDMixer import Model
# from model.Informer import Model
from model.MSGNet import Model
from model.Utils import *
from torch.utils.data import DataLoader, TensorDataset
import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from types import SimpleNamespace


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# bandwidth = 0.05
# K = 5
# in_dim = 256
# out_dim = 64

# higru_hid = 64
# higru_out = 128
# temperature = 0.07
# scale = 0.001

B = 32
lr = 0.001
epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_model_path = 'best_model.pth'
result_path = result_timemixer.txt'


def load_data(data_path, file_name):
    # Load data
    data = np.load(os.path.join(data_path, file_name))
    x_train = data['train_feature']
    y_train = data['train_targets']
    x_test = data['val_feature']
    y_test = data['val_targets']
    return x_train, y_train, x_test, y_test


def train(model, train_loader, val_loader, optimizer, scheduler):
    best_epoch = 1
    best_acc = 0

    for epoch in range(epochs):
        print('Epoch #', epoch + 1)
        model.train()
        loss_sum = 0
        acc_sum = 0
        n = 0
        for x, y in tqdm.tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs, loss = model(x, y)
            loss.backward()
            optimizer.step()

            acc = accuracy(outputs, y)
            loss_sum += loss
            acc_sum += acc
            n += 1
        train_loss = loss_sum / n
        train_acc = acc_sum / n

        val_loss, val_acc = val(model, val_loader)
        scheduler.step()
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Train accuracy: {:.4f}'\
              ' Val loss: {:.6f} | Val accuracy: {:.4f}| GPU occupy: {:.6f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], train_loss, train_acc, val_loss, val_acc, gpu_mem_alloc))
        
        if val_acc >= best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print('Model saved at', best_model_path)
    
    print('Best epoch:', best_epoch, 'Best accuracy:', best_acc)


@torch.no_grad()
def val(model, val_loader):
    model.eval()
    loss_sum = 0
    n = 0
    pred_list = []
    label_list = []
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)
        outputs, loss = model(x, y)
        loss_sum += loss
        n += 1
        pred_list.append(outputs)
        label_list.append(y)
    loss_avg = loss_sum / n
    pred = torch.cat(pred_list, dim=0)
    label = torch.cat(label_list, dim=0)
    acc = accuracy(pred, label)
    return loss_avg, acc


@torch.no_grad()
def test(test_loader):
    # load best model
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    loss_sum = 0
    n = 0
    pred_list = []
    label_list = []
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        outputs, loss = model(x, y)
        # acc = accuracy(outputs, y)
        loss_sum += loss
        # acc_sum += acc
        n += 1
        pred_list.append(outputs)
        label_list.append(y)
    loss_avg = loss_sum / n
    # stack all the predictions and labels
    pred = torch.cat(pred_list, dim=0)
    label = torch.cat(label_list, dim=0)
    acc = accuracy(pred, label)
    print('Test loss:', loss_avg.item(), 'Test accuracy:', acc.item())
    label = torch.argmax(label, dim=1)
    pred = torch.argmax(pred, dim=1)
    label = label.cpu().numpy()
    pred = pred.cpu().numpy()
    PrintScore(label, pred, result_path)
    


def scale(data):
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    std = np.maximum(std, 1.0)
    data = (data - mean) / std
    return data


if __name__ == '__main__':

    # Load data
    data_path = 'data/Features/'
    file_name = 'Feature_9.npz'

    x_train, y_train, x_test, y_test = load_data(data_path, file_name)
    print('Train x: {}, Train y: {}, Test x: {}, Test y: {}'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # package data by torch dataloader
    train_data = TensorDataset(x_train, y_train)
    test_data = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_data, batch_size=B, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=B, shuffle=False, drop_last=False)

    # build model
    configs = {
        'seq_len': x_train.shape[2],
        'pred_len': x_train.shape[2],
        'label_len': 0,
        'top_k': 3,
        'd_model': 64,
        'd_ff': 128,
        'n_heads': 8,
        'dropout': 0.1,
        'c_out': x_train.shape[1],
        'conv_channel': 64,
        'skip_channel': 64,
        'gcn_depth': 2,
        'propalpha': 0.3,
        'node_dim': 10,
        'task_name': 'classification',
        'e_layers': 2,
        'enc_in': x_train.shape[1],
        'embed': 'timeF',
        'freq': 'h',
        'individual': False,
        'num_classes': y_train.shape[1],
    }
    configs = SimpleNamespace(**configs)
    model = Model(configs)
    # print(model)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    print('-----------------------------Training-----------------------------')
    train(model, train_loader, test_loader, optimizer, scheduler)

    print('-----------------------------{}-----------------------------'.format(file_name))
    print('-----------------------------Testing-----------------------------')
    test(test_loader)
