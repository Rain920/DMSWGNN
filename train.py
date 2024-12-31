import os
from os import path

import numpy as np
import torch.optim as optim
import configparser
from model.DG import *
from model.DMSWGNN import *
from model.Utils import *
import scipy.io

#读参数
config_path = './config/MR_DAKGNN.config'
config = configparser.ConfigParser()
config.read(config_path)
cfgTrain = config['train']
cfgModel = config['model']
cfgSave = config['save']
#train
context = int(cfgTrain["context"])
epoch_num = int(cfgTrain["epoch_num"])
fold_num = int(cfgTrain["fold_num"])
lr = float(cfgTrain["lr"])
cuda = cfgTrain["cuda"]
#model
bandwidth = float(cfgModel["bandwidth"])
K = int(cfgModel["K"])
in_dim = int(cfgModel["in_dim"])
out_dim = int(cfgModel["out_dim"])
num_of_nodes = int(cfgModel["num_of_nodes"])
higru_hid = int(cfgModel["higru_hid"])
higru_out = int(cfgModel["higru_out"])
temperature = float(cfgModel["temperature"])
scale = float(cfgModel["scale"])
#save_path
data_path = cfgSave["data_path"]
model_save_path = cfgSave["model_save_path"]
result_save_path = cfgSave["result_save_path"]

#读入数据
ReadList = np.load(data_path ,allow_pickle=True)
Fold_Num = ReadList['Fold_Num']
Fold_Data = ReadList['Fold_Data']
Fold_Label   = ReadList['Fold_Label']
print("Read data successfully")

DataGenerator = kFoldGenerator(fold_num, Fold_Data, Fold_Label)

FFeatures = []
#十折交叉验证
for k in range(fold_num):

    print('Fold #', k)

    train_targets, val_targets, num_train,num_train_lab, num_val,num_val_lab = DataGenerator.getFold(k)

    Features = np.load('./data/Features/' + 'Feature_' + str(k) + '.npz', allow_pickle=True)
    train_data = Features['train_feature']
    val_data = Features['val_feature']

    train_data = torch.from_numpy(train_data.astype(np.float32))
    train_data = torch.FloatTensor(train_data)

    val_data = torch.from_numpy(val_data.astype(np.float32))
    val_data = torch.FloatTensor(val_data)
    print(train_data.shape,val_data.shape)
    DEVICE = torch.device(cuda)
    net = make_model(bandwidth=bandwidth,
                     K=K, in_dim=in_dim,
                     out_dim=out_dim,
                     num_of_nodes=num_of_nodes,
                     higru_hid=higru_hid,
                     higru_out=higru_out,
                     temperature=temperature,
                     scale=scale,
                     DEVICE = DEVICE)
    net.to(DEVICE)
    optimizer = optim.Adam(net.parameters(),lr=lr)
    #model_loss = nn.CrossEntropyLoss()

    best_epoch = 0
    best_acc = 0

    for epoch in range(epoch_num):
        print('Epoch #', epoch)
        net.train()
        loss_ = 0
        acc_ = 0

        start_num = 0
        end_num = 0
        ssstart_num = 0
        eeend_num = 0
        for i in range(9):
            end_num = end_num + num_train[i]
            num_time_slice = num_train[i]                
            eeend_num = eeend_num + num_train_lab[i]
            nnnum_time_slice = num_train_lab[i]
            train_input_d1 = train_data[start_num:end_num, :, :]
            label = train_targets[ssstart_num:eeend_num]
            train_input_d1 = train_input_d1.to(DEVICE)
            label = label.to(DEVICE)
            start_num = end_num
            ssstart_num=eeend_num
            optimizer.zero_grad()
            outputs, loss, fea = net(train_input_d1,label)
            #loss = model_loss(outputs, label)
            loss.backward(retain_graph=True)
            optimizer.step()
            acc_train = accuracy(outputs, label)
            loss_ = loss_ + loss
            acc_ = acc_ + acc_train
        loss_ = loss_ / (i + 1)
        acc_ = acc_ / (i + 1)

        net.eval()
        with torch.no_grad():
            val_data_d1 = val_data.to(DEVICE)
            val_targets = val_targets.to(DEVICE)
            outputs_val, loss_val, fea_val = net(val_data_d1,val_targets)
            #loss_val = model_loss(outputs_val, val_targets)
            acc_val = accuracy(outputs_val, val_targets)
        if acc_val > best_acc:
            best_acc = acc_val
            best_epoch = epoch
            best_output = outputs_val
            best_label = val_targets
            torch.save(net, path.join(model_save_path, 'Best_%d.pth' % (k)))
            #torch.save(net.state_dict(), './result/Best_' + str(k) + '.pt')
            print('better')
        print('loss_train=', loss_.item(), 'acc_train=', acc_, 'loss_val=', loss_val.item(), 'acc_val=', acc_val)

    #Evaluate
    model = torch.load(path.join(model_save_path, 'Best_%d.pth' % (k)))
    model.eval()
    with torch.no_grad():
        evaluate,_,fea_eval = model(val_data_d1,val_targets)
        fold_acc = accuracy(evaluate, val_targets)
        pre = np.array(nn.Softmax(dim=1)(evaluate).cpu())
        pre = np.argmax(pre, 1)
        tru = np.array(val_targets.cpu())
        # fold_acc = accuracy(best_output,val_targets)
        if k == 0:                                                   
            Tru = tru
            Pre = pre
        else:
            Tru = np.concatenate((Tru, tru), axis=0)
            Pre = np.concatenate((Pre, pre), axis=0)
    FFeatures.append(fea_eval.cpu())
    print('------------------------Fold----------------------------')
    print('best-epoc', best_epoch, 'best_acc', best_acc, 'fold-acc',fold_acc)
    print('------------------------Fold----------------------------')
print(Tru.shape, Pre.shape)
scipy.io.savemat('predict_result_1.mat', {'Tru': Tru, 'Pre': Pre})
PrintScore(Tru, Pre, result_save_path)
#np.savez('features_64', fea_eval = FFeatures)