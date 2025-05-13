import sys

import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset
import copy
from sklearn.metrics import accuracy_score
import numpy as np
import time
from utils import *
import pandas as pd
from transformers import AdamW
import time

class Base(object):
    def __init__(self, args):
        super(Base, self).__init__()
        self.args = args
        
    def FL_Train(self, init_global_model, client_all_loaders, test_loader, FL_params):

        print('\n')
        print(5 * "#" + "  Federated Training Start  " + 5 * "#")
        global_model = init_global_model
        result_list = []

        checkpoints_ls = []
        avg_acc = 0
        for client_idx in range(len(client_all_loaders)):
            (test_loss, test_acc) = self.test(init_global_model, test_loader[client_idx], self.args)
            print("Client {} Test Loss = {}, Test Accuracy = {}".format(client_idx, test_loss, test_acc))

        std_time = time.time()
        for epoch in range(FL_params.global_epoch):
            num_client = int(FL_params.frac * len(client_all_loaders))
            selected_client = np.random.choice([i for i in range(len(client_all_loaders))], num_client, replace=False)
            client_all_loaders_select = [client_all_loaders[i] for i in selected_client]
            test_loader_select = [test_loader[i] for i in selected_client]
            # for idx in self.args.new_users:
            #     client_all_loaders_select.append(client_all_loaders[idx])
            data_size = []
            for dataloader in client_all_loaders_select:
                data_size.append(len(dataloader.dataset))
            agg_weights = [x / sum(data_size) for x in data_size]
            client_models_dict = self.global_train_once(epoch, global_model, client_all_loaders_select, test_loader_select, FL_params, checkpoints_ls)

            global_model_dict = self.fedavg(client_models_dict, agg_weights)

            global_model.load_state_dict(global_model_dict)

            if epoch < FL_params.global_epoch - 1:
                del client_models_dict
                # torch.cuda.empty_cache()
           
            all_idx = [k for k in range(len(client_all_loaders))]
            client_test_acc = []
            end_time = time.time()
            consume_time = end_time - std_time

            for client_idx in all_idx:
                (test_loss, test_acc) = self.test(global_model, test_loader[client_idx], FL_params)
                client_test_acc.append(test_acc)
                result_list.append([epoch, client_idx, test_loss, test_acc, consume_time])
                print("Epoch {} Client {} Test Loss = {}, Test Accuracy = {}".format(epoch, client_idx, test_loss, test_acc))
           
            avg_acc = sum(client_test_acc) / len(client_test_acc)

            print("FL Round = {}, Global Model Accuracy= {}".format(epoch, avg_acc))
        if len(client_all_loaders) < FL_params.num_user:
            if FL_params.paradigm == 'motivation':
                path_global = './save_model/motivation_pretrained_globalmodel_shift_{}_{}+{}.pth'.format(FL_params.shift, FL_params.target_domain, FL_params.source_domain)
            elif FL_params.paradigm == 'sequence':
                path_global = './save_model/sequence_pretrained_globalmodel_shift_{}_{}+{}.pth'.format(FL_params.shift, FL_params.target_domain, FL_params.source_domain)
            else:
                path_global = './save_model/pretrained_globalmodel_shift_{}_{}+{}.pth'.format(FL_params.shift, FL_params.target_domain, FL_params.source_domain)
            torch.save(global_model.state_dict(), path_global)
            for idx, param in enumerate(client_models_dict):
                if FL_params.paradigm == 'sequence':  
                    path_client = './save_model/sequence_client{}model_shift_{}.pth'.format(idx, FL_params.shift)
                else:
                    path_client = './save_model/pretrained_client{}model_shift_{}_{}+{}.pth'.format(idx, FL_params.shift, FL_params.target_domain, FL_params.source_domain)

                torch.save(param, path_client)
            del client_models_dict

        df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id','Test_loss', 'Test_acc', 'Time'])
        if len(client_all_loaders) == FL_params.num_user:
            df.to_csv('./results/Acc_loss_{}_shift_{}_{}+{}.csv'.format(FL_params.paradigm, FL_params.shift, FL_params.target_domain, FL_params.source_domain))
        elif len(client_all_loaders) < FL_params.num_user:
            df.to_csv('./results/Acc_loss_pretrain_shift_{}_{}+{}.csv'.format(FL_params.shift, FL_params.target_domain, FL_params.source_domain))

        print(5 * "#" + "  Federated Training End  " + 5 * "#")

        return global_model


    # training sub function
    def global_train_once(self, epoch, global_model, client_data_loaders, test_loaders, FL_params, checkpoints_ls=None):
        global_model.to(FL_params.device)
     
        client_models_dict = []
        lr = FL_params.lr

        for idx, client_data in enumerate(client_data_loaders):
            model = copy.deepcopy(global_model)
            
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.2, weight_decay=5e-4)
            # model.to(device)
            model.train()
            print("Epoch {} Client {} Training Start".format(epoch, idx))

            model = self.local_train(idx, model, optimizer, client_data, FL_params)

            client_models_dict.append(model.state_dict())

        return client_models_dict


    """
    Function：
    Test the performance of the model on the test set
    """
    def local_train(self, idx, model, optimizer, data_loader, FL_params):
        for local_epoch in range(FL_params.local_epoch):
            criteria = nn.CrossEntropyLoss()
            for batch_idx, (data, target) in enumerate(data_loader):
                optimizer.zero_grad()
                data = data.to(FL_params.device)
                target = target.to(FL_params.device)
                target = target.long()
                _, pred = model(data)

                loss = criteria(pred, target)
                loss.backward()
                optimizer.step()
        return model


    def test(self, model, test_loader, FL_params):
        for param in model.parameters():
            device = param.device
            break
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        criteria = nn.CrossEntropyLoss()
        with torch.no_grad():
            
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                target = target.long()
                model.to(device)
                _, output = model(data)
                test_loss += criteria(output, target)  # sum up batch loss
                pred = torch.argmax(output, axis=1)
                correct += torch.sum(torch.eq(pred, target)).item()
                total += len(target)
       
        test_loss /= len(test_loader.dataset)
        test_acc = correct/total
        return (test_loss, test_acc)


    """
    Function：
    FedAvg
    """
    def fedavg(self, local_models_dict, weights):
        avg_state_dict = copy.deepcopy(local_models_dict[0])

        for layer in avg_state_dict.keys():
            avg_state_dict[layer] = 0
            for client_idx in range(len(local_models_dict)):
                avg_state_dict[layer] += local_models_dict[client_idx][layer] * weights[client_idx]

        return avg_state_dict