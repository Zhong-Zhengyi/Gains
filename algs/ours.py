from algs.fl_base import Base
from models.digit5 import *
from models.amazon import *
import torch
import torch.nn as nn
import numpy as np
import copy
import pandas as pd
import time
import copy


class Ours(Base):
    def __init__(self, args):
        super(Ours, self).__init__(args)
        self.args = args

    def FL_Train(self, init_global_model, client_all_loaders, test_loader, FL_params):
        self.client_all_loaders = client_all_loaders
        self.test_loader = test_loader
        detect_data_loader = self.construct_detect_loader()
        print('\n')
        print(5 * "#" + "  Ours Training Start  " + 5 * "#")
        global_model = copy.deepcopy(init_global_model)
        result_list = []

        checkpoints_ls = []
        avg_acc = 0
        for client_idx in range(len(client_all_loaders)):
            (test_loss, test_acc) = self.test(init_global_model, test_loader[client_idx], self.args)
            print("Client {} Test Loss = {}, Test Accuracy = {}".format(client_idx, test_loss, test_acc))
        data_size = []
        for dataloader in client_all_loaders:
            data_size.append(len(dataloader.dataset))
        data_agg_weights = [x / sum(data_size) for x in data_size]
        old_agg_weights = [x / sum(data_size[0:-1]) for x in data_size[0:-1]]
        print("Initial_Agg_weights: ", data_agg_weights)

        std_time = time.time()
        for epoch in range(FL_params.global_epoch):
            client_all_loaders_select = client_all_loaders
            test_loader_select = test_loader
            
            
            client_models_dict = self.global_train_once(epoch, global_model, client_all_loaders_select, test_loader_select, FL_params, checkpoints_ls)

            # knowledge discovery
            if epoch == 0:
                new_client_model = copy.deepcopy(init_global_model)
                new_client_model.load_state_dict(client_models_dict[-1])
                domain_shift, class_shift, diff_encoder, diff_classifier, diff_feature = self.detect_shift(init_global_model, new_client_model, self.detect_data_loader)

            std1 = time.time()
            encoder_agg_weights, classifier_agg_weights = self.cal_agg_weights(global_model, client_models_dict[:-1], client_models_dict[-1], data_agg_weights)
            end1 = time.time()
            agg_time = end1 - std1
            print("Agg_weights: ", encoder_agg_weights, classifier_agg_weights)
            # encoder aggregation
            global_encoder_dict = self.fedavg(client_models_dict, encoder_agg_weights)
            # classifier aggregation
            global_classifier_dict = self.fedavg(client_models_dict, classifier_agg_weights)

            global_encoder_dict['classifier.linear.fc.weight'] = global_classifier_dict['classifier.linear.fc.weight']
            global_encoder_dict['classifier.linear.fc.bias'] = global_classifier_dict['classifier.linear.fc.bias']
            global_model.load_state_dict(global_encoder_dict)

            if class_shift == True:
                source_model_dict = self.fedavg(client_models_dict[0:-1], old_agg_weights)#'classifier.linear.fc.weight'[10,2048],'classifier.linear.fc.bias'[10]
                dict_classifier = global_model.classifier.state_dict()
                dict_classifier['linear.fc.weight'] = source_model_dict['classifier.linear.fc.weight']
                dict_classifier['linear.fc.bias'] = source_model_dict['classifier.linear.fc.bias']
                for new_class in self.args.target_domain:
                    dict_classifier['linear.fc.weight'][new_class] = client_models_dict[-1]['classifier.linear.fc.weight'][new_class]
                    dict_classifier['linear.fc.bias'][new_class] = client_models_dict[-1]['classifier.linear.fc.bias'][new_class]
                global_model.classifier.load_state_dict(dict_classifier)
            if epoch < FL_params.global_epoch - 1:
                del client_models_dict
                # torch.cuda.empty_cache()
           
            all_idx = [k for k in range(len(client_all_loaders))]
            client_test_acc = []
            end_time = time.time()
            consume_time = end_time - std_time-agg_time

            for client_idx in all_idx:
                (test_loss, test_acc) = self.test(global_model, test_loader[client_idx], FL_params)
                client_test_acc.append(test_acc)
                result_list.append([epoch, client_idx, test_loss, test_acc, consume_time])
                print("Epoch {} Client {} Test Loss = {}, Test Accuracy = {}".format(epoch, client_idx, test_loss, test_acc))
           
            avg_acc = sum(client_test_acc) / len(client_test_acc)

            print("FL Round = {}, Global Model Accuracy= {}".format(epoch, avg_acc))
        if FL_params.paradigm == 'sequence':
            # path_global = './save_model/sequence_globalmodel_shift_{}_{}+{}.pth'.format(FL_params.shift, FL_params.target_domain, FL_params.source_domain)
            # torch.save(global_model.state_dict(), path_global)
            for idx, param in enumerate(client_models_dict):
                path_client = './save_model/sequence_client{}model_shift_{}.pth'.format(idx, FL_params.shift)
                torch.save(param, path_client)
        del client_models_dict

        df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id','Test_loss', 'Test_acc', 'Time'])
       
        df.to_csv('./results/Acc_loss_{}_shift_{}_{}+{}.csv'.format(FL_params.paradigm, FL_params.shift, FL_params.target_domain, FL_params.source_domain))
       
        print(5 * "#" + "  Ours Training End  " + 5 * "#")

        return global_model
 
    
    def local_train(self, idx, model, optimizer, data_loader, FL_params):
        if idx != len(self.client_all_loaders) - 1:
            old_model = copy.deepcopy(model)
            if FL_params.paradigm == 'sequence':
                param = torch.load('./save_model/sequence_client{}model_shift_{}.pth'.format(idx, self.args.shift))
            else:
                param = torch.load('./save_model/pretrained_client{}model_shift_{}_{}+{}.pth'.format(idx, self.args.shift, self.args.target_domain, self.args.source_domain))
            old_model.load_state_dict(param)
            num_epochs = FL_params.local_epoch
        else:
            num_epochs = FL_params.local_epoch+1
        #     old_model = copy.deepcopy(model)
       
        for local_epoch in range(num_epochs):
            criteria = nn.CrossEntropyLoss()
            for batch_idx, (data, target) in enumerate(data_loader):
                optimizer.zero_grad()
                data = data.to(FL_params.device)
                target = target.to(FL_params.device)
                target = target.long()
                _, pred = model(data)

                loss = criteria(pred, target)
                if idx != len(self.client_all_loaders) - 1:
                    prox_reg = 0.0
                    for param, global_param in zip(model.parameters(), old_model.parameters()):
                        prox_reg += 0.3 * torch.norm(param - global_param) ** 2
                
                    loss += prox_reg
                loss.backward()
                optimizer.step()
        return model
    

    def construct_detect_loader(self):
        subset_ls = []
        subset_size_per_class = 10
        user_detect_loader = []
        for c_idx in range(len(self.test_loader)):
            dataset = self.test_loader[c_idx].dataset
            class_indices = {}

            for idx, (_, label) in enumerate(dataset):
                label = label.item() 
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(idx)

            selected_indices = []
            for label, indices in class_indices.items():
                if len(indices) >= subset_size_per_class:
                    selected_indices.extend(torch.randperm(len(indices))[:subset_size_per_class].tolist())
                else:
                    raise ValueError(f"Class {label} is insufficient {subset_size_per_class} ")
                
            subset = torch.utils.data.Subset(dataset, selected_indices)
            subset_ls.append(subset)
            user_detect_loader.append(torch.utils.data.DataLoader(subset, batch_size=subset_size_per_class))

        detect_data_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(subset_ls), batch_size=subset_size_per_class)
        
        self.user_detect_loader = user_detect_loader
        self.detect_data_loader = detect_data_loader
        
        return detect_data_loader


    def detect_shift(self, old_model, new_model, datasets):
        
        # datasets_loader = DataLoader(datasets, batch_size=16, shuffle=False)
        datasets_loader = datasets

        diff_feature = 0
        n = 0
        for batch_idx, (x, target) in enumerate(datasets_loader):
            x, target = x.to(self.args.device), target.to(self.args.device)

            features_new, _ = new_model(x)
            features_old, _ = old_model(x)
            
            pdist = nn.PairwiseDistance(p=1)
            diff = pdist(features_old, features_new)
            gap = torch.norm(diff)
            diff_feature += gap
            n += 1
        diff_feature = diff_feature / n

        new_c_encoder_dict = new_model.encoder.state_dict()
        new_c_classifier_dict = new_model.classifier.state_dict()
        old_encoder_dict = old_model.encoder.state_dict()
        old_classifier_dict = old_model.classifier.state_dict()
        diff_encoder = 0
        diff_classifier = 0
        for key in new_c_encoder_dict.keys():
            diff_encoder += torch.norm(new_c_encoder_dict[key].float() - old_encoder_dict[key].float())
        diff_encoder = diff_encoder / len(new_c_encoder_dict.keys())
        for key in new_c_classifier_dict.keys():
            diff_classifier += torch.norm(new_c_classifier_dict[key].float() - old_classifier_dict[key].float())
        diff_classifier = diff_classifier / len(new_c_classifier_dict.keys())
        # print('Encoder diff: {}, Feature diff: {}, Classifier diff: {}'.format(diff_encoder, diff_feature, diff_classifier))
     
        if self.args.data == 'digitfive':
            feature_T = 1000
            classifier_T = 0.25
            if diff_feature >= feature_T:
                if diff_classifier >= classifier_T:
                    domain_shift = False
                    class_shift = True
                else:
                    domain_shift = True
                    class_shift = False
            else:
                domain_shift = False
                class_shift = False
        elif self.args.data == 'amazonreview':
            feature_T = 400
            if diff_feature >= feature_T:
                domain_shift = True
                class_shift = False
            else:
                domain_shift = False
                class_shift = False
        return domain_shift, class_shift, diff_encoder, diff_classifier, diff_feature
    

    def cal_agg_weights(self, model, old_model_dict_ls, new_model_dict, agg_weights):
        new_user_model = copy.deepcopy(model)
        new_user_model.load_state_dict(new_model_dict)
        diff_encoder_ls = []
        diff_classifier_ls = []
        diff_feature_ls = []
        for model_dict in old_model_dict_ls:
            old_user_model = copy.deepcopy(model)
            old_user_model.load_state_dict(model_dict)
            domain_shift, class_shift, diff_encoder, diff_classifer, diff_feature = self.detect_shift(old_user_model, new_user_model, self.detect_data_loader)
            diff_encoder_ls.append(diff_encoder)
            diff_classifier_ls.append(diff_classifer)
            diff_feature_ls.append(diff_feature)
        
        if domain_shift == True:
            encoder_agg_weights = [i for i in range(len(agg_weights))]
            classifier_agg_weights = [i for i in range(len(agg_weights))]
            distri_weights = 1-agg_weights[-1]
            diff_feature_ls = [1 / (x + 1) for x in diff_feature_ls]
            diff_feature_ls = [x / sum(diff_feature_ls) for x in diff_feature_ls]
            for i in range(len(agg_weights)-1):
                encoder_agg_weights[i] = distri_weights * diff_feature_ls[i].item()

            diff_classifier_ls = [1 / (x + 1) for x in diff_classifier_ls]
            diff_classifier_ls = [x / sum(diff_classifier_ls) for x in diff_classifier_ls]
            for i in range(len(agg_weights)-1):
                classifier_agg_weights[i] = distri_weights * diff_classifier_ls[i].item()
                    
        elif class_shift == True:
            encoder_agg_weights = [i for i in range(len(agg_weights))]
            classifier_agg_weights = copy.deepcopy(agg_weights)
            classifier_agg_weights[-1] = 0
            distri_weights = 1-agg_weights[-1]
            diff_feature_ls = [1 / (x + 1) for x in diff_feature_ls]
            diff_feature_ls = [x / sum(diff_feature_ls) for x in diff_feature_ls]
            for i in range(len(agg_weights)-1):
                encoder_agg_weights[i] = distri_weights * diff_feature_ls[i].item()

            for i in range(len(agg_weights)-1):
                classifier_agg_weights[i] = classifier_agg_weights[i]/sum(agg_weights[0:-1])
            
        
        else:
            encoder_agg_weights = agg_weights
            classifier_agg_weights = agg_weights
        return encoder_agg_weights, classifier_agg_weights
