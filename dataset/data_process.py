from dataset.DigitFive import digit5_dataset_read
from dataset.AmazonReview import amazon_dataset_read
import yaml
from torch.utils.data import DataLoader, DataLoader, Dataset, TensorDataset
import numpy as np
import torch
import random

def dirichlet_distribution(args, data, num_clients):
    client_data_indices = [[] for _ in range(num_clients)]
    for k in range(args.num_classes):
        idx_k = [i for i, label in enumerate(data) if label == k]
        np.random.shuffle(idx_k)
        num_samples_per_client = len(idx_k) // num_clients
        remainder = len(idx_k) % num_clients
        # proportions = np.random.dirichlet(np.repeat(args.alpha, num_clients))

        for client_id in range(num_clients):
            if client_id < num_clients - 1:
                num_samples = num_samples_per_client
                idxes = random.sample(idx_k, num_samples)
                client_data_indices[client_id].extend(idxes)
                idx_k = list(set(idx_k) - set(idxes))
            else:
                num_samples = num_samples_per_client + remainder
                idxes = random.sample(idx_k, num_samples)
                client_data_indices[client_id].extend(idxes)
    return client_data_indices

def data_distri(args):
    train_loaders = []
    test_loaders = []
    

    if args.shift == 'mild':
        '''
        将args.mild_domain的数据在user间分配, 其中args.target_domain类别的数据分配给目标域客户端(客户端0), 其他数据按照狄利克雷函数分配给源域客户端
        '''
        # 加载args.mild_domain数据集
        args.num_classes = 10
        train_dloader, test_dloader = digit5_dataset_read(args.base_path, args.mild_domain, args.batch_size)
        args.source_domain = [i for i in range(args.num_classes) if i not in args.target_domain]
        # 选择args.mild_target_classes类别数据
        from collections import defaultdict
        target_indices_by_class = defaultdict(list)
        
        target_indices = [i for i, (_, label) in enumerate(train_dloader.dataset) if label in args.target_domain]
        target_indices_test = [i for i, (_, label) in enumerate(test_dloader.dataset) if label in args.target_domain]
        # 将args.target_domain类别数据分配给目标域客户端
        remaining_indices = [i for i, (_, label) in enumerate(train_dloader.dataset) if label in args.source_domain]
        remaining_indices_test = [i for i, (_, label) in enumerate(test_dloader.dataset) if label in args.source_domain]
       
        # 从train_dloader中选择remaining_indices的数据
        remaining_train_subset_x = []
        remaining_train_subset_y = []
        target_train_subset_x = []
        target_train_subset_y = []
        remaining_test_subset_x = []
        remaining_test_subset_y = []
        target_test_subset_x = []
        target_test_subset_y = []
        for i, (image, label) in enumerate(train_dloader.dataset):
            if i in remaining_indices:
                remaining_train_subset_x.append(image)
                remaining_train_subset_y.append(label)
        for j, (image, label) in enumerate(test_dloader.dataset):
            if j in remaining_indices_test:
                remaining_test_subset_x.append(image)
                remaining_test_subset_y.append(label)
            else:
                target_test_subset_x.append(image)
                target_test_subset_y.append(label)
        
        # 将剩余数据按照狄利克雷函数分配给源域客户端
        train_client_indices = dirichlet_distribution(args, remaining_train_subset_y, args.num_user - len(args.new_users))
        test_client_indices = dirichlet_distribution(args, remaining_test_subset_y, args.num_user - len(args.new_users))

        for client_id in range(0, args.num_user-len(args.new_users)):
            train_subset_x = []
            train_subset_y = []
            test_subset_x = []
            test_subset_y = []
            for idx in train_client_indices[client_id]:
                train_subset_x.append(remaining_train_subset_x[idx])
                train_subset_y.append(remaining_train_subset_y[idx])
            for idx in test_client_indices[client_id]:
                test_subset_x.append(remaining_test_subset_x[idx])
                test_subset_y.append(remaining_test_subset_y[idx])
            train_subset_x = torch.tensor(np.array(train_subset_x))
            train_subset_y = torch.tensor(np.array(train_subset_y))
            test_subset_x = torch.tensor(np.array(test_subset_x))
            test_subset_y = torch.tensor(np.array(test_subset_y))

            train_loaders.append(DataLoader(TensorDataset(train_subset_x, train_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))
            test_loaders.append(DataLoader(TensorDataset(test_subset_x, test_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))

        for i, (_, label) in enumerate(train_dloader.dataset):
            if label in args.target_domain:
                target_indices_by_class[label].append(i)
        
        # 计算源域中每个客户端的平均数据量
        average_source_data_size = len(remaining_train_subset_x) // (args.num_user - len(args.new_users))

        # 调整目标域客户端的数据量
        adjusted_target_indices = []
        for label, indices in target_indices_by_class.items():
            np.random.shuffle(indices)
            # 计算每个类别应保留的样本数量
            num_samples_per_class = min(len(indices), average_source_data_size // len(args.target_domain))
            adjusted_target_indices.extend(indices[:num_samples_per_class])

        for i, (image, label) in enumerate(train_dloader.dataset):
            if i in adjusted_target_indices:
                target_train_subset_x.append(image)
                target_train_subset_y.append(label)
               
        target_train_subset_x = torch.tensor(np.array(target_train_subset_x))
        target_train_subset_y = torch.tensor(np.array(target_train_subset_y))
        target_test_subset_x = torch.tensor(np.array(target_test_subset_x))
        target_test_subset_y = torch.tensor(np.array(target_test_subset_y))

        train_loaders.append(DataLoader(TensorDataset(target_train_subset_x, target_train_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))
        test_loaders.append(DataLoader(TensorDataset(target_test_subset_x, target_test_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))

        print("target domain {} loaded finished".format(args.target_domain))
            
    elif args.shift == 'medium':
        '''
        将args.medium_target_domain的数据分配给目标域客户端(客户端0), args.source_domain的数据按照狄利克雷函数分配给源域客户端
        '''
        if args.data == "digitfive":
            args.num_classes = 10
            # 加载args.medium_target_domain数据集
            target_train_dloader, target_test_dloader = digit5_dataset_read(args.base_path, args.target_domain, args.batch_size)
            # 加载args.medium_source_domain数据集
            source_train_dloader, source_test_dloader = digit5_dataset_read(args.base_path, args.source_domain, args.batch_size)
        elif args.data == "amazonreview":
            args.num_classes = 2
            target_train_dloader, target_test_dloader = amazon_dataset_read(args.base_path, args.target_domain, args.batch_size)
            source_train_dloader, source_test_dloader = amazon_dataset_read(args.base_path, args.source_domain, args.batch_size)
        
        remaining_train_subset_x = []
        remaining_train_subset_y = []
        target_train_subset_x = []
        target_train_subset_y = []
        remaining_test_subset_x = []
        remaining_test_subset_y = []
        target_test_subset_x = []
        target_test_subset_y = []
        for i, (image, label) in enumerate(source_train_dloader.dataset):
            remaining_train_subset_x.append(image)
            remaining_train_subset_y.append(label)
        for j, (image, label) in enumerate(source_test_dloader.dataset):
            remaining_test_subset_x.append(image)
            remaining_test_subset_y.append(label)
      
        for j, (image, label) in enumerate(target_test_dloader.dataset):
            target_test_subset_x.append(image)
            target_test_subset_y.append(label)

        # 将源域数据按照狄利克雷函数分配给其他客户端
        train_client_indices = dirichlet_distribution(args, remaining_train_subset_y, args.num_user - len(args.new_users))
        test_client_indices = dirichlet_distribution(args, remaining_test_subset_y, args.num_user - len(args.new_users))

        for client_id in range(0, args.num_user-len(args.new_users)):
            train_subset_x = []
            train_subset_y = []
            test_subset_x = []
            test_subset_y = []
            for idx in train_client_indices[client_id]:
                train_subset_x.append(remaining_train_subset_x[idx])
                train_subset_y.append(remaining_train_subset_y[idx])
            for idx in test_client_indices[client_id]:
                test_subset_x.append(remaining_test_subset_x[idx])
                test_subset_y.append(remaining_test_subset_y[idx])
            train_subset_x = torch.tensor(np.array(train_subset_x))
            train_subset_y = torch.tensor(np.array(train_subset_y))
            test_subset_x = torch.tensor(np.array(test_subset_x))
            test_subset_y = torch.tensor(np.array(test_subset_y))

            train_loaders.append(DataLoader(TensorDataset(train_subset_x, train_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))
            test_loaders.append(DataLoader(TensorDataset(test_subset_x, test_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))

        # 计算源域中每个客户端的平均数据量
        average_source_data_size = len(source_train_dloader.dataset) // (args.num_user - 1)

        # 调整目标域客户端的数据量
        from collections import defaultdict
        target_indices_by_class = defaultdict(list)
        for i, (_, label) in enumerate(target_train_dloader.dataset):
            target_indices_by_class[label].append(i)

        adjusted_target_indices = []
        for label, indices in target_indices_by_class.items():
            np.random.shuffle(indices)
            num_samples_per_class = min(len(indices), average_source_data_size // len(target_indices_by_class))
            adjusted_target_indices.extend(indices[:num_samples_per_class])

        for i, (image, label) in enumerate(target_train_dloader.dataset):
            if i in adjusted_target_indices:
                target_train_subset_x.append(image)
                target_train_subset_y.append(label)
               
        target_train_subset_x = torch.tensor(np.array(target_train_subset_x))
        target_train_subset_y = torch.tensor(np.array(target_train_subset_y))
        target_test_subset_x = torch.tensor(np.array(target_test_subset_x))
        target_test_subset_y = torch.tensor(np.array(target_test_subset_y))

        train_loaders.append(DataLoader(TensorDataset(target_train_subset_x,target_train_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))
        test_loaders.append(DataLoader(TensorDataset(target_test_subset_x,target_test_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))
            
    elif args.shift == 'strong':
        if args.data == "digitfive":
            domains = ['mnistm', 'mnist', 'syn', 'usps', 'svhn']
            # [0]: target dataset, target backbone, [1:-1]: source dataset, source backbone
            # generate dataset for train and target
            print("load target domain {}".format(args.target_domain))
            target_train_dloader, target_test_dloader = digit5_dataset_read(args.base_path,
                                                                            args.target_domain,
                                                                            args.batch_size)
            
            # classifiers.append(Classifier(args.data_parallel).to(args.device))
            domains.remove(args.target_domain)
            args.source_domain = domains
            print("target domain {} loaded".format(args.target_domain))
            # create DigitFive dataset
            print("Source Domains :{}".format(domains))
            for domain in domains:
                # generate dataset for source domain
                source_train_dloader, source_test_dloader = digit5_dataset_read(args.base_path, domain,args.batch_size)
                train_loaders.append(source_train_dloader)
                test_loaders.append(source_test_dloader)
                
                print("Domain {} Preprocess Finished".format(domain))
            train_loaders.append(target_train_dloader)
            test_loaders.append(target_test_dloader)
            args.num_classes = 10
        elif args.data == "amazonreview":
            domains = ["books", "dvd", "electronics", "kitchen"]
            print("load target domain {}".format(args.target_domain))
            target_train_dloader, target_test_dloader = amazon_dataset_read(args.base_path, args.target_domain, args.batch_size)
            # generate MLP and Classifier for target domain
            # classifiers.append(AmazonClassifier(args.data_parallel).to(args.device))
            domains.remove(args.target_domain)
            args.source_domain = domains
            print("target domain {} loaded".format(args.target_domain))
            # create DigitFive dataset
            print("Source Domains :{}".format(domains))
            for domain in domains:
                # generate dataset for source domain
                source_train_dloader, source_test_dloader = amazon_dataset_read(args.base_path, domain, args.batch_size)
                train_loaders.append(source_train_dloader)
                test_loaders.append(source_test_dloader)
                # generate CNN and Classifier for source domain
                # models.append(AmazonMLP(args.data_parallel).to(args.device))
                # classifiers.append(AmazonClassifier(args.data_parallel).to(args.device))
                print("Domain {} Preprocess Finished".format(domain))
            train_loaders.append(target_train_dloader)
            test_loaders.append(target_test_dloader)
            args.num_classes = 2
        args.num_user = len(args.source_domain)+1
    return train_loaders, test_loaders, args

def data_motivation(args):
    train_loaders = []
    test_loaders = []

    if args.data == "digitfive":

        '''
        将args.mild_domain的数据在user间分配, 其中args.target_domain类别的数据分配给目标域客户端(客户端0), 其他数据按照狄利克雷函数分配给源域客户端
        '''
        # 加载args.mild_domain数据集
        args.num_classes = 10
        train_dloader, test_dloader = digit5_dataset_read(args.base_path, args.mild_domain, args.batch_size)
        args.source_domain = [3,4,6,7,8,9]
        # 选择args.mild_target_classes类别数据
        from collections import defaultdict
        target_indices_by_class = defaultdict(list)
        
        # target_indices = [i for i, (_, label) in enumerate(train_dloader.dataset) if label in args.target_domain]
        # target_indices_test = [i for i, (_, label) in enumerate(test_dloader.dataset) if label in args.target_domain]
        # 将args.target_domain类别数据分配给目标域客户端
        remaining_indices = [i for i, (_, label) in enumerate(train_dloader.dataset) if label in args.source_domain]
        remaining_indices_test = [i for i, (_, label) in enumerate(test_dloader.dataset) if label in args.source_domain]
        
        # 从train_dloader中选择remaining_indices的数据
        remaining_train_subset_x = []
        remaining_train_subset_y = []
        target_train_subset_x = []
        target_train_subset_y = []
        remaining_test_subset_x = []
        remaining_test_subset_y = []
        target_test_subset_x = []
        target_test_subset_y = []
        for i, (image, label) in enumerate(train_dloader.dataset):
            if i in remaining_indices:
                remaining_train_subset_x.append(image)
                remaining_train_subset_y.append(label)
        for j, (image, label) in enumerate(test_dloader.dataset):
            if j in remaining_indices_test:
                remaining_test_subset_x.append(image)
                remaining_test_subset_y.append(label)
            else:
                target_test_subset_x.append(image)
                target_test_subset_y.append(label)
        
        # 将剩余数据按照狄利克雷函数分配给源域客户端
        train_client_indices = dirichlet_distribution(args, remaining_train_subset_y, args.num_user - len(args.new_users))
        test_client_indices = dirichlet_distribution(args, remaining_test_subset_y, args.num_user - len(args.new_users))

        for client_id in range(0, args.num_user-len(args.new_users)):
            train_subset_x = []
            train_subset_y = []
            test_subset_x = []
            test_subset_y = []
            for idx in train_client_indices[client_id]:
                train_subset_x.append(remaining_train_subset_x[idx])
                train_subset_y.append(remaining_train_subset_y[idx])
            for idx in test_client_indices[client_id]:
                test_subset_x.append(remaining_test_subset_x[idx])
                test_subset_y.append(remaining_test_subset_y[idx])
            train_subset_x = torch.tensor(np.array(train_subset_x))
            train_subset_y = torch.tensor(np.array(train_subset_y))
            test_subset_x = torch.tensor(np.array(test_subset_x))
            test_subset_y = torch.tensor(np.array(test_subset_y))

            train_loaders.append(DataLoader(TensorDataset(train_subset_x, train_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))
            test_loaders.append(DataLoader(TensorDataset(test_subset_x, test_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))

        for i, (_, label) in enumerate(train_dloader.dataset):
            if label in args.new_classes:
                target_indices_by_class[label].append(i)
        
        # 计算源域中每个客户端的平均数据量
        average_source_data_size = len(remaining_train_subset_x) // (args.num_user - len(args.new_users))

        # 调整目标域客户端的数据量
        adjusted_target_indices = []
        for label, indices in target_indices_by_class.items():
            np.random.shuffle(indices)
            # 计算每个类别应保留的样本数量
            num_samples_per_class = min(len(indices), average_source_data_size // len(args.target_domain))
            adjusted_target_indices.extend(indices[:num_samples_per_class])

        for i, (image, label) in enumerate(train_dloader.dataset):
            if i in adjusted_target_indices:
                target_train_subset_x.append(image)
                target_train_subset_y.append(label)
                
        target_train_subset_x = torch.tensor(np.array(target_train_subset_x))
        target_train_subset_y = torch.tensor(np.array(target_train_subset_y))
        target_test_subset_x = torch.tensor(np.array(target_test_subset_x))
        target_test_subset_y = torch.tensor(np.array(target_test_subset_y))

        train_loaders.append(DataLoader(TensorDataset(target_train_subset_x, target_train_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))
        test_loaders.append(DataLoader(TensorDataset(target_test_subset_x, target_test_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))

        # 加上其他域的数据
        target_domain_train_dloader, target_domain_test_dloader = digit5_dataset_read(args.base_path, args.target_domain, args.batch_size)
        
        # 调整目标域客户端的数据量
        from collections import defaultdict
        target_indices_by_class = defaultdict(list)
        for i, (_, label) in enumerate(target_domain_train_dloader.dataset):
            if label in args.source_domain:
                target_indices_by_class[label].append(i)

        adjusted_target_indices = []
        target_train_domain_subset_x = []
        target_train_domain_subset_y = []
        target_test_domain_subset_x = []
        target_test_domain_subset_y = []
        for label, indices in target_indices_by_class.items():
            np.random.shuffle(indices)
            num_samples_per_class = min(len(indices), average_source_data_size // len(target_indices_by_class))
            adjusted_target_indices.extend(indices[:num_samples_per_class])
        
        for j, (image, label) in enumerate(target_domain_test_dloader.dataset):
            target_test_domain_subset_x.append(image)
            target_test_domain_subset_y.append(label)

        for i, (image, label) in enumerate(target_domain_train_dloader.dataset):
            if i in adjusted_target_indices:
                target_train_domain_subset_x.append(image)
                target_train_domain_subset_y.append(label)
                
        target_train_domain_subset_x = torch.tensor(np.array( target_train_domain_subset_x))
        target_train_domain_subset_y = torch.tensor(np.array(target_train_domain_subset_y))
        target_test_domain_subset_x = torch.tensor(np.array(target_test_domain_subset_x))
        target_test_domain_subset_y = torch.tensor(np.array(target_test_domain_subset_y))

        train_loaders.append(DataLoader(TensorDataset(target_train_domain_subset_x, target_train_domain_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))
        test_loaders.append(DataLoader(TensorDataset(target_test_domain_subset_x, target_test_domain_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))

        print("target domain {} loaded finished".format(args.target_domain))
    
    elif args.data == "amazonreview":
        '''
        将args.mild_domain的数据在user间分配, 其中args.target_domain类别的数据分配给目标域客户端(客户端0), 其他数据按照狄利克雷函数分配给源域客户端
        '''
        # 加载args.mild_domain数据集
        args.num_classes = 2
        target_train_dloader, target_test_dloader = amazon_dataset_read(args.base_path, args.target_domain, args.batch_size)
        source_train_dloader, source_test_dloader = amazon_dataset_read(args.base_path, args.source_domain, args.batch_size)
        remaining_train_subset_x = []
        remaining_train_subset_y = []
        target_train_subset_x = []
        target_train_subset_y = []
        remaining_test_subset_x = []
        remaining_test_subset_y = []
        target_test_subset_x = []
        target_test_subset_y = []
        for i, (image, label) in enumerate(source_train_dloader.dataset):
            remaining_train_subset_x.append(image)
            remaining_train_subset_y.append(label)
        for j, (image, label) in enumerate(source_test_dloader.dataset):
            remaining_test_subset_x.append(image)
            remaining_test_subset_y.append(label)
      
        for j, (image, label) in enumerate(target_test_dloader.dataset):
            target_test_subset_x.append(image)
            target_test_subset_y.append(label)

        # 将源域数据按照狄利克雷函数分配给其他客户端
        train_client_indices = dirichlet_distribution(args, remaining_train_subset_y, args.num_user - len(args.new_users))
        test_client_indices = dirichlet_distribution(args, remaining_test_subset_y, args.num_user - len(args.new_users))

        for client_id in range(0, args.num_user-len(args.new_users)):
            train_subset_x = []
            train_subset_y = []
            test_subset_x = []
            test_subset_y = []
            for idx in train_client_indices[client_id]:
                train_subset_x.append(remaining_train_subset_x[idx])
                train_subset_y.append(remaining_train_subset_y[idx])
            for idx in test_client_indices[client_id]:
                test_subset_x.append(remaining_test_subset_x[idx])
                test_subset_y.append(remaining_test_subset_y[idx])
            train_subset_x = torch.tensor(np.array(train_subset_x))
            train_subset_y = torch.tensor(np.array(train_subset_y))
            test_subset_x = torch.tensor(np.array(test_subset_x))
            test_subset_y = torch.tensor(np.array(test_subset_y))

            train_loaders.append(DataLoader(TensorDataset(train_subset_x, train_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))
            test_loaders.append(DataLoader(TensorDataset(test_subset_x, test_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))

        # 计算源域中每个客户端的平均数据量
        average_source_data_size = len(source_train_dloader.dataset) // (args.num_user - 1)

        # 调整目标域客户端的数据量
        from collections import defaultdict
        target_indices_by_class = defaultdict(list)
        for i, (_, label) in enumerate(target_train_dloader.dataset):
            target_indices_by_class[label].append(i)

        adjusted_target_indices = []
        for label, indices in target_indices_by_class.items():
            np.random.shuffle(indices)
            num_samples_per_class = min(len(indices), average_source_data_size // len(target_indices_by_class))
            adjusted_target_indices.extend(indices[:num_samples_per_class])

        for i, (image, label) in enumerate(target_train_dloader.dataset):
            if i in adjusted_target_indices:
                target_train_subset_x.append(image)
                target_train_subset_y.append(label)
               
        target_train_subset_x = torch.tensor(np.array(target_train_subset_x))
        target_train_subset_y = torch.tensor(np.array(target_train_subset_y))
        target_test_subset_x = torch.tensor(np.array(target_test_subset_x))
        target_test_subset_y = torch.tensor(np.array(target_test_subset_y))

        train_loaders.append(DataLoader(TensorDataset(target_train_subset_x,target_train_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))
        test_loaders.append(DataLoader(TensorDataset(target_test_subset_x,target_test_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))

    return train_loaders, test_loaders, args

def data_sequence(args):
    train_loaders = []
    test_loaders = []

    if args.data == "digitfive":

        '''
        将args.mild_domain的数据在user间分配, 其中args.target_domain类别的数据分配给目标域客户端(客户端0), 其他数据按照狄利克雷函数分配给源域客户端
        '''
        # 加载args.mild_domain数据集
        if args.shift == 'mild':
            args.num_user = 7
            args.new_users = [args.num_user-1, args.num_user-2, args.num_user-3]
     
            train_dloader, test_dloader = digit5_dataset_read(args.base_path, args.mild_domain, args.batch_size)
            args.source_domain = [0,1,2,3]
            args.target_domain_ls = [[4,5],[6,7],[8,9]]
            # 选择args.mild_target_classes类别数据
            from collections import defaultdict
            target_indices_by_class = defaultdict(list)
        
            # 将args.target_domain类别数据分配给目标域客户端
            remaining_indices = [i for i, (_, label) in enumerate(train_dloader.dataset) if label in args.source_domain]
            remaining_indices_test = [i for i, (_, label) in enumerate(test_dloader.dataset) if label in args.source_domain]
            
            # 从train_dloader中选择remaining_indices的数据
            remaining_train_subset_x = []
            remaining_train_subset_y = []
            remaining_test_subset_x = []
            remaining_test_subset_y = []
            
            for i, (image, label) in enumerate(train_dloader.dataset):
                if i in remaining_indices:
                    remaining_train_subset_x.append(image)
                    remaining_train_subset_y.append(label)
            for j, (image, label) in enumerate(test_dloader.dataset):
                if j in remaining_indices_test:
                    remaining_test_subset_x.append(image)
                    remaining_test_subset_y.append(label)
            
            # 将剩余数据按照狄利克雷函数分配给源域客户端
            train_client_indices = dirichlet_distribution(args, remaining_train_subset_y, args.num_user - len(args.new_users))
            test_client_indices = dirichlet_distribution(args, remaining_test_subset_y, args.num_user - len(args.new_users))

            
            for client_id in range(0, args.num_user-len(args.new_users)):
                train_subset_x = []
                train_subset_y = []
                test_subset_x = []
                test_subset_y = []
                for idx in train_client_indices[client_id]:
                    train_subset_x.append(remaining_train_subset_x[idx])
                    train_subset_y.append(remaining_train_subset_y[idx])
                for idx in test_client_indices[client_id]:
                    test_subset_x.append(remaining_test_subset_x[idx])
                    test_subset_y.append(remaining_test_subset_y[idx])
                train_subset_x = torch.tensor(np.array(train_subset_x))
                train_subset_y = torch.tensor(np.array(train_subset_y))
                test_subset_x = torch.tensor(np.array(test_subset_x))
                test_subset_y = torch.tensor(np.array(test_subset_y))

                train_loaders.append(DataLoader(TensorDataset(train_subset_x, train_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))
                test_loaders.append(DataLoader(TensorDataset(test_subset_x, test_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))
            
            # 计算源域中每个客户端的平均数据量
            average_source_data_size = len(remaining_train_subset_x) // (args.num_user - len(args.new_users))
            for new_user in range(len(args.new_users)):
                args.target_domain = args.target_domain_ls[new_user]

                target_train_subset_x = []
                target_train_subset_y = []
                target_test_subset_x = []
                target_test_subset_y = []
                for j, (image, label) in enumerate(test_dloader.dataset):
                    if label in args.target_domain:
                        target_test_subset_x.append(image)
                        target_test_subset_y.append(label)

                for i, (_, label) in enumerate(train_dloader.dataset):
                    if label in args.target_domain:
                        target_indices_by_class[label].append(i)

                # 调整目标域客户端的数据量
                adjusted_target_indices = []
                for label, indices in target_indices_by_class.items():
                    np.random.shuffle(indices)
                    # 计算每个类别应保留的样本数量
                    num_samples_per_class = min(len(indices), average_source_data_size // len(args.target_domain))
                    adjusted_target_indices.extend(indices[:num_samples_per_class])

                for i, (image, label) in enumerate(train_dloader.dataset):
                    if i in adjusted_target_indices:
                        target_train_subset_x.append(image)
                        target_train_subset_y.append(label)
            
                        
                target_train_subset_x = torch.tensor(np.array(target_train_subset_x))
                target_train_subset_y = torch.tensor(np.array(target_train_subset_y))
                target_test_subset_x = torch.tensor(np.array(target_test_subset_x))
                target_test_subset_y = torch.tensor(np.array(target_test_subset_y))

                train_loaders.append(DataLoader(TensorDataset(target_train_subset_x, target_train_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))
                test_loaders.append(DataLoader(TensorDataset(target_test_subset_x, target_test_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))
        elif args.shift == 'medium':
            args.source_domain = ['svhn']
            args.target_domain_ls = [['mnist'], ['mnistm'], ['syn'], ['usps']]
            args.num_user = 8
            args.new_users = [args.num_user-1, args.num_user-2, args.num_user-3, args.num_user-4]
            # 加载源域数据集
            source_train_dloader, source_test_dloader = digit5_dataset_read(args.base_path, args.source_domain[0], args.batch_size)
            remaining_train_subset_x = []
            remaining_train_subset_y = []
            remaining_test_subset_x = []
            remaining_test_subset_y = []
          
            for i, (image, label) in enumerate(source_train_dloader.dataset):
                remaining_train_subset_x.append(image)
                remaining_train_subset_y.append(label)
            for j, (image, label) in enumerate(source_test_dloader.dataset):
                remaining_test_subset_x.append(image)
                remaining_test_subset_y.append(label)

            # 将源域数据按照狄利克雷函数分配给其他客户端
            train_client_indices = dirichlet_distribution(args, remaining_train_subset_y, args.num_user - len(args.new_users))
            test_client_indices = dirichlet_distribution(args, remaining_test_subset_y, args.num_user - len(args.new_users))

            for client_id in range(0, args.num_user-len(args.new_users)):
                train_subset_x = []
                train_subset_y = []
                test_subset_x = []
                test_subset_y = []
                for idx in train_client_indices[client_id]:
                    train_subset_x.append(remaining_train_subset_x[idx])
                    train_subset_y.append(remaining_train_subset_y[idx])
                for idx in test_client_indices[client_id]:
                    test_subset_x.append(remaining_test_subset_x[idx])
                    test_subset_y.append(remaining_test_subset_y[idx])
                train_subset_x = torch.tensor(np.array(train_subset_x))
                train_subset_y = torch.tensor(np.array(train_subset_y))
                test_subset_x = torch.tensor(np.array(test_subset_x))
                test_subset_y = torch.tensor(np.array(test_subset_y))

                train_loaders.append(DataLoader(TensorDataset(train_subset_x, train_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))
                test_loaders.append(DataLoader(TensorDataset(test_subset_x, test_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))

            # 计算源域中每个客户端的平均数据量
            average_source_data_size = len(source_train_dloader.dataset) // (args.num_user - len(args.new_users))


            for new_user in range(len(args.new_users)):
                args.target_domain = args.target_domain_ls[new_user][0]
                # 加载args.medium_target_domain数据集
                # 加上其他域的数据
                target_domain_train_dloader, target_domain_test_dloader = digit5_dataset_read(args.base_path, args.target_domain, args.batch_size)
                
                # 调整目标域客户端的数据量
                from collections import defaultdict
                target_indices_by_class = defaultdict(list)
                for i, (_, label) in enumerate(target_domain_train_dloader.dataset):
                    target_indices_by_class[label].append(i)

                adjusted_target_indices = []
                target_train_domain_subset_x = []
                target_train_domain_subset_y = []
                target_test_domain_subset_x = []
                target_test_domain_subset_y = []
                for label, indices in target_indices_by_class.items():
                    np.random.shuffle(indices)
                    num_samples_per_class = min(len(indices), average_source_data_size // len(target_indices_by_class))
                    adjusted_target_indices.extend(indices[:num_samples_per_class])
                
                for j, (image, label) in enumerate(target_domain_test_dloader.dataset):
                    target_test_domain_subset_x.append(image)
                    target_test_domain_subset_y.append(label)

                for i, (image, label) in enumerate(target_domain_train_dloader.dataset):
                    if i in adjusted_target_indices:
                        target_train_domain_subset_x.append(image)
                        target_train_domain_subset_y.append(label)
                        
                target_train_domain_subset_x = torch.tensor(np.array( target_train_domain_subset_x))
                target_train_domain_subset_y = torch.tensor(np.array(target_train_domain_subset_y))
                target_test_domain_subset_x = torch.tensor(np.array(target_test_domain_subset_x))
                target_test_domain_subset_y = torch.tensor(np.array(target_test_domain_subset_y))

                train_loaders.append(DataLoader(TensorDataset(target_train_domain_subset_x, target_train_domain_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))
                test_loaders.append(DataLoader(TensorDataset(target_test_domain_subset_x, target_test_domain_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))

                print("target domain {} loaded finished".format(args.target_domain))
    
    elif args.data == "amazonreview":
        args.source_domain = ['dvd']
        args.target_domain_ls = [['books'], ['electronics'], ['kitchen']]
        args.num_user = 7
        args.new_users = [args.num_user-1, args.num_user-2, args.num_user-3]
        # 加载源域数据集
        source_train_dloader, source_test_dloader = amazon_dataset_read(args.base_path, args.source_domain[0], args.batch_size)
        remaining_train_subset_x = []
        remaining_train_subset_y = []
        remaining_test_subset_x = []
        remaining_test_subset_y = []
        
        for i, (image, label) in enumerate(source_train_dloader.dataset):
            remaining_train_subset_x.append(image)
            remaining_train_subset_y.append(label)
        for j, (image, label) in enumerate(source_test_dloader.dataset):
            remaining_test_subset_x.append(image)
            remaining_test_subset_y.append(label)

        # 将源域数据按照狄利克雷函数分配给其他客户端
        train_client_indices = dirichlet_distribution(args, remaining_train_subset_y, args.num_user - len(args.new_users))
        test_client_indices = dirichlet_distribution(args, remaining_test_subset_y, args.num_user - len(args.new_users))

        for client_id in range(0, args.num_user-len(args.new_users)):
            train_subset_x = []
            train_subset_y = []
            test_subset_x = []
            test_subset_y = []
            for idx in train_client_indices[client_id]:
                train_subset_x.append(remaining_train_subset_x[idx])
                train_subset_y.append(remaining_train_subset_y[idx])
            for idx in test_client_indices[client_id]:
                test_subset_x.append(remaining_test_subset_x[idx])
                test_subset_y.append(remaining_test_subset_y[idx])
            train_subset_x = torch.tensor(np.array(train_subset_x))
            train_subset_y = torch.tensor(np.array(train_subset_y))
            test_subset_x = torch.tensor(np.array(test_subset_x))
            test_subset_y = torch.tensor(np.array(test_subset_y))

            train_loaders.append(DataLoader(TensorDataset(train_subset_x, train_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))
            test_loaders.append(DataLoader(TensorDataset(test_subset_x, test_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))

        # 计算源域中每个客户端的平均数据量
        average_source_data_size = len(source_train_dloader.dataset) // (args.num_user - len(args.new_users))


        for new_user in range(len(args.new_users)):
            args.target_domain = args.target_domain_ls[new_user][0]
            # 加载args.medium_target_domain数据集
            # 加上其他域的数据
            target_domain_train_dloader, target_domain_test_dloader = amazon_dataset_read(args.base_path, args.target_domain, args.batch_size)
            
            # 调整目标域客户端的数据量
            from collections import defaultdict
            target_indices_by_class = defaultdict(list)
            for i, (_, label) in enumerate(target_domain_train_dloader.dataset):
                target_indices_by_class[label].append(i)

            adjusted_target_indices = []
            target_train_domain_subset_x = []
            target_train_domain_subset_y = []
            target_test_domain_subset_x = []
            target_test_domain_subset_y = []
            for label, indices in target_indices_by_class.items():
                np.random.shuffle(indices)
                num_samples_per_class = min(len(indices), average_source_data_size // len(target_indices_by_class))
                adjusted_target_indices.extend(indices[:num_samples_per_class])
            
            for j, (image, label) in enumerate(target_domain_test_dloader.dataset):
                target_test_domain_subset_x.append(image)
                target_test_domain_subset_y.append(label)

            for i, (image, label) in enumerate(target_domain_train_dloader.dataset):
                if i in adjusted_target_indices:
                    target_train_domain_subset_x.append(image)
                    target_train_domain_subset_y.append(label)
                    
            target_train_domain_subset_x = torch.tensor(np.array( target_train_domain_subset_x))
            target_train_domain_subset_y = torch.tensor(np.array(target_train_domain_subset_y))
            target_test_domain_subset_x = torch.tensor(np.array(target_test_domain_subset_x))
            target_test_domain_subset_y = torch.tensor(np.array(target_test_domain_subset_y))

            train_loaders.append(DataLoader(TensorDataset(target_train_domain_subset_x, target_train_domain_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))
            test_loaders.append(DataLoader(TensorDataset(target_test_domain_subset_x, target_test_domain_subset_y), batch_size=args.batch_size, shuffle=True, num_workers=4))

            print("target domain {} loaded finished".format(args.target_domain))

    return train_loaders, test_loaders, args
