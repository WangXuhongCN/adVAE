# from torchvision import datasets
import numpy as np
import scipy.io as scio

def load_tab_data(data_path, dataset_name):
    data = scio.loadmat(data_path+dataset_name+'.mat')
    dataset = data['X']
    label = data['y']
    if dataset_name=='letter':
        sample_dim=32
        train_data=dataset[:1200]
        train_label=label[:1200]
        test_data=dataset[1200:]
        test_label=label[1200:]
        rep_dim=11
    if dataset_name=='pendigits':
        sample_dim=16
        anomaly_idx=np.where(label==1)[0]
        normal_idx=np.where(label==0)[0]
        test_idx=np.concatenate((normal_idx[:1343],anomaly_idx)) 
        train_idx=normal_idx[1343:]
        train_data=dataset[train_idx]
        train_label=label[train_idx]
        test_data=dataset[test_idx]
        test_label=label[test_idx]
        rep_dim=5
    if dataset_name=='satellite':
        sample_dim=36
        anomaly_idx=np.where(label==1)[0]
        normal_idx=np.where(label==0)[0]
        test_idx=np.concatenate((normal_idx[:1080],anomaly_idx)) 
        train_idx=normal_idx[1080:]
        train_data=dataset[train_idx]
        train_label=label[train_idx]
        test_data=dataset[test_idx]
        test_label=label[test_idx]
        rep_dim=12
    if dataset_name=='cardio':
        sample_dim=21
        train_data=dataset[:-507]
        train_label=label[:-507]
        test_data=dataset[-507:]
        test_label=label[-507:]
        rep_dim=7
    if dataset_name=='optdigits':
        sample_dim=64
        train_data=dataset[:-1163]
        train_label=label[:-1163]
        test_data=dataset[-1163:]
        test_label=label[-1163:] 
        rep_dim=16

    return train_data,train_label,test_data,test_label,sample_dim,rep_dim