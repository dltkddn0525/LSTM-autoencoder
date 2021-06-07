import pandas as pd
import os
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from datetime import datetime

def preprocess(path):
    '''
    Input : Path to dataset directory
    Output : Dataframe of Normal train&val, Abnormal(test)
    '''
    # AMPds2
    if 'AMPds2' in path:
        print("Loading AMPds2 dataset")
        df = pd.read_csv(path)
        # Drop unneccessary columns
        if 'Electricity' in path:
            df = df.drop(["Unnamed: 0","V","I","f","DPF","APF","Pt","Q","Qt","S","St"],axis=1)
        else:
            df = df.drop(["Unnamed: 0","counter","inst_rate"],axis=1)

        df['date'] = df['unix_ts'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
        del df['unix_ts']
        df = df.set_index('date')

        # Split Train+vali : Test = 8:2
        normal = df.iloc[:int(len(df)*0.8)]
        abnormal = df.iloc[int(len(df)*0.8):]

        # delete abnormal data in train+val
        normal = normal[normal['label']==0]
        
        # Split Train : Vali = 7:3
        normal_trn = normal.iloc[:int(len(normal)*0.7)]
        normal_val = normal.iloc[int(len(normal)*0.7):]

        # Compute normalize statistics
        mean = normal_trn.mean()[0]
        std = normal_trn.std()[0]

    # SWaT
    elif 'SWaT' in path:
        print("Loading SWaT dataset")
        normal = pd.read_csv(os.path.join(path, 'SWaT_Dataset_Normal_v1.csv'))
        abnormal = pd.read_csv(os.path.join(path, 'SWaT_Dataset_Attack_v0.csv'))

        # Drop first 6 hours
        drop_idx = np.arange(19800)
        normal = normal.drop(drop_idx)

        # Set index column
        normal['date'] = pd.to_datetime(normal[' Timestamp'])
        del normal[' Timestamp']
        normal = normal.set_index('date')
        abnormal['date'] = pd.to_datetime(abnormal[' Timestamp'])
        del abnormal[' Timestamp']
        abnormal = abnormal.set_index('date')

        # Change Normal -> 0, Attack -> 1
        normal["Normal/Attack"]=normal["Normal/Attack"].replace(["Normal","Attack"],[0,1])
        abnormal["Normal/Attack"]=abnormal["Normal/Attack"].replace(["Normal","Attack","A ttack"],[0,1,1])

        # Split 7:3, MLE on normal_val
        normal_trn = normal.iloc[:int(len(normal)*0.7)]
        normal_val = normal.iloc[int(len(normal)*0.7):]

        # Compute normalize statistics
        mean = normal_trn.mean()
        std = normal_trn.std()

        # Drop std=0 column (51->37, drop 14 column)
        col_idx = np.where(std==0)
        normal_trn = normal_trn.drop(normal_trn.columns[col_idx[0][:-1]],axis=1)
        normal_val = normal_val.drop(normal_val.columns[col_idx[0][:-1]],axis=1)
        abnormal = abnormal.drop(abnormal.columns[col_idx[0][:-1]],axis=1)

        # recompute mean & std
        mean = normal_trn.mean()[:-1]
        std = normal_trn.std()[:-1]

    else:
        raise NotImplementedError
    # Input dim 
    input_dim = len(normal_trn.columns)-1

    return normal_trn, normal_val, abnormal, np.array(mean), np.array(std), input_dim

class TimeSeriesDataset(Dataset):
    def __init__(self,df,mean,std,seq_length=120,shift_length=10):
        self.df = df
        self.mean = mean
        self.std = std
        self.seq_length = seq_length
        self.shift_length = shift_length

        # Normalize
        label_col = self.df.columns[-1]
        if label_col == "Normal/Attack":
            sensor_columns = [item for item in self.df.columns if not 'Normal/Attack' in item]
            self.df[sensor_columns] = (self.df[sensor_columns]-self.mean)/self.std
        elif label_col == "label":
            sensor_columns = [item for item in self.df.columns if not 'label' in item]
            self.df[sensor_columns] = (self.df[sensor_columns]-self.mean)/self.std

        # Data, Label
        self.data = self.df[sensor_columns].values.tolist()
        self.label = self.df[label_col].values.tolist()

        # make data index
        self.indices = np.arange(0,len(self.df)-self.seq_length+1,self.shift_length)

        print(f"Dataset initialized. Number of sequence : {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,idx):
        index = self.indices[idx]
        data = self.data[index:index+self.seq_length]
        label = self.label[index:index+self.seq_length]

        return torch.Tensor(data),torch.LongTensor(label)



if __name__ == '__main__':
    path = '/daintlab/data/SWaT'
    normal_trn, normal_val, abnormal, mean, std, input_dim = preprocess(path)
    train_dataset = TimeSeriesDataset(normal_trn, mean,std)
    val_dataset = TimeSeriesDataset(normal_val, mean,std)
    test_dataset = TimeSeriesDataset(abnormal, mean,std,shift_length=120)
    import ipdb;ipdb.set_trace()
    train_loader = DataLoader(train_dataset,batch_size=2, shuffle=False)
    data,label = next(iter(train_loader))
    import ipdb;ipdb.set_trace()
