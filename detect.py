import os
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix,roc_auc_score
from dataset import preprocess, TimeSeriesDataset
from model import LSTMAutoEncoder
from utils import *

def estimate(model, val_loader, criterion):
    '''
    estimate mean & covariance of recon error on validation set
    criterion : absolute error
    return : mean(feature_dim,), covariance(feature dim, feature dim)
    '''
    model.eval()
    recon_err_list = []
    with torch.no_grad():
        for i,(data,label) in enumerate(val_loader):
            data = data.cuda()
            feature_dim = data.shape[-1]

            output = model(data)
            recon_err = criterion(output,data)
            recon_err = recon_err.detach().cpu().numpy()
            recon_err = recon_err.reshape(-1,feature_dim)
            recon_err_list.extend(recon_err)
            print(f"Iter [{i}/{len(val_loader)}] Completed")

    recon_err_list = np.array(recon_err_list)
    mean = recon_err_list.mean(axis=0)
    cov = np.cov(recon_err_list,rowvar=False)
    print(f"Mean shape : {mean.shape}")
    print(f"Covariance shape : {cov.shape}")

    return mean, cov

def cal_anomaly_score(recon_err, mean, cov):
    score_list = []
    for err in recon_err:
        z = err - mean
        if z.shape == (1,):
            score = z*z*cov
        else:
            score = np.matmul(np.matmul(z,cov),z.T)
        score_list.append(score)
    return score_list

def anomaly_detect(model, test_loader, criterion, mean, cov):
    model.eval()
    recon_err_list = []
    score_list = []
    label_list = []
    with torch.no_grad():
        for i,(data,label) in enumerate(test_loader):
            data = data.cuda()
            feature_dim = data.shape[-1]
                        
            output = model(data)
            recon_err = criterion(output,data)
            recon_err = recon_err.detach().cpu().numpy()
            recon_err = recon_err.reshape(-1,feature_dim)
            recon_err_list.extend(recon_err)

            batch_score = cal_anomaly_score(recon_err,mean,cov)
            score_list.extend(batch_score)

            labels = label.detach().cpu().numpy().reshape(-1)
            label_list.extend(labels)

            print(f"Iter [{i}/{len(test_loader)}] Completed")

    return recon_err_list, score_list, label_list

def get_performance(score_list,label_list,threshold):
    y_pred = [int(score>threshold) for score in score_list]
    confusion_mat = confusion_matrix(label_list, y_pred)
    precision = confusion_mat[1,1]/(confusion_mat[0,1]+confusion_mat[1,1])
    recall = confusion_mat[1,1]/(confusion_mat[1,0]+confusion_mat[1,1])
    f1 = 2 * (precision*recall)/(precision+recall)

    auroc = roc_auc_score(label_list,score_list)

    return auroc, precision, recall, f1



if __name__ == '__main__':
    path = '/daintlab/data/AMPds2/labeled/Electricity_WHE_labeled.csv'
    save_path = './ampds2_elec'
    normal_trn, normal_val, abnormal, mean, std, input_dim = preprocess(path)
    val_dataset = TimeSeriesDataset(normal_val, mean,std,
                            seq_length=10,shift_length=10)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_dataset = TimeSeriesDataset(abnormal, mean,std,
                            seq_length=10,shift_length=10)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False,
                            num_workers=4, pin_memory=True)
    model = LSTMAutoEncoder(input_dim=input_dim,
                            hidden_dim=64,
                            attention=False,
                            seq_length=10,
                            num_layers=2).cuda()
    
    model.load_state_dict(torch.load(os.path.join(save_path,'last.pth')))
    criterion = torch.nn.L1Loss(reduction='none').cuda()

    mean, cov = estimate(model,val_loader,criterion)
    recon_err_list, score_list, label_list = anomaly_detect(model,test_loader,criterion,mean,cov)
    #
    threshold = np.mean(score_list)+3*np.std(score_list)
    #
    auroc,precision, recall, f1 = get_performance(score_list,label_list,threshold)
    
    np.save(os.path.join(save_path,'recon_err.npy'),recon_err_list)
    np.save(os.path.join(save_path,'score_list.npy'),score_list)
    np.save(os.path.join(save_path,'label_list.npy'),label_list)
    tst_recon_mean = torch.Tensor(recon_err_list).mean()

    perf_dict = {"auroc":auroc,
                "precision":precision,
                "recall":recall,
                "f1":f1,
                "recon mean":tst_recon_mean}
                
    with open(os.path.join(save_path,'perf_dict.json'),'w') as f:
        json.dump(perf_dict,f)
    train_logger = Logger(os.path.join(save_path,'train.log'))
    val_logger = Logger(os.path.join(save_path,'val.log'))
    draw_loss_curve(train_logger,val_logger,save_path)
    
    import ipdb;ipdb.set_trace()
    mean_err = torch.mean(recon_err_list)
    