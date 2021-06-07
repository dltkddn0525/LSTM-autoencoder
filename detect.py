import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import preprocess, SWaTDataset
from model import LSTMAutoEncoder

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
    


if __name__ == '__main__':
    path = '/daintlab/data/SWaT'
    normal_trn, normal_val, abnormal, mean, std, input_dim = preprocess(path)
    val_dataset = SWaTDataset(normal_val, mean,std,
                            seq_length=60,shift_length=60)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_dataset = SWaTDataset(abnormal, mean,std,
                            seq_length=60,shift_length=60)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False,
                            num_workers=4, pin_memory=True)
    model = LSTMAutoEncoder(input_dim=input_dim,
                        hidden_dim=128,
                        num_layers=2).cuda()
    #model.load_state_dict('./')
    
    criterion = torch.nn.L1Loss(reduction='none').cuda()

    mean, cov = estimate(model,val_loader,criterion)
    recon_err_list, score_list, label_list = anomaly_detect(model,test_loader,criterion,mean,cov)
    import ipdb;ipdb.set_trace()