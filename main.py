import torch
import os
import argparse
from torch.utils.data import DataLoader
import time
import json

from dataset import preprocess, SWaTDataset
from model import LSTMAutoEncoder
from utils import AverageMeter, Logger,draw_loss_curve
from detect import *
from model2 import RecurrentAutoencoder

parser = argparse.ArgumentParser(description='Anomaly detection')
parser.add_argument('--save_path', default='./result', type=str,
                    help='save path')
parser.add_argument('--data_path', default='/daintlab/data/SWaT', type=str,
                    help='Path to dataset')
parser.add_argument('--seq_length', default=60, type=int,
                    help='Sequence length')
parser.add_argument('--trn_shift_length', default=1, type=int,
                    help='Train shift length')
parser.add_argument('--tst_shift_length', default=60, type=int,
                    help='Test shift length')
parser.add_argument('--hidden_dim', default=128, type=int,
                    help='Latent dimension')
parser.add_argument('--num_layers', default=2, type=int,
                    help='Number of lstm layers')
parser.add_argument('--batch_size', default=1024, type=int,
                    help='batch size')
parser.add_argument('--epoch', default=150, type=int,
                    help='Train Epoch')
parser.add_argument('--lr', default=0.001, type=float,
                    help='Learning rate')
args = parser.parse_args()

def main():
    # Save path
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Get dataset
    assert args.seq_length == args.tst_shift_length

    normal_trn, normal_val, abnormal, mean, std, input_dim = preprocess(args.data_path)

    train_dataset = SWaTDataset(normal_trn, mean,std,
                                seq_length=args.seq_length,shift_length=args.trn_shift_length)
    val_dataset = SWaTDataset(normal_val, mean,std,
                                seq_length=args.seq_length,shift_length=args.tst_shift_length)
    test_dataset = SWaTDataset(abnormal, mean,std,
                                seq_length=args.seq_length,shift_length=args.tst_shift_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Model
    model = LSTMAutoEncoder(input_dim=input_dim,
                            hidden_dim=args.hidden_dim,
                            num_layers=args.num_layers).cuda()

    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-04)
    criterion = torch.nn.MSELoss().cuda()

    # Logger
    train_logger = Logger(os.path.join(save_path,'train.log'))
    val_logger = Logger(os.path.join(save_path,'val.log'))

    # Train & Validation
    for epoch in range(args.epoch):
        train(model,criterion,optimizer,train_loader,train_logger,epoch)
        validate(model,criterion,val_loader,val_logger,epoch)
        torch.save(model.state_dict(),os.path.join(save_path,'last.pth'))

    # Test(anomaly detection)
    # estimate mean&cov on validation set
    l1_criterion = torch.nn.L1Loss(reduction='none').cuda()
    mean, cov = estimate(model,val_loader,l1_criterion)
    # Compute anomaly score on test set
    recon_err_list, score_list, label_list = anomaly_detect(model,test_loader,l1_criterion,mean,cov)
    # Get performance
    threshold = np.mean(score_list)+3*np.std(score_list)
    auroc,precision, recall, f1 = get_performance(score_list,label_list,threshold)
    
    # save recon_err_list, score_list, label_list
    np.save(os.path.join(save_path,'recon_err.npy'),recon_err_list)
    np.save(os.path.join(save_path,'score_list.npy'),score_list)
    np.save(os.path.join(save_path,'label_list.npy'),label_list)
    # save performance
    perf_dict = {"auroc":auroc,
                "precision":precision,
                "recall":recall,
                "f1":f1}
    with open(os.path.join(save_path,'perf_dict.json'),'w') as f:
        json.dump(perf_dict,f)
    # save fig
    draw_loss_curve(train_logger,val_logger,save_path)


def train(model,criterion,optimizer,train_loader,train_logger,epoch):
    model.train()
    trn_loss = AverageMeter()
    iter_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    for i,(data,label) in enumerate(train_loader):
    #for data,label in train_loader.dataset:
        data = data.cuda()
        data_time.update(time.time()-end)
        
        output = model(data)
        loss = criterion(output,data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        trn_loss.update(loss.item())
        iter_time.update(time.time()-end)
        end = time.time()

        if i%10==0:
            print(f"Iteration [{i+1}/{len(train_loader)}] Epoch [{epoch+1}/{args.epoch}] Train Loss : {trn_loss.avg:.4f} Iter Time : {iter_time.avg:.4f} Data Time : {data_time.avg:.4f}")

    train_logger.write([epoch, trn_loss.avg, iter_time.avg, data_time.avg])

def validate(model,criterion,val_loader,val_logger,epoch):
    model.eval()
    val_loss = AverageMeter()

    with torch.no_grad():
        for i,(data,label) in enumerate(val_loader):
            data = data.cuda()
            output = model(data)
            loss = criterion(output,data)

            val_loss.update(loss.item())
            
            if i%10 == 0:
                print(f"Validate Iter [{i+1}/{len(val_loader)}]")
    
    print(f"Epoch : {epoch+1} Validation Loss : {val_loss.avg}")
    val_logger.write([epoch,val_loss.avg])


if __name__ == '__main__':
    main()