import torch 
import pandas as pd
from copy import deepcopy
import logging
import os
from utils import use_optimizer,save_checkpoint
from metrics import metric_at_k

from pathlib import Path

class pipeline:

    def __init__(self,config):
        self.config=config
        self.log_file=config["log_file"]
        self.log_file=Path(self.log_file)
        if not self.log_file.exists():
            self.log_file.touch()
        logging.basicConfig(filename=str(self.log_file),level=logging.INFO,format="%(asctime)s-%(levelname)s-%(message)s")
        logging.info("Log start")
        self.loss_fn=torch.nn.BCELoss()
        self.opt=use_optimizer(self.model,self.config)
        self._metric_at_k=metric_at_k(top_k=10)
    
    def train_single_batch(self,user,item,rating):
        if self.config["use_cuda"]:
            user,item,rating=user.cuda(),item.cuda(),rating.cuda()
        if self.config["use_metal"]:
            device=torch.device("mps")
            user,item,rating=user.to(device),item.to(device),rating.to(device)
        self.opt.zero_grad()
        pred=self.model(user,item)
        loss=self.loss_fn(pred.view(-1),rating)
        loss.backward()
        self.opt.step()
        return loss.item()
    
    def train_epoch(self,train_loader,epoch_no):
        self.model.train()
        total_loss=0
        for batch_no,batch in enumerate(train_loader):
            user,item,rating=batch[0],batch[1],batch[2]
            rating=rating.float()
            loss=self.train_single_batch(user,item,rating)
            total_loss+=loss
            logging.info(f"epoch no-{epoch_no} batch number-{batch_no} loss - {loss}")
        print(f"Epoch {epoch_no} total loss: {total_loss}")

    def evaluate(self,evaluate_data,epoch_no):
        test_user,test_item=evaluate_data[0],evaluate_data[1]
        negative_user,negative_item=evaluate_data[2],evaluate_data[3]
        if self.config["use_cuda"]:
            test_user = test_user.cuda()
            test_item = test_item.cuda()
            negative_user = negative_user.cuda()
            negative_item = negative_item.cuda()
        elif self.config["use_metal"]:
            device = torch.device("mps")
            test_user = test_user.to(device)
            test_item = test_item.to(device)
            negative_user = negative_user.to(device)
            negative_item = negative_item.to(device)
                
        if not self.config["bachify_eval"]:
            test_score=self.model(test_user,test_item)
            negative_score=self.model(negative_user,negative_item)
        else:
            test_score=[]
            negative_score=[]
            batch_size=self.config["batch_size"]
            for left in range(0,len(test_user),batch_size):
                right=min(left+batch_size,len(test_user))
                batch_test_user=test_user[left:right]
                batch_test_item=test_item[left:right]
                test_score.append(self.model(batch_test_user,batch_test_item))
            for left in range(0,len(negative_user),batch_size):
                right=min(left+batch_size,len(negative_user))
                batch_negative_user=negative_user[left:right]
                batch_negative_item=negative_item[left:right]
                negative_score.append(self.model(batch_negative_user,batch_negative_item))
            test_score=torch.concatenate(test_score,dim=0)
            negative_score=torch.concatenate(negative_score,dim=0)
            self._metric_at_k.subjects=[test_user,test_item,test_score,negative_user,negative_item,negative_score]
            hit_ratio,ndcg=self._metric_at_k.hit_ratio(),self._metric_at_k.ndcg()
            print(f"Epoch no . {epoch_no} hit ratio: {hit_ratio}, ndcg: {ndcg}")
            return hit_ratio,ndcg
        
    def save(self,alias,epoch_no,hit_ratio,ndcg):
        model_dir=self.config["model_dir"].format(alias,epoch_no,hit_ratio,ndcg)
        save_checkpoint(self.model,model_dir)




