import torch
import pandas as pd
from training_pipeline import pipeline
from torch import nn
from utils import resume_checkpoint,use_gpu

class ncfmodel(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.num_user=config["num_user"]
        self.num_items=config["num_items"]
        self.latent_dim_mlp=config["latent_dim_mlp"]
        self.latent_dim_mf=config["latent_dim_mf"]
        self.hidden_layers=config["hidden_layers"]

        self.user_embed_mlp=nn.Embedding(self.num_user,self.latent_dim_mlp)
        self.item_embed_mlp=nn.Embedding(self.num_items,self.latent_dim_mlp)
        self.user_embed_mf=nn.Embedding(self.num_user,self.latent_dim_mf)
        self.item_embed_mf=nn.Embedding(self.num_items,self.latent_dim_mf)

        layers=[]
        for i,j in zip(self.hidden_layers[:-1],self.hidden_layers[1:]):
            layers.append(nn.Linear(i,j))
            layers.append(nn.ReLU())
        self.fc_layers=nn.Sequential(*layers)
        self.logits=nn.Linear(self.hidden_layers[-1]+self.latent_dim_mf,1)
        self.logistic=nn.Sigmoid()

        if config["init_gaussian"]:
            for module in self.modules():
                if isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                elif isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self,user_indices,item_indices):
        user_embedding_mlp=self.user_embed_mlp(user_indices)
        item_embedding_mlp=self.item_embed_mlp(item_indices)
        user_embedding_mf=self.user_embed_mf(user_indices)
        item_embedding_mf=self.item_embed_mf(item_indices)

        mlp=torch.cat((user_embedding_mlp,item_embedding_mlp),dim=-1)
        mf=torch.mul(user_embedding_mf,item_embedding_mf)

        mlp_processed=self.fc_layers(mlp)

        concatenated=torch.cat((mlp_processed,mf),dim=-1)
        logit_pred=self.logits(concatenated)
        pred=self.logistic(logit_pred)
        return pred
    
    def init_weight(self):
        pass

    def pretrained_weights(self):
        config=self.config
        model= ncfmodel(config)
        if config["use_cuda"]:
            model.cuda()
        if config["use_metal"]:
            model.to(torch.device("mps"))
        resume_checkpoint(model,config["pretrained_dir"],config["device_id"])
        
        self.user_embed_mlp.weight.data=model.user_embed_mlp.weight.data
        self.item_embed_mlp.weight.data=model.item_embed_mlp.weight.data 
        self.user_embed_mf.weight.data=model.user_embed_mf.weight.data
        self.item_embed_mf.weight.data=model.item_embed_mf.weight.data
        for self_l,model_l in zip(self.fc_layers,model.fc_layers):
            if isinstance(self_l,nn.Linear) and isinstance(model_l,nn.Linear):
                self_l.weight.data=model_l.weight.data
        self.logits.weight.data=model.logits.weight.data
        self.logits.bias.data=model.logits.bias.data

    
class ncf_pipeline(pipeline):
    def __init__(self, config):
        self.model=ncfmodel(config=config)
        if config["use_cuda"]:
            use_gpu(True,device_id=config["device_id"])
            self.model.cuda()
        if config["use_metal"]:
            self.model.to(torch.device("mps"))
            
        super().__init__(config)  
        print(self.model)

        if config["pretrain"]:
            self.model.pretrained_weights()
            