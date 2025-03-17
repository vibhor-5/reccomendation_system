from custom_dataset import sample_loader
from ncfmodel import ncf_pipeline
import pandas as pd
import numpy as np
import polars as pl
data_dir='ml-32m/ratings.csv'
# chunksize = 10**3  # Adjust based on memory limits  
# chunks = pd.read_csv(data_dir, chunksize=chunksize,engine='python')
# ratings = pd.concat(chunks, ignore_index=True)
ratings=pl.read_csv(data_dir)
print(ratings.dtypes)
ratings=ratings.to_pandas()
ratings['userId']=ratings['userId'].astype('category').cat.codes.values
ratings['movieId']=ratings['movieId'].astype('category').cat.codes.values
num_user=ratings["userId"].nunique()
num_items=ratings["movieId"].nunique()

config={
    "alias":"neumf",
    "num_epoch":200,
    "batch_size":1024,
    "optimizer":"adam",
    "adam_lr":1e-3,
    "num_user":num_user,
    "num_items":num_items,
    "latent_dim_mf":8,
    "latent_dim_mlp":8,
    "num_negative":4,
    "hidden_layers":[16,64,32,16,8],
    "l2_regularization":0.0,
    'init_gaussian':True,
    'use_metal':True,
    'bachify_eval':True,
    'use_cuda':False,
    'device_id':0,
    'pretrain':False,
    'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model',
    'log_file': 'logs/model.log'
}


data=sample_loader(ratings,4)
eval_data=data.evaluation_data()
model=ncf_pipeline(config)
for epoch in range(config["num_epoch"]):
    print(f"Epoch {epoch}")
    print("-"*80)
    train_loader=data.data_train_custom(config["batch_size"])
    model.train_epoch(train_loader,epoch)
    hit_ratio,ndcg_score=model.evaluate(eval_data,epoch)
    print(f"Hit Ratio: {hit_ratio:.4f}")
    print(f"NDCG Score: {ndcg_score:.4f}")
    model.save(config["alias"],epoch,hit_ratio,ndcg_score)


