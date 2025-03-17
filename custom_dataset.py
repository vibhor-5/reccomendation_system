import torch
import pandas as pd
import random 
from copy import deepcopy
from torch.utils.data import DataLoader,Dataset

class custom_dataset(Dataset):
    def __init__(self,users,items,ratings):
        super().__init__()
        
        self.users=users
        self.items=items
        self.ratings=ratings

    def __getitem__(self, index):
        return self.users[index],self.items[index],self.ratings[index]
    
    def __len__(self):
        return self.users.size(0)
    
class sample_loader:
    def __init__(self,data:pd.DataFrame,num_negatives:int):
        self.data=data
        self.num_negatives=num_negatives
        self.users_pool=set(data["userId"])
        self.movies_pool=set(data["movieId"])

        self.pre_processed_ratings=self._binarize(data)
        self.negative_samples=self.sample_negatives(self.data[["userId","movieId"]],self.num_negatives)
        self.train_data,self.test_data=self.train_test_split(self.pre_processed_ratings)

    def _binarize(self,ratings:pd.DataFrame):
        ratings=deepcopy(ratings)
        ratings.loc[ratings['rating']>0,"rating"]=1.0
        return ratings
    
    def sample_negatives(self,data:pd.DataFrame,num_negatives:int):
        # Group by userId and get set of movies each user interacted with
        processed_data = data.groupby('userId')["movieId"].apply(set).reset_index()
        # Rename the column for clarity
        processed_data = processed_data.rename(columns={"movieId": "interacted_items"})
        
        # For each user, find items they haven't interacted with
        processed_data["negative_items"] = [self.movies_pool - x for x in processed_data["interacted_items"]]
        
        # Sample 99 negative items for test evaluation
        processed_data["all_negative_samples"] = [random.sample(list(x), min(99, len(x))) for x in processed_data["negative_items"]]
        
        # Sample num_negatives items for training
        processed_data["negative_samples"] = [random.sample(list(x), min(num_negatives, len(x))) for x in processed_data["negative_items"]]
        
        return processed_data[["userId", "negative_samples", "all_negative_samples"]]    
    
    def train_test_split(self,data:pd.DataFrame):
        data["ranks"]=data.groupby("userId")["timestamp"].rank(method='first',ascending=False)
        train_data=data[data["ranks"]>1]
        test_data=data[data["ranks"]==1]
        assert train_data["userId"].nunique()==test_data["userId"].nunique()
        return train_data[["userId", "movieId", "rating"]], test_data[["userId", "movieId", "rating"]]
    
    def data_train_custom(self, batch_size):
        user, item, rating = [], [], []
        added_negative = set()
        
        for row in self.train_data.itertuples():
            # Add positive samples
            user.append(int(row.userId))
            item.append(int(row.movieId))
            rating.append(float(row.rating))
            
            # Add negative samples (once per user)
            if row.userId not in added_negative:
                # Get negative samples for this user
                user_neg_samples = self.negative_samples.loc[
                    self.negative_samples["userId"] == row.userId, 
                    "negative_samples"
                ].iloc[0]
                
                # Add each negative sample
                for neg_item in user_neg_samples:
                    user.append(int(row.userId))
                    item.append(int(neg_item))
                    rating.append(float(0))
                    
                # Mark that we've added negatives for this user
                added_negative.add(row.userId)
        
        final_dataset = custom_dataset(
            torch.LongTensor(user),
            torch.LongTensor(item),
            torch.FloatTensor(rating)
        )
        return DataLoader(final_dataset, batch_size=batch_size, shuffle=True)

    def evaluation_data(self):
        test_df=pd.merge(self.test_data,self.negative_samples,on="userId")
        test_user,test_item,negative_test_user,negative_test_item=[],[],[],[]
        for row in test_df.itertuples():
            test_user.append(int(row.userId))
            test_item.append(int(row.movieId))
            for i in range(len(row.all_negative_samples)):
                negative_test_user.append(row.userId)
                negative_test_item.append(row.all_negative_samples[i])
        return (torch.LongTensor(test_user),torch.LongTensor(test_item),torch.LongTensor(negative_test_user),torch.LongTensor(negative_test_item))
    