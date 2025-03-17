import torch 
import math 

class metric_at_k:
    def __init__(self,top_k):
        self._top_k=top_k
        self._subjects=None
    
    @property
    def top_k(self):
        return self._top_k
    
    @top_k.setter
    def top_k(self,top_k):
        self._top_k=top_k
    
    @property
    def subjects(self):
        return self._subjects
    
    @subjects.setter
    def subjects(self,subjects):
        device=torch.device("cpu")
        subjects=[subject.to(device) for subject in subjects]
        test_user,test_item,test_rating=torch.LongTensor(subjects[0]),torch.LongTensor(subjects[1]),torch.FloatTensor(subjects[2])
        neg_user,neg_item,neg_rating=torch.LongTensor(subjects[3]),torch.LongTensor(subjects[4]),torch.FloatTensor(subjects[5])

        test_rating = test_rating.squeeze()
        neg_rating = neg_rating.squeeze()

        user_full=torch.cat((neg_user,test_user))
        item_full=torch.cat((neg_item,test_item))
        rating_full=torch.cat((neg_rating,test_rating))

        test_set=torch.zeros_like(user_full,dtype=torch.bool)
        test_item_indices=torch.arange(len(test_user))+len(neg_user)
        test_set[test_item_indices]=True

        full_data=torch.stack((user_full,item_full,rating_full),dim=1)
        _,sorted_indices=torch.sort(rating_full,descending=True)
        full_data=full_data[sorted_indices]
        test_set=test_set[sorted_indices]

        unique_user,unique_user_indices=torch.unique(full_data[:,0],return_inverse=True)
        ranks=torch.zeros_like(rating_full,dtype=torch.float32)
        for i,user in enumerate(unique_user):
            user_indices=(full_data[:,0]==user).nonzero(as_tuple=True)[0]
            ranks[user_indices]=torch.arange(1,len(user_indices)+1,dtype=torch.float32)

        self._subjects=torch.cat((full_data,ranks.unsqueeze(1)),dim=1)
        self._test_set=test_item
    
    def hit_ratio(self):
        full, top_k = self._subjects, self._top_k
        
        # Get unique users
        unique_users = full[:,0].unique()
        hit_count = 0
        
        for user in unique_users:
            # Get all items for this user
            user_items = full[full[:,0] == user]
            
            # Get top-k items for this user
            user_top_k = user_items[user_items[:,3] <= top_k]
            
            # Get test items for this user
            user_test_items = []
            for i, item in enumerate(user_items[:,1]):
                if item in self._test_set:
                    user_test_items.append(item)
            
            # Check if any test item is in top-k
            hit = False
            for test_item in user_test_items:
                if test_item in user_top_k[:,1]:
                    hit = True
                    break
            
            if hit:
                hit_count += 1
        
        return hit_count / len(unique_users) if len(unique_users) > 0 else 0

    def ndcg(self):
        full, top_k = self._subjects, self._top_k
        
        # Get unique users
        unique_users = full[:,0].unique()
        ndcg_sum = 0
        
        for user in unique_users:
            # Get all items for this user
            user_items = full[full[:,0] == user]
            
            # Get top-k items for this user
            user_top_k = user_items[user_items[:,3] <= top_k]
            
            # Get positions of test items in the ranking
            test_positions = []
            for i, item in enumerate(user_top_k[:,1]):
                if item in self._test_set:
                    test_positions.append(int(user_top_k[i,3]))
            
            # Calculate DCG
            dcg = 0
            for pos in test_positions:
                # Using the standard DCG formula with binary relevance
                dcg += 1.0 / torch.log2(torch.tensor(1.0 + pos))
            
            # Calculate IDCG - for binary relevance, this is the sum of 1/log2(1+i) for positions 1 to len(test_positions)
            idcg = 0
            for i in range(1, len(test_positions) + 1):
                idcg += 1.0 / torch.log2(torch.tensor(1.0 + i))
            
            # Add to sum
            ndcg_sum += dcg / idcg if idcg > 0 else 0
        
        return ndcg_sum / len(unique_users)
        

        