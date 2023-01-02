import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import torch
import ast
import pickle as pkl
import os

def fillMissingProfileTimeline(config):
    user_bow_mask = pd.read_csv(
        "{}/users_bow/{}_{}_users_bow.csv".format(config['init_dir_root'], config["dataset"], config["label"]))

    tasks = config["task"]
    profile_missing_sub_dict = {}
    tl_missing_sub_dict = {}
    for task in  tasks:

        false_task_bow = user_bow_mask[user_bow_mask[task] == False]['bow']  # bow of missing
        false_task_df = user_bow_mask[user_bow_mask[task] == False]
        false_task_df.reset_index(inplace=True, drop=True)

        true_task_bow = user_bow_mask[user_bow_mask[task] == True]['bow']  # bow of available profiles #true_profile_bow
        true_task_df = user_bow_mask[user_bow_mask[task] == True]
        true_task_df.reset_index(inplace=True, drop=True)

        # converting them to 2d arrays
        false_task_bow = [ast.literal_eval(bow) for bow in false_task_bow.tolist()]
        true_task_bow = [ast.literal_eval(bow) for bow in true_task_bow.tolist()]

        # converting to numpy arrays
        fpbow_arr = np.array(false_task_bow,dtype=np.float32)
        tpbow_arr = np.array(true_task_bow,dtype=np.float32)

        print("shape of missing {} bow".format(task), fpbow_arr.shape)
        print("shape of available {} bow".format(task), tpbow_arr.shape)

        def csm(A, B):
            num = torch.matmul(A, torch.t(B))
            p1 = torch.sqrt(torch.sum(A ** 2, dim=1))[:, None]
            p2 = torch.sqrt(torch.sum(B ** 2, dim=1))[None, :]
            return num / (p1 * p2)

        def csm_np(A,B):
            num=np.dot(A,B.T)
            p1=np.sqrt(np.sum(A**2,axis=1))[:,np.newaxis]
            p2=np.sqrt(np.sum(B**2,axis=1))[np.newaxis,:]
            return num/(p1*p2)

        if torch.cuda.is_available():
            tensor1 = torch.tensor(fpbow_arr, dtype=torch.float32).cuda()
            tensor2 = torch.tensor(tpbow_arr, dtype=torch.float32).cuda()
            res = csm(tensor1, tensor2)
            res_arr = res.cpu().numpy()
        else:
            res_arr = csm_np(fpbow_arr,tpbow_arr)

        max_index = np.argmax(res_arr, axis=1)
        max_idx_ls = max_index.tolist()

        path_to_all_users =  "../code/upfd_dataset/{}_{}_all".format(config["dataset"],config["label"])

        for i in tqdm(range(len(false_task_df))):
            missing_user_profile = false_task_df.iloc[i]['user_id']
            col_id = max_idx_ls[i]
            sub_user_profile_id = true_task_df.iloc[col_id]['user_id']

            if task == "tl_mask":
                tl_missing_sub_dict[missing_user_profile] = sub_user_profile_id
            
            #     file_name = "user_timeline_tweets"

            #     with open("{}/{}/{}.json".format(path_to_all_users, file_name, sub_user_profile_id), "r") as f1:
            #         user_timeline_j = json.load(f1)

            #     # dump the same file but user_id changed
            #     with open("{}/{}/{}.json".format(path_to_all_users, file_name, missing_user_profile), "x") as f2:
            #         json.dump(user_timeline_j, f2)

            elif task == "profile_mask":
                profile_missing_sub_dict[missing_user_profile] = sub_user_profile_id

            #     file_name = "user_profiles"

            #     # load the json file sub_user_profile_id
            #     with open("{}/{}/{}.json".format(path_to_all_users, file_name, sub_user_profile_id), "r") as f1:
            #         user_profile_j = json.load(f1)
            #     user_profile_j["id"] = int(missing_user_profile)
            #     user_profile_j["id_str"] = str(missing_user_profile)

            #     # dump the same file but user_id changed
            #     with open("{}/{}/{}.json".format(path_to_all_users, file_name, missing_user_profile), "x") as f2:
            #         json.dump(user_profile_j, f2)

    target = "missing_sub/{}/profile_missing_sub.pickle".format(config["dataset"])
    if os.path.exists(target): 
        with open(target,"rb") as f1:
            profile_missing_sub_ = pkl.load(f1)
        profile_missing_sub_.update(profile_missing_sub_dict)
    
        with open(target,"wb") as f1:
            pkl.dump(profile_missing_sub_,f1)
    else:
        with open(target,"wb") as f1:
            pkl.dump(profile_missing_sub_dict,f1)


    target = "missing_sub/{}/tl_missing_sub.pickle".format(config["dataset"])
    if os.path.exists(target):
        with open(target,"rb") as f2:
            tl_missing_sub_ = pkl.load(f2)
        tl_missing_sub_.update(tl_missing_sub_dict)

        with open(target,"wb") as f2:
            pkl.dump(tl_missing_sub_,f2)
    else:        
        with open(target,"wb") as f2:
            pkl.dump(tl_missing_sub_dict,f2)


