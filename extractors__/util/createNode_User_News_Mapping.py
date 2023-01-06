import pandas as pd
import pickle
from tqdm import tqdm

def createNodeUserNewsMapping(config):
    dataset = config['dataset']
    # root to pkl files
    pkl_root =  f"{config['init_dir_root']}/pkl_files"
    # root to dumping location
    save_root =  f"{config['init_dir_root']}/node_user_mappings"

    if dataset == 'gossipcop':
        pkl_str = 'gos'
        qkey = dataset + "-"
    if dataset == 'politifact':
        qkey = dataset
        pkl_str = 'pol'

    # set the root path to corresponding pkl file
    with open("{}/{}_id_twitter_mapping.pkl".format(pkl_root, pkl_str), 'rb') as f:
        upfd_dic = pickle.load(f)

    news_id = ""
    node_user_news_mapping = pd.DataFrame()
    for key in tqdm(list(upfd_dic.keys())):
        row_dict = {"node_id": int(key)}
        if "{}".format(qkey) in upfd_dic[key]:
            news_id = upfd_dic[key]
        else:
            row_dict['user_id'] = upfd_dic[key]
            row_dict['news_id'] = news_id
            node_user_news_mapping = node_user_news_mapping.append(row_dict, ignore_index=True)

    node_user_news_mapping.to_csv("{}/{}_node_user_news_mapping.csv".format(save_root, pkl_str), index=False)
