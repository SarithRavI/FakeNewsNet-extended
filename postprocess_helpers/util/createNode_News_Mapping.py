import pandas as pd
import pickle
from tqdm import tqdm

def createNodeNewsMapping(config):
    dataset = config['dataset']
    # root to pkl files
    pkl_root =  f"{config['init_dir_root']}/pkl_files"
    # root to dumping location
    save_root =  f"{config['init_dir_root']}/node_article_mappings"

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
    node_news_mapping = pd.DataFrame()
    for key in tqdm(list(upfd_dic.keys())):
        row_dict = {"node_id": int(key)}
        if "{}".format(qkey) in upfd_dic[key]:
            row_dict['news_article'] =  upfd_dic[key]
            node_news_mapping = node_news_mapping.append(row_dict, ignore_index=True)

    node_news_mapping.to_csv("{}/{}_node_article_mapping.csv".format(save_root, pkl_str), index=False)
