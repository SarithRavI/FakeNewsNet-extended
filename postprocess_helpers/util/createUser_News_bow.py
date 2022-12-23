import pandas as pd
import os
from tqdm import tqdm


def createUser_News_bow(config):
    #node_user_mappings/gos_node_user_news_mapping.csv
    if config["dataset"] == 'politifact':
        shrt_key = 'pol'
    elif config["dataset"] == "gossipcop":
        shrt_key = 'gos'

    node_user_news_mapping = pd.read_csv("{}/node_user_mappings/{}_node_user_news_mapping.csv".format(config["root_utils"], shrt_key))
    # news_user_groups = node_user_news_mapping.groupby('news_id', sort=False)['user_id'].apply(list)

    df = pd.read_csv("{}/{}_{}.csv".format(config["dataset_root"], config["dataset"], config["label"]))
    news_ls = list(df['id'])

    news_user_mask = node_user_news_mapping.isin(news_ls)

    node_user_news_mapping['flag'] = news_user_mask['news_id']

    news_user = node_user_news_mapping[node_user_news_mapping['flag'] == True]

    true_user_news_groups = news_user.groupby('user_id', sort=False)['news_id'].apply(list)

    bow_ls = []
    for user_share in tqdm(list(true_user_news_groups)):
        bow = [1 if s in user_share else 0 for s in news_ls]
        bow_ls.append(bow)

    user_news_bow_df = pd.DataFrame(true_user_news_groups)
    user_news_bow_df['bow'] = bow_ls
    user_news_bow_df.reset_index(inplace=True)

    user_ids_have = os.listdir("../code/upfd_dataset/{}_{}_all/user_profiles".format(config['dataset'],config["label"]))
    user_ids_profile_have = [int(user_id.replace(".json", "")) for user_id in user_ids_have]

    tl_ids_have = os.listdir("../code/upfd_dataset/{}_{}_all/user_timeline_tweets".format(config['dataset'],config["label"]))
    user_ids_tl_have = [int(user_id.replace(".json", "")) for user_id in tl_ids_have]

    profile_mask = user_news_bow_df['user_id'].isin(user_ids_profile_have)
    tl_mask = user_news_bow_df['user_id'].isin(user_ids_tl_have)

    user_news_bow_df["profile_mask"] = profile_mask
    user_news_bow_df["tl_mask"] = tl_mask

    user_news_bow_df.to_csv(
        "{}/{}_{}_users_bow.csv".format(config["dump_location_root"], config["dataset"], config["label"]), index=False)

def getMissingTweetNodeIds(missing_uids):
    tot_nodes = []
    for uid in missing_uids:
        for node in uid:
            tot_nodes.append(node)
    return tot_nodes





