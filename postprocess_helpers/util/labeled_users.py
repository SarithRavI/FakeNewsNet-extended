import pandas as pd


def getLabeledUsers(dataset, label,init_file_root):
    node_user_news_mapping = pd.read_csv('{}/node_user_mappings/{}_node_user_news_mapping.csv'.format(init_file_root,dataset[:3]))
    df = pd.read_csv('../../dataset/{}_{}.csv'.format(dataset, label))
    label_news_ls = list(df['id'])
    label_news_user_mask = node_user_news_mapping.isin(label_news_ls)
    node_user_news_mapping['flag'] = label_news_user_mask['news_id']
    label_news_user = node_user_news_mapping[node_user_news_mapping['flag'] == True]
    label_user_news_groups = label_news_user.groupby('news_id', sort=False)['user_id'].apply(set)
    # print(label_user_news_groups)
    return pd.DataFrame(label_user_news_groups)
