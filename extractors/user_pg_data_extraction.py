import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse 
import os

root_node_user_mappings = "../utils/node_user_mappings"
root_node_article_mappings = "../utils/node_article_mappings"
root_news = "../code/upfd_dataset"
dump_root = "../post_processing/train_test_data"
root_labels = "../post_processing/train_test_data"

def addNan(df,news_node):
    all_new_nodes = news_node
    for new_node in all_new_nodes:
        df.loc[new_node] = [np.nan]*len(df.columns)
    df.sort_index(inplace=True)
    return df

def doImpute(df):
    imputer= SimpleImputer(strategy="median")
    imputer.fit(df)
    x = imputer.transform(df)
    return x

def ScaleToRange(df):
    min_max_scaler = MinMaxScaler()
    #min_max_scaler.fit(df)
    return pd.DataFrame(min_max_scaler.fit_transform(df),
                       columns = df.columns,
                       index = df.index)
def saveDf(df,loc):
    x= df.to_numpy()
    sparse_matrix = scipy.sparse.csr_matrix(x)
    scipy.sparse.save_npz('{}.npz'.format(loc), sparse_matrix)
    df.to_csv("{}.csv".format(loc))

def saveCompound(df,dataset,loc):
    x = df.to_numpy()
    X_u = scipy.sparse.load_npz("../../../UPFD/{}/new_profile_feature.npz".format(dataset)).todense().astype(np.float32)
    all_x = np.hstack((X_u,x))
    # print(x.shape)
    # print(X_u.shape)
    sparse_matrix = scipy.sparse.csr_matrix(all_x)
    scipy.sparse.save_npz('{}.npz'.format(loc), sparse_matrix)

def extract_pg_data(ds):
    dataset =ds

    root_mg_data = "../post_processing/visualization_data/{}".format(dataset) 
    # get node-user id- news map
    ds_node_user_news_map = pd.read_csv("{}/{}_node_user_news_mapping.csv".format(root_node_user_mappings,dataset[:3]))
    #get real mention graphs
    ds_mention_graph_real = pd.read_csv("{}/{}_real_mg_features.csv".format(root_mg_data,dataset[:3]))
    # get fake mention graphs
    ds_mention_graph_fake = pd.read_csv("{}/{}_fake_mg_features.csv".format(root_mg_data,dataset[:3]))
    # get real news list 
    ds_real_news_ls = os.listdir("{}/{}_real_all/{}/{}".format(root_news,dataset,dataset,"real"))
    # get fake news list 
    ds_fake_news_ls = os.listdir("{}/{}_fake_all/{}/{}".format(root_news,dataset,dataset,"fake"))
    # get article-node mapping 
    node_article_mapping = pd.read_csv("{}/{}_node_article_mapping.csv".format(root_node_article_mappings,dataset[:3]))
    
    ds_real_news_nodes = node_article_mapping.loc[node_article_mapping.apply(lambda x: x.news_article in ds_real_news_ls , axis=1)]["node_id"].astype(float).tolist()
    ds_fake_news_nodes = node_article_mapping.loc[node_article_mapping.apply(lambda x: x.news_article in ds_fake_news_ls , axis=1)]["node_id"].astype(float).tolist()

    ds_mention_graph_all = ds_mention_graph_real.append(ds_mention_graph_fake).copy()
    ds_node_mg_all = ds_node_user_news_map.merge(ds_mention_graph_all, how='left',on=['user_id','news_id']).copy()
    ds_node_mg_all.set_index('node_id',drop=True,inplace=True)
    ds_node_mg_all.drop(['user_id','news_id'],axis=1,inplace=True)

    ds_node_mg_all_nan = addNan(ds_node_mg_all,ds_real_news_nodes+ds_fake_news_nodes)

    # get the y_label 
    y_label = np.load("{}/{}.npy".format(root_labels,dataset))
    label_bin = np.bincount(y_label)

    ds_node_mg_real_nan = ds_node_mg_all_nan.iloc[:label_bin[0]]
    ds_node_mg_fake_nan = ds_node_mg_all_nan.iloc[label_bin[0]:(label_bin[0]+label_bin[1])]

    ds_mg_real_impute = pd.DataFrame(doImpute(ds_node_mg_real_nan),columns = ds_node_mg_real_nan.columns,
                                            index = ds_node_mg_real_nan.index)

    ds_mg_fake_impute = pd.DataFrame(doImpute(ds_node_mg_fake_nan),columns = ds_node_mg_fake_nan.columns,
                                            index = ds_node_mg_fake_nan.index)
    
    ds_mg_all_impute = ds_mg_real_impute.append(ds_mg_fake_impute).copy()

    ds_mg_all_scaled = ScaleToRange(ds_mg_all_impute)

    loc = "{}/{}/new_user_mg_feature".format(dump_root,dataset)
    saveDf(ds_mg_all_scaled,loc)

    ds_mg_real_scaled = ds_mg_all_scaled.iloc[:label_bin[0]]
    
    loc = "{}/{}/sep/new_user_mg_feature".format(dump_root,dataset)
    saveDf(ds_mg_real_scaled,loc)

    ds_mg_fake_scaled =  ds_mg_all_scaled.iloc[label_bin[0]:(label_bin[0]+label_bin[1])]
    loc = "{}/{}/sep/new_user_mg_feature".format(dump_root,dataset)
    saveDf(ds_mg_fake_scaled,loc)

    loc = "{}/{}/new_user_mg_compound_feature".format(dump_root,dataset)
    saveCompound(ds_mg_all_scaled,dataset,loc)