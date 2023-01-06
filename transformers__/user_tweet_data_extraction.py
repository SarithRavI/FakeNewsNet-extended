import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sp
import os

# setting paths 
root_raw_data = "../post_processing/visualization_data"
root_node_article_mappings = "../utils/node_article_mappings" #
root_node_user_mappings = "../utils/node_user_mappings" #
root_missing_nodes = "../utils/tweet_node_mapping/df_missing_nodes" #
root_news = "../code/upfd_dataset" #
dump_root = "../post_processing/train_test_data"

def doImpute(df):
    imputer= SimpleImputer(strategy="median")
    imputer.fit(df)
    x = imputer.transform(df)
    return x
def addNan(df,missing,news_node):
    all_new_nodes = missing+news_node
    for new_node in all_new_nodes:
        df.loc[new_node] = [np.nan]*len(df.columns)
    df.sort_index(inplace=True)
    return df

def addNanSpacy(arr,missing,news_node):
    all_new_nodes = missing+news_node
    all_new_nodes.sort()
    # print(arr.shape[1])
    for new_node in all_new_nodes:
        if int(new_node) > arr.shape[0]:
            arr = np.append(arr,np.array([[0]*(arr.shape[1])]),axis=0)
        else:
            # print(int(new_node),arr.shape[0])
            arr = np.insert(arr,int(new_node),[0]*(arr.shape[1]),axis=0)
    return arr


def ScaleToRange(df):
    min_max_scaler = MinMaxScaler()
    #min_max_scaler.fit(df)
    return pd.DataFrame(min_max_scaler.fit_transform(df),
                       columns = df.columns,
                       index = df.index)
def saveDf(df,loc):
    x= df.to_numpy()
    sparse_matrix = sp.csr_matrix(x)
    sp.save_npz('{}.npz'.format(loc), sparse_matrix)
    df.to_csv("{}.csv".format(loc))

def saveCompound(df,dataset,loc):
    x = df.to_numpy()
    X_u = sp.load_npz("../../../UPFD/{}/new_profile_feature.npz".format(dataset)).todense().astype(np.float32)
    all_x = np.hstack((X_u,x))    
    # print(x.shape)
    # print(X_u.shape)
    sparse_matrix = sp.csr_matrix(all_x)
    sp.save_npz('{}.npz'.format(loc), sparse_matrix)

def saveSpacyCompound(arr,dataset,loc):
    X_u = sp.load_npz("../../../UPFD/{}/new_profile_feature.npz".format(dataset)).todense().astype(np.float32)
    all_x = np.hstack((X_u,arr)) 
    # print(x.shape)
    # print(X_u.shape)
    print("Here I print 2: ",all_x.shape)
    sparse_matrix = sp.csr_matrix(all_x)
    sp.save_npz('{}.npz'.format(loc), sparse_matrix)

def extract_tweet_data(ds):
    dataset= ds
    # real news id list
    # /home/in36cs19/FakeNewsNet-fyp/FakeNewsNet-master/dataset
    ds_real_news_ls = os.listdir("{}/{}_real_all/{}/{}".format(root_news,dataset,dataset,"real"))
    # fake news id list 
    # /home/in36cs19/FakeNewsNet-fyp/FakeNewsNet-master/dataset
    ds_fake_news_ls = os.listdir("{}/{}_fake_all/{}/{}".format(root_news,dataset,dataset,"fake"))
    # get node-article_id mapping
    # /home/in36cs19/FakeNewsNet-fyp/FakeNewsNet-master/utils/node_article_mappings
    node_article_mapping = pd.read_csv("{}/{}_node_article_mapping.csv".format(root_node_article_mappings,dataset[:3]))
    # node ids of real news
    ds_real_news_nodes = node_article_mapping.loc[node_article_mapping.apply(lambda x: x.news_article in ds_real_news_ls , axis=1)]["node_id"].astype(float).tolist()
    # node ids of fake news
    ds_fake_news_nodes = node_article_mapping.loc[node_article_mapping.apply(lambda x: x.news_article in ds_fake_news_ls , axis=1)]["node_id"].astype(float).tolist()

    # get missing data
    # /home/in36cs19/FakeNewsNet-fyp/FakeNewsNet-master/utils/tweet_node_mapping/df_missing_nodes
    with open("{}/{}_missing_nodes.txt".format(root_missing_nodes,dataset[:3])) as f:
        missing_all = f.read()
    missing_in_real ,missing_in_fake =missing_all.split("\n\n")

    missing_in_real = [float(m) for m in missing_in_real.split("\n")]
    missing_in_fake = [float(m) for m in missing_in_fake.split("\n")[:-1]]

    # /home/in36cs19/FakeNewsNet-fyp/FakeNewsNet-master/post_processing/visualization_data/gossipcop
    # real
    ds_real_textual_features = pd.read_csv("{}/{}/{}_{}_textual_features.csv".format(root_raw_data,dataset,dataset[:3],"real"),index_col=0)
    # fake
    ds_fake_textual_features =  pd.read_csv("{}/{}/{}_{}_textual_features.csv".format(root_raw_data,dataset,dataset[:3],"fake"),index_col=0)

    # print("err1",len(ds_real_textual_features),len(ds_fake_textual_features))
    
    # real node tweet text with nan not scaled not filled the nan 
    ds_real_textual_features_nan = addNan(ds_real_textual_features,missing_in_real,ds_real_news_nodes)
    # real node tweet text with nan not scaled , not filled the nan
    ds_fake_textual_features_nan = addNan(ds_fake_textual_features,missing_in_fake,ds_fake_news_nodes)

    # print("err2",len(ds_real_textual_features_nan),len(ds_fake_textual_features_nan))
    # real nodes all 
    real_nodes_all = list(ds_real_textual_features_nan.index)
    # fake nodes all 
    fake_nodes_all = list(ds_fake_textual_features_nan.index)
    # fake node tweet-text filled nan not scaled 
    ds_fake_textual_features_sep = pd.DataFrame(doImpute(ds_fake_textual_features_nan),columns = ds_fake_textual_features_nan.columns,
                                            index = ds_fake_textual_features_nan.index)
    # real node tweet-text filled nan not scaled 
    ds_real_textual_features_sep = pd.DataFrame(doImpute(ds_real_textual_features_nan),columns =ds_real_textual_features_nan.columns,
                                            index=ds_real_textual_features_nan.index)
    # combined fake and real news
    ds_all_textual_features = ds_real_textual_features_sep.append(ds_fake_textual_features_sep)

    # scaled it 
    all_textual_features_scaled =ScaleToRange(ds_all_textual_features) 

    # save all in one 
    loc = "{}/{}/new_user_tweet_feature".format(dump_root,dataset)
    saveDf(all_textual_features_scaled,loc)

    new_profile_text_feature_real = all_textual_features_scaled.loc[real_nodes_all]
    new_profile_text_feature_fake = all_textual_features_scaled.loc[fake_nodes_all]
    # save real news only
    loc = "{}/{}/sep/new_user_tweet_feature_real".format(dump_root,dataset)
    saveDf(new_profile_text_feature_real,loc)
    # save fake news only
    loc = "{}/{}/sep/new_user_tweet_feature_fake".format(dump_root,dataset)
    saveDf(new_profile_text_feature_fake,loc)

    #save node an corresponding label : for visualization
    node_labels = np.append([0]*len(real_nodes_all),[1]*len(fake_nodes_all))
    np.save("{}/{}".format(dump_root,dataset),node_labels)

    loc = "{}/{}/new_user_tweet_compound_feature".format(dump_root,dataset)
    saveCompound(all_textual_features_scaled,dataset,loc)

def extract_spacy_data(ds):
    dataset= ds
    # real news id list

    ds_real_news_ls = os.listdir("{}/{}_real_all/{}/{}".format(root_news,dataset,dataset,"real"))
    # fake news id list 

    ds_fake_news_ls = os.listdir("{}/{}_fake_all/{}/{}".format(root_news,dataset,dataset,"fake"))
    # get node-article_id mapping

    node_article_mapping = pd.read_csv("{}/{}_node_article_mapping.csv".format(root_node_article_mappings,dataset[:3]))
    # node ids of real news
    ds_real_news_nodes = node_article_mapping.loc[node_article_mapping.apply(lambda x: x.news_article in ds_real_news_ls , axis=1)]["node_id"].astype(float).tolist()
    # node ids of fake news
    ds_fake_news_nodes = node_article_mapping.loc[node_article_mapping.apply(lambda x: x.news_article in ds_fake_news_ls , axis=1)]["node_id"].astype(float).tolist()


    with open("{}/{}_missing_nodes.txt".format(root_missing_nodes,dataset[:3])) as f:
        missing_all = f.read()
    missing_in_real ,missing_in_fake =missing_all.split("\n\n")

    missing_in_real = [float(m) for m in missing_in_real.split("\n")]
    missing_in_fake = [float(m) for m in missing_in_fake.split("\n")[:-1]]
    # real
    ds_real_spacy = np.load("{}/{}/{}_{}_textual_features_spacy.npy".format(root_raw_data,dataset,dataset[:3],"real"))
    # fake
    ds_fake_spacy =  np.load("{}/{}/{}_{}_textual_features_spacy.npy".format(root_raw_data,dataset,dataset[:3],"fake"))
        # real node tweet text with nan not scaled not filled the nan 
    ds_real_spacy_nan = addNanSpacy(ds_real_spacy,missing_in_real,ds_real_news_nodes)
    # real node tweet text with nan not scaled , not filled the nan
    ds_fake_spacy_nan = addNanSpacy(ds_fake_spacy,missing_in_fake,ds_fake_news_nodes)

    ds_all_spacy = np.vstack((ds_real_spacy_nan,ds_fake_spacy_nan))
    loc = "{}/{}/new_spacy_compound_feature".format(dump_root,dataset)
    saveSpacyCompound(ds_all_spacy,dataset,loc)

