import pandas as pd
import numpy as np
import os

from util import addNan, doImpute, ScaleToRange, saveDf, saveCompound

# setting paths 
root_raw_data = "../post_processing/visualization_data"
root_node_article_mappings = "../utils/node_article_mappings" #
root_node_user_mappings = "../utils/node_user_mappings" #
root_missing_nodes = "../utils/tweet_node_mapping/df_missing_nodes" #
root_news = "../code/upfd_dataset" #
dump_root = "../post_processing/train_test_data"


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

