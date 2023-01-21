import pandas as pd
import numpy as np
import os

from util import addNanSpacy,saveSpacyCompound
# setting paths 
root_raw_data = "../post_processing/visualization_data"
root_node_article_mappings = "../utils/node_article_mappings" #
root_node_user_mappings = "../utils/node_user_mappings" #
root_missing_nodes = "../utils/tweet_node_mapping/df_missing_nodes" #
root_news = "../code/upfd_dataset" #
dump_root = "../post_processing/train_test_data"


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

