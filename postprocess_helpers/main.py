import os
from TweetRetweetMerge import TweetRetweetMerge
from util.createUser_News_bow import createUser_News_bow
from fillMissingProfileTimeline import fillMissingProfileTimeline
from TweetNodeMapping import TweetNodeMapper
from ExtractMentionGraphIndex import ExtractMentionGraphIndex

from enum import Enum

class Dataset(Enum):
    GOS = "gossipcop"
    POL = "politifact"

class Label(Enum):
    REAL = "real"  
    FAKE = "fake"

def extractMentionGraphs():
    config = {
        "dataset": "politifact",
        "label": ["real", "fake"],
        "num_process": 30,
        "root_upfd": "../code/upfd_dataset",
        "dump_root": "../utils/news_user_mention_graph",
    }
    graph_extractor = ExtractMentionGraphIndex(config)
    graph_extractor.getMentionGraphIndex()

def merge():
    config = {
        'node_user_news_mapping_file': 'C:/Users/MSI/Fakenewsnet-hpc/FakeNewsNet-master/utils'
                                       '/pol_node_user_news_mapping.csv',
        'news_file': 'C:/Users/MSI/Fakenewsnet-hpc/FakeNewsNet-master/utils/test_news7',
        # change the num of processes
        'num_process': 12}

    news_ls = os.listdir(config['news_file'])
    merger = TweetRetweetMerge(config)
    missing_uids = merger.aggregate(news_ls)
    missing_uid_nid = zip(news_ls, missing_uids)
    merger.generate(list(missing_uid_nid))


def createBow():
    config = {'dataset_root': '../dataset',
              'root_utils': '../utils',
              #'node_user_news_mapping_file': 'gos_node_user_news_mapping.csv',
              #'upfd_user_profiles_path': 'C:/Users/MSI/Fakenewsnet-hpc/FakeNewsNet-master/utils/test_up',
              'dump_location_root': '../utils/users_bow',
              'dataset': 'politifact',
              'label': 'fake',
              }
    createUser_News_bow(config)

def mapTweetNode():
    config = {"util_root": '../utils',
              "dataset_root": "../dataset",
              #"news_file": "C:/Users/MSI/Fakenewsnet-hpc/FakeNewsNet-master/code/upfd_dataset/gossipcop/real",
              "pickle_root": "pkl_files",
              "dataset": "gossipcop",
              "label": "real"
            }

    mapper = TweetNodeMapper(config)
    df = mapper.getTweetNodeMap()
    df.sort_values(by="node_id", inplace=True, ignore_index=True)
    df.to_csv("tweet_node_mapping/df_{}_{}.csv".format(config["dataset"][:3],config["label"]),index=False)


def fillMissing():
    config = {#'dataset_root': 'C:/Users/MSI/Fakenewsnet-hpc/FakeNewsNet-master/dataset',
              'root_utils': '../utils',
              #'node_user_news_mapping_file': 'gos_node_user_news_mapping.csv',
              #'upfd_user_profiles_path': 'C:/Users/MSI/Fakenewsnet-hpc/FakeNewsNet-master/utils/test_up',
              #'upfd_user_timeline_tweets_path': 'C:/Users/MSI/Fakenewsnet-hpc/FakeNewsNet-master/utils/test_up',
              'dataset': 'politifact',
              'label': 'fake',
              "task": ["profile_mask","tl_mask"]
              }
    fillMissingProfileTimeline(config)


if __name__ == "__main__":
    # mapTweetNode()

    fillMissing()
    # merge()
    #createBow()

    # extractMentionGraphs()

