import os
from TweetRetweetMerge import TweetRetweetMerge
from util.createUser_News_bow import createUser_News_bow
from fillMissingProfileTimeline import fillMissingProfileTimeline
from TweetNodeMapping import TweetNodeMapper
from ExtractMentionGraphIndex import ExtractMentionGraphIndex
from util import createNode_User_News_Mapping

import argparse
from enum import Enum

class Dataset(Enum):
    GOS = "gossipcop"
    POL = "politifact"

class Label(Enum):
    REAL = "real"  
    FAKE = "fake"

def extractMentionGraphs(dataset):
    config = {
        "dataset": dataset,
        "label": ["real", "fake"],
        "num_process": 30,
        "root_upfd": "../code/upfd_dataset",
        "dump_root": "../utils/news_user_mention_graph",
    }
    graph_extractor = ExtractMentionGraphIndex(config)
    graph_extractor.getMentionGraphIndex()

def merge(dataset):
    config = {
        'dataset' : dataset,
        'node_user_news_mapping_file': '..',
        'news_file': '..',
        # change the num of processes
        'num_process': 12}

    news_ls = os.listdir(config['news_file'])
    merger = TweetRetweetMerge(config)
    missing_uids = merger.aggregate(news_ls)
    missing_uid_nid = zip(news_ls, missing_uids)
    merger.generate(list(missing_uid_nid))


def createBow(dataset,label):
    config = {'dataset_root': '../dataset',
              'root_utils': '../utils',
              #'node_user_news_mapping_file': 'gos_node_user_news_mapping.csv',
              #'upfd_user_profiles_path': 'C:/Users/MSI/Fakenewsnet-hpc/FakeNewsNet-master/utils/test_up',
              'dump_location_root': '../utils/users_bow',
              'dataset': dataset,
              'label': label,
              }
    createUser_News_bow(config)

def mapTweetNode(dataset,label):
    config = {"util_root": '../utils',
              "dataset_root": "../dataset",
              #"news_file": "C:/Users/MSI/Fakenewsnet-hpc/FakeNewsNet-master/code/upfd_dataset/gossipcop/real",
              "pickle_root": "pkl_files",
              "dataset": dataset,
              "label": label
            }

    mapper = TweetNodeMapper(config)
    df = mapper.getTweetNodeMap()
    df.sort_values(by="node_id", inplace=True, ignore_index=True)
    df.to_csv("tweet_node_mapping/df_{}_{}.csv".format(config["dataset"][:3],config["label"]),index=False)


def fillMissing(dataset,label):
    config = {#'dataset_root': 'C:/Users/MSI/Fakenewsnet-hpc/FakeNewsNet-master/dataset',
              'root_utils': '../utils',
              #'node_user_news_mapping_file': 'gos_node_user_news_mapping.csv',
              #'upfd_user_profiles_path': 'C:/Users/MSI/Fakenewsnet-hpc/FakeNewsNet-master/utils/test_up',
              #'upfd_user_timeline_tweets_path': 'C:/Users/MSI/Fakenewsnet-hpc/FakeNewsNet-master/utils/test_up',
              'dataset': dataset,
              'label': label,
              "task": ["profile_mask","tl_mask"]
              }
    fillMissingProfileTimeline(config)

def createNodeUserMapping(dataset):
        config = {
              'dataset':dataset,
              'pickle_root': "pkl_files",
              'save_root': '../node_user_mappings',
              }
        createNode_User_News_Mapping.createNodeUserMapping(config)


def main():
    dataset = None 
    label = None 
    parser = argparse.ArgumentParser(description='Initiating the postprocessing')
    parser.add_argument('--dataset', type=str, default=Dataset.GOS.value,
                        help='which dataset to be used: pol for politifact. gos for gossipcop')
    parser.add_argument('--label', type=str, default=Label.REAL.value,
                        help='Indicate the label: real for true label. fake for false label')
    
    args = parser.parse_args()
    print(args)

    if args.dataset == 'gos':
        dataset =[Dataset.GOS.value]
    elif args.dataset == 'pol':
        dataset = [Dataset.POL.value]
    elif args.dataset == 'all':
        dataset = [Dataset.GOS.value, Dataset.POL.value]
    if args.label == 'real':
        label = [Label.REAL.value]
    elif args.label == 'fake':
        label = [Label.FAKE.value]
    elif args.label == 'all':
        label = [Label.REAL.value, Label.FAKE.value]
    
    for inp_dataset in dataset:
        createNodeUserMapping(inp_dataset) 
    
    for inp_dataset in dataset:
        for inp_label in label:
             mapTweetNode(inp_dataset,inp_label) 
             createBow(inp_dataset,inp_label)
             fillMissing(inp_dataset,inp_label)
    
    for inp_dataset in dataset:
        merge(inp_dataset) 

    for inp_dataset in dataset:
        extractMentionGraphs(inp_dataset) 

    # run postprocess_helpers\util\createNode_User_News_Mapping.py -done
    # mapTweetNode() -done
    # createBow() -done
    # fillMissing() -done 
    # merge() - not done 
    # extractMentionGraphs() -done
                       
if __name__ == "__main__":
    main()


