import os
from TweetRetweetMerge import TweetRetweetMerge
from util.createUser_News_bow import createUser_News_bow
from FillMissingProfileTimeline import fillMissingProfileTimeline
from TweetNodeMapping import TweetNodeMapper
from ExtractMentionGraphIndex import ExtractMentionGraphIndex
from util import createNode_User_News_Mapping

import os
import shutil
import argparse
from enum import Enum

class Dataset(Enum):
    GOS = "gossipcop"
    POL = "politifact"

class Label(Enum):
    REAL = "real"  
    FAKE = "fake"

def extractMentionGraphs(dataset,init_dir):
    config = {
        "dataset": dataset,
        "label": ["real", "fake"],
        "num_process": 4,
        'init_dir_root': init_dir,
    }
    graph_extractor = ExtractMentionGraphIndex(config)
    graph_extractor.getMentionGraphIndex()

def merge(dataset,label,init_dir):
    config = {
        'dataset' : dataset,
        'init_dir_root': init_dir,
        'label':label,
        'num_process': 4}
    config["news_file"] = "../code/upfd_dataset/{}_{}_all/{}/{}".format(config["dataset"],
                                     config["label"],config["dataset"],config["label"])
    news_ls = os.listdir(config['news_file'])
    merger = TweetRetweetMerge(config)
    missing_uids = merger.aggregate(news_ls)
    missing_uid_nid = zip(news_ls, missing_uids)
    merger.generate(list(missing_uid_nid))


def createBow(dataset,label,init_dir):
    config = {'dataset_root': '../dataset',
              'init_dir_root': init_dir,
              'dataset': dataset,
              'label': label,
              }
    createUser_News_bow(config)

def mapTweetNode(dataset,label,init_dir):
    config = { "dataset_root": "../dataset",
              'init_dir_root': init_dir,
              "dataset": dataset,
              "label": label
            }

    mapper = TweetNodeMapper(config)
    df = mapper.getTweetNodeMap()
    df.sort_values(by="node_id", inplace=True, ignore_index=True)
    df.to_csv("{}/tweet_node_mapping/df_{}_{}.csv".format(config['init_dir_root'],config["dataset"][:3],config["label"]),index=False)


def fillMissing(dataset,label,init_dir):
    config = {'init_dir_root': init_dir,
              'dataset': dataset,
              'label': label,
              "task": ["profile_mask","tl_mask"]
              }
    fillMissingProfileTimeline(config)

def createNodeUserNewsMapping(dataset,init_dir):
        config = {
              'dataset':dataset,
              'init_dir_root': init_dir
              }
        createNode_User_News_Mapping.createNodeUserNewsMapping(config)


def main():
    dataset = None 
    label = None 
    parser = argparse.ArgumentParser(description='Initiating the postprocessing')
    parser.add_argument('--dataset', type=str, default=Dataset.GOS.value,
                        help='which dataset to be used: pol for politifact. gos for gossipcop')
    parser.add_argument('--label', type=str, default=Label.REAL.value,
                        help='Indicate the label: real for true label. fake for false label')
    parser.add_argument('--init_dir_root', type=str,
                        help='root path of init_dir folder')
    parser.add_argument('--testing', type=str,
                        help='denote if testing : True or False')
    
    args = parser.parse_args()
    print(args)

    if 'gos' in args.dataset  :
        dataset =[Dataset.GOS.value]
    elif 'pol' in args.dataset  :
        dataset = [Dataset.POL.value]
    elif args.dataset == 'all':
        dataset = [Dataset.GOS.value, Dataset.POL.value]
    if args.label == 'real':
        label = [Label.REAL.value]
    elif args.label == 'fake':
        label = [Label.FAKE.value]
    elif args.label == 'all':
        label = [Label.REAL.value, Label.FAKE.value]

    isTest = eval(args.testing)
    
    folder_ls = ['news_user_mention_graph',
                'node_article_mappings',
                'node_user_mappings',
                'tweet_node_mapping',
                'pkl_files',
                'users_bow']

    init_dir = args.init_dir_root

    for folder in folder_ls:
        os.makedirs(os.path.join(init_dir,folder),exist_ok=True)
    
    if isTest:
        for file in os.listdir('util/pkl_files_testing'):
            shutil.copy2(f'util/pkl_files_testing/{file}',f'{init_dir}/pkl_files')
    elif isTest == False:
        for file in os.listdir('util/pkl_files'):
            shutil.copy2(f'util/pkl_files/{file}',f'{init_dir}/pkl_files')

    for inp_dataset in dataset:
        # gossipcop real
        createNodeUserNewsMapping(inp_dataset,init_dir) 
    
    for inp_dataset in dataset:
        for inp_label in label:
            merge(inp_dataset,inp_label,init_dir) 
            mapTweetNode(inp_dataset,inp_label,init_dir) 
            createBow(inp_dataset,inp_label,init_dir)
            fillMissing(inp_dataset,inp_label,init_dir)
    
    # for inp_dataset in dataset:
    #     merge(inp_dataset,init_dir) 

    for inp_dataset in dataset:
        extractMentionGraphs(inp_dataset,init_dir) 

    # run postprocess_helpers\util\createNode_User_News_Mapping.py -done
    # mapTweetNode() -done
    # createBow() -done
    # fillMissing() -done 
    # merge() - not done 
    # extractMentionGraphs() -done
                       
if __name__ == "__main__":
    main()


