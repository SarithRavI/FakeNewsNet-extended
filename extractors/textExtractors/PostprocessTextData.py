import os
import pandas as pd
import numpy as np
import json
from multiprocessing.pool import Pool
from tqdm import tqdm
from .util.SentimentAnalysis import SentimentAnalyzer
from .util.util import create_dir
import spacy

class PostData:
    def __init__(self, config):
        self.config = config
        self.label = None
        self.label_ls = config["label_ls"]
        self.nlp = spacy.load('en_core_web_md')

    def collectSpacy(self, label_node_tweet_news_row):
        try:
            with open(
                    "{}/{}_{}_all/{}/{}/{}/tweets/{}.json".format(self.config["root_upfd_data"], self.config["dataset"],
                                                                  self.label, self.config["dataset"], self.label,
                                                                  label_node_tweet_news_row["news_id"],
                                                                  label_node_tweet_news_row["tweet_id"])) as tweet:
                tweet_j = json.load(tweet)
            text_arr = np.array(self.nlp(tweet_j["text"]).vector)
            return text_arr
        except Exception as ex:
            print("{}".format(ex))



    def collectVisData(self, label_node_tweet_news_row):
        try:
            with open(
                    "{}/{}_{}_all/{}/{}/{}/tweets/{}.json".format(self.config["root_upfd_data"], self.config["dataset"],
                                                                  self.label, self.config["dataset"], self.label,
                                                                  label_node_tweet_news_row["news_id"],
                                                                  label_node_tweet_news_row["tweet_id"])) as tweet:
                tweet_j = json.load(tweet)

            entities = tweet_j["entities"]
            # call the vader function
            analyzer= SentimentAnalyzer()
            vaderScores = analyzer.getVaderSentimentScores(tweet_j["text"])
            subjectivityScore = analyzer.getTextBlobSubjectivity(tweet_j["text"])
            # spacy_vector = self.nlp(tweet_j["text"])

            text_arr= [                     # pre spacy
                len(entities["user_mentions"]),
                len(entities["urls"]),
                len(entities["symbols"]),
                len(entities["hashtags"]),
                vaderScores["neg"],
                vaderScores["neu"],
                vaderScores["pos"],
                vaderScores["compound"],
                subjectivityScore
            ]
            # text_arr.append(spacy_vector) # post spacy

            return  text_arr

        except Exception as ex:
            print("{}".format(ex))

    def processTweetData(self):
        all_news_node = pd.read_csv("{}/{}_node_user_news_mapping.csv".format(self.config["root_node_user_mapping"],
                                                                              self.config["dataset"][:3]))
        all_news_node.drop(["user_id"], axis=1, inplace=True)

        for label in self.label_ls:
            self.label = label
            label_node_tweetData = pd.DataFrame()
            label_node_tweet = pd.read_csv("{}/df_{}_{}.csv".format(self.config["root_tweet_node_mapping"],
                                                                    self.config["dataset"][:3], label))
            label_node_tweet.drop(["pub_time"], axis=1, inplace=True)
            label_node_tweet_news = label_node_tweet.merge(all_news_node, how="inner", on="node_id").copy()
            label_node_tweet_news["tweet_id"]=label_node_tweet_news["tweet_id"].astype(str)
            label_node_tweet_news["news_id"]=label_node_tweet_news["news_id"].astype(str)
            pool = Pool(self.config["num_process"])
            pbar = tqdm(total=len(label_node_tweet_news))

            def update(arg):
                pbar.update()

            res = []
            for i in range(len(label_node_tweet_news)):
                data=pool.map_async(self.collectVisData, [label_node_tweet_news.iloc[i]],callback = update)
                res.append(data.get()[0])
            pool.close()
            pool.join()

            res_np = np.array(res)
            #print(res_np)
            label_node_tweetData["node_id"] = label_node_tweet_news["node_id"].tolist()
            label_node_tweetData["num_user_mentions"] = res_np[:, 0]
            label_node_tweetData["num_urls"] = res_np[:, 1]
            label_node_tweetData["num_emojis"] = res_np[:, 2]
            label_node_tweetData["num_hashtags"] = res_np[:, 3]
            label_node_tweetData["neg"] = res_np[:,4]
            label_node_tweetData["neu"] = res_np[:,5]
            label_node_tweetData["pos"] = res_np[:,6]
            label_node_tweetData["compound"] = res_np[:,7]
            label_node_tweetData["subjectivity"] =res_np[:,8]

            # label_node_tweetData["subjectivity"] =res_np[:,9]
            target_root = self.config["dump_location"]
            target = os.path.join(target_root,"{}_{}_textual_features.csv".format(self.config["dataset"][:3], self.label))
            label_node_tweetData.to_csv(target,index=False)

    def processSpacy(self):
        all_news_node = pd.read_csv("{}/{}_node_user_news_mapping.csv".format(self.config["root_node_user_mapping"],
                                                                              self.config["dataset"][:3]))
        all_news_node.drop(["user_id"], axis=1, inplace=True)

        for label in self.label_ls:
            self.label = label
            label_node_tweetData = pd.DataFrame()
            label_node_tweet = pd.read_csv("{}/df_{}_{}.csv".format(self.config["root_tweet_node_mapping"],
                                                                    self.config["dataset"][:3], label))
            label_node_tweet.drop(["pub_time"], axis=1, inplace=True)
            label_node_tweet_news = label_node_tweet.merge(all_news_node, how="inner", on="node_id").copy()
            label_node_tweet_news["tweet_id"]=label_node_tweet_news["tweet_id"].astype(str)
            label_node_tweet_news["news_id"]=label_node_tweet_news["news_id"].astype(str)
            pool = Pool(self.config["num_process"])
            pbar = tqdm(total=len(label_node_tweet_news))

            def update(arg):
                pbar.update()

            res = []
            for i in range(len(label_node_tweet_news)): 
                data=pool.map_async(self.collectSpacy, [label_node_tweet_news.iloc[i]],callback = update)
                res.append(data.get()[0])
            pool.close()
            pool.join()

            res_np = np.array(res)

            target_root = self.config["dump_location"]
            np.save(os.path.join(target_root,"{}_{}_textual_features_spacy.npy".format(self.config["dataset"][:3],self.label)),
                                                                               res_np)
        
