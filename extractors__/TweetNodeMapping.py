import pandas as pd
import numpy as np
from datetime import datetime
import json
from tqdm import tqdm
import os
import pickle
from util.createUser_News_bow import getMissingTweetNodeIds
from util.util import create_dir

class TweetNodeMapper:
    def __init__(self,config):
        self.config = config
        self.config['pickle_root'] = f'{self.config["init_dir_root"]}/pkl_files'
        self.config["news_file"] = "../code/upfd_dataset/{}_{}_all/{}/{}".format(self.config["dataset"],
                                     self.config["label"],self.config["dataset"],self.config["label"])

        self.news_mapping_file = "{}_{}.csv".format(self.config["dataset"], self.config["label"])
        self.node_user_news_mapping = pd.read_csv(
            "{}/node_user_mappings/{}_node_user_news_mapping.csv".format(self.config["init_dir_root"], self.config["dataset"][:3]))

        self.news_user_groups = self.node_user_news_mapping.groupby('news_id', sort=False)['user_id'].apply(list)
        self.all_label_news = os.listdir(self.config["news_file"])

        self.node_tweet_mapping = pd.DataFrame(columns=["node_id", "tweet_id", "pub_time"])

        self.nodes_expect = pd.DataFrame()

        self.missing_nodes = []

    def crawlTweetFiles(self):
        uid_tid_news_map = pd.DataFrame(columns=['news_id', 'user_id', 'tweet_ids'])

        for news in tqdm(self.all_label_news):
            try:
                tweet_files = os.listdir("{}/{}/tweets".format(self.config["news_file"], news))
                for tweet in tweet_files:
                    with open("{}/{}/tweets/{}".format(self.config["news_file"], news, tweet), "r") as f1:
                        tweet_j = json.load(f1)
                    news_id = news
                    tweet_id = tweet_j['id_str']
                    tweet_uid = tweet_j['user']['id']

                    if tweet[0] == 'r':
                        uid_tid_news_map = uid_tid_news_map.append({'news_id': news_id,
                                                                    'user_id': tweet_uid}, ignore_index=True)
                    else:
                        uid_tid_news_map = uid_tid_news_map.append({'news_id': news_id,
                                                                    'user_id': tweet_uid,
                                                                    'tweet_ids': tweet_id}, ignore_index=True)

            except Exception as ex:
                print(ex)
        return uid_tid_news_map

    def mergeNodeTweet(self):
        uid_tid_news_map = self.crawlTweetFiles()
        tweets_have = pd.DataFrame(
            uid_tid_news_map.groupby(['news_id', 'user_id'], sort=False)['tweet_ids'].apply(list))

        tweet_node_all = pd.DataFrame(
            self.node_user_news_mapping.groupby(['news_id', 'user_id'], sort=False)['node_id'].apply(list))
        self.nodes_expect = tweet_node_all.loc[self.all_label_news]

        tweets_nodes_res = self.nodes_expect.merge(tweets_have, left_index=True, right_on=['news_id', 'user_id'])

        return tweets_nodes_res

    # @staticmethod
    # def getUpftNodeUserMapping():
    #     with open('C:/Users/MSI/Downloads/gos_id_twitter_mapping.pkl', 'rb') as f:
    #         upfd_dic = pickle.load(f)  # gos_upfd_dic
    #     return upfd_dic

    def getTweetDataset(self):
        news_tweet_mapping = pd.read_csv(
            "{}/{}".format(self.config["dataset_root"], self.news_mapping_file))
        test_news_tweet_mapping = news_tweet_mapping.loc[
            news_tweet_mapping.apply(lambda x: x.id in self.all_label_news, axis=1)]
        return test_news_tweet_mapping

    def get_node_news_time_eq(self, news, tweets, node_user, l1):
        # actually you should merge/map here
        # get the shares _tweets that match the l1 length
        tweets = tweets[:l1]
        # loop through the shares_tweets
        # get their time
        pub_times = []
        for tweet in tweets:
            try:
                with open("{}/{}/tweets/{}.json".format(self.config["news_file"], news, tweet), 'r') as f:
                    tweet_j = json.load(f)
                pub_times.append(datetime.strptime(tweet_j["created_at"], '%a %b %d %H:%M:%S +0000 %Y'))
            except Exception as ex:
                print(ex)

                
        # zip tweets with pub_time
        tweet_time_tup = zip(pub_times, tweets)
        # sort tweet_time_tup by pub_times
        tweet_time_tup = sorted(tweet_time_tup)
        tweets = [t for p, t in list(tweet_time_tup)]
        tup = zip(node_user, tweets)
        node_news_time = pd.DataFrame(tup, columns=self.node_tweet_mapping.columns[:2])
        node_news_time["pub_time"] = [p for p, t in list(tweet_time_tup)]
        return node_news_time

    def mapEqualLenTweetNode(self, test_res, test_news_tweet_mapping):
        # test_res <- mergeNodeTweet #test_news_tweet_mapping <-getTweetDataset

        for news in tqdm(self.all_label_news):
            try:
                news_users = test_res.loc[(news, slice(None)),
                                          list(test_res.columns)]
                for i in range(len(news_users)):
                    news_user = news_users.iloc[i]
                    node_user = news_user['node_id']
                    l1 = len(node_user)
                    shares = news_user['tweet_ids']
                    # getting tweets among the shares
                    shares_tweets = [str(share) for share in shares if share in \
                                     list(test_news_tweet_mapping[test_news_tweet_mapping["id"] == news]['tweet_ids'])[
                                         0].split(
                                         '\t')]
                    shares_retweets = [str(share) for share in shares if share not in \
                                       list(
                                           test_news_tweet_mapping[test_news_tweet_mapping["id"] == news]['tweet_ids'])[
                                           0].split('\t')]

                    shares_flt = [float(t) for t in shares]

                    if (len(shares) >= l1) & (np.isnan(shares_flt).any() == False):
                        node_news_time = self.get_node_news_time_eq(news, shares, node_user, l1)
                        self.node_tweet_mapping = self.node_tweet_mapping.append(node_news_time)

                    self.node_tweet_mapping.sort_values(by="node_id", inplace=True, ignore_index=True)
                    self.node_tweet_mapping.pub_time = pd.to_datetime(self.node_tweet_mapping.pub_time)
            except Exception as e:
                #print("err1",str(e))
                self.missing_nodes.extend(getMissingTweetNodeIds(self.nodes_expect.loc[(news,slice(None)),
                                                        (self.nodes_expect.columns)]["node_id"].tolist()))

    def get_node_news_time_uneq(self, news, node_range_news, tweets, node_user, l1):
        # actually you should merge/map here
        # get the shares _tweets that match the l1 length
        tweets = tweets[:l1]
        # loop through the shares_tweets
        # get their time
        pub_times = []
        for tweet in tweets:
            try:
                with open("{}/{}/tweets/{}.json".format(self.config["news_file"], news, tweet), 'r') as f:
                    tweet_j = json.load(f)
                pub_times.append(datetime.strptime(tweet_j["created_at"], '%a %b %d %H:%M:%S +0000 %Y'))
            except Exception as ex:
                print(ex)

        # zip tweets with pub_time
        tweet_time_tup = zip(pub_times, tweets)
        # sort tweet_time_tup by pub_times
        tweet_time_tup = sorted(tweet_time_tup)
        neighs = []
        ls_map = list(self.node_tweet_mapping["node_id"])
        ls_map_have = [n for n in ls_map if n in node_range_news]
        lmap = len(ls_map_have)

        if lmap != 0:
            for node in node_user:
                for n in range(lmap):
                    if node < ls_map_have[n]:
                        if n != 0:
                            neighs.append([node, (ls_map.index(ls_map_have[n - 1]), ls_map.index(ls_map_have[n]))])
                        else:
                            neighs.append([node, (None, ls_map.index(ls_map_have[n]))])
                        break
                    elif n == lmap - 1:
                        neighs.append([node, (ls_map.index(ls_map_have[n]), None)])
        else:
            for r in range(len(tweet_time_tup)):
                neighs.append([node_user[r], (None, None)])

        node_ls = []
        tweets_ls = []
        pub_time_ls = []
        for neigh in neighs:
            for tweet_time in list(tweet_time_tup):
                time_left_neigh = \
                    list(
                        self.node_tweet_mapping[self.node_tweet_mapping["node_id"] == ls_map[neigh[1][0]]]["pub_time"])[
                        0] if neigh[1][0] is not None \
                        else datetime.fromisoformat('2006-03-21')
                time_right_neigh = \
                    list(
                        self.node_tweet_mapping[self.node_tweet_mapping["node_id"] == ls_map[neigh[1][1]]]["pub_time"])[
                        0] if neigh[1][1] is not None \
                        else datetime.now()

                if (str(time_left_neigh) < str(tweet_time[0])) & (str(time_right_neigh) >= str(tweet_time[0])):
                    node_ls.append(neigh[0])
                    tweets_ls.append(tweet_time[1])
                    pub_time_ls.append(tweet_time[0])

                    break

        node_news_time = pd.DataFrame(columns=self.node_tweet_mapping.columns)
        node_news_time["node_id"] = node_ls
        node_news_time["tweet_id"] = tweets_ls
        node_news_time["pub_time"] = pub_time_ls

        return node_news_time


    def mapUnequalLenTweetNode(self, test_res, test_news_tweet_mapping):

        for news in tqdm(self.all_label_news):
            try:
                news_users = test_res.loc[(news, slice(None)),
                                          list(test_res.columns)]

                node_range_news = []
                for u in news_users["node_id"].tolist():
                    for n in u:
                        node_range_news.append(n)

                for i in range(len(news_users)):
                    news_user = news_users.iloc[i]
                    node_user = news_user['node_id']
                    l1 = len(node_user)
                    shares = news_user['tweet_ids']
                    shares_tweets = [str(share) for share in shares if share in \
                                     list(test_news_tweet_mapping[test_news_tweet_mapping["id"] == news]['tweet_ids'])[
                                         0].split(
                                         '\t')]
                    shares_flt = [float(t) for t in shares]

                    if (len(shares) < l1) & (np.isnan(shares_flt).any() == False):
                        # print(shares)
                        node_news_time = self.get_node_news_time_uneq(news,node_range_news ,shares, node_user, len(shares))
                        self.node_tweet_mapping = self.node_tweet_mapping.append(node_news_time)

                    self.node_tweet_mapping.sort_values(by="node_id", inplace=True, ignore_index=True)
            except Exception as ex:
                #print(str(ex))
                self.missing_nodes.extend(getMissingTweetNodeIds(self.nodes_expect.loc[(news,slice(None)),
                                                        (self.nodes_expect.columns)]["node_id"].tolist()))

    def getNewsNodeMapping(self):
        return pd.read_csv("{}/node_article_mappings/{}_node_article_mapping.csv".format(self.config["init_dir_root"],self.config["dataset"][:3]))

    def fillMissingTweets(self, node_article_df, node_article_have_df):

        def getUpfd():
            with open('{}/{}_id_twitter_mapping.pkl'.format(self.config["pickle_root"],self.config["dataset"][:3]), 'rb') as f:
                upfd_dic = pickle.load(f)
            return list(upfd_dic.keys())[-1]

        def expand(start_node, end_node, tweets):
            # global node_news_mapping_all
            # print(start_node," ",end_node)
            node_news_unavailable = pd.DataFrame(columns=self.node_tweet_mapping.columns)
            gap = (end_node - start_node) + 1
            # print(list(tweets[0]["tweet_id"])[0])
            for k in range(start_node, start_node + int(gap / 2) + gap % 2):
                node_news_unavailable = node_news_unavailable.append({"node_id": float(k),
                                                                      "tweet_id": tweets[0]["tweet_id"],
                                                                      "pub_time": tweets[0]["pub_time"]},
                                                                     ignore_index=True)
            if len(tweets) == 2:
                nxt = 1
            else:
                nxt = 0

            for j in range(start_node + int(gap / 2) + gap % 2, end_node + 1):
                # global node_news_mapping_all
                node_news_unavailable = node_news_unavailable.append({"node_id": float(j),
                                                                      "tweet_id": tweets[nxt]["tweet_id"],
                                                                      "pub_time": tweets[nxt]["pub_time"]},
                                                                     ignore_index=True)
            return node_news_unavailable

        def appendTodf(node_news_unavailable):
            # nonlocal node_news_mapping_aft_2
            self.node_tweet_mapping = self.node_tweet_mapping.append(node_news_unavailable)

        for i in tqdm(range(len(node_article_have_df))):
            try:
                news_article_node = node_article_have_df.iloc[i]["node_id"]
                # if i != len(node_article_have_df)-1:
                news_article_node_end_inx = node_article_df[node_article_df["node_id"] == news_article_node].index[
                                                0] + 1
                if news_article_node_end_inx != len(node_article_df):
                    next_news_article_node = node_article_df.iloc[news_article_node_end_inx]["node_id"]

                elif news_article_node_end_inx == len(node_article_df):
                    next_news_article_node = getUpfd() + 1

                # first node in article
                first_node_article = news_article_node + 1
                # last node in article
                last_node_article = next_news_article_node - 1

                node_range = range(first_node_article, next_news_article_node)
                node_have_range_df = self.node_tweet_mapping.loc[
                    self.node_tweet_mapping.apply(lambda x: x.node_id in node_range, axis=1)]
                node_have_range_ls = node_have_range_df["node_id"].tolist()

                nodes_have_df_srt = node_have_range_df.sort_values(by="pub_time", ignore_index=True)

                node_have_range_ls.sort()
                # get the first node we have in range
                # complete if there is any gap

                for i in range(len(node_have_range_ls)):
                    if i == 0:
                        if node_have_range_ls[0] != first_node_article:
                            appendTodf(
                                expand(int(first_node_article), int(node_have_range_ls[0]) - 1,
                                       [nodes_have_df_srt.iloc[0]])
                            )
                            # expand first available node

                    if i == len(node_have_range_ls) - 1:
                        if node_have_range_ls[i] != last_node_article:
                            # expand the last  available node
                            appendTodf(
                                expand(int(node_have_range_ls[i] + 1), int(last_node_article),
                                       [nodes_have_df_srt.iloc[-1]])
                            )

                    if i != len(node_have_range_ls) - 1:
                        #           node_news_unavailable = interExpand(i)
                        node_down = int(node_have_range_ls[i + 1])
                        node_up = int(node_have_range_ls[i])
                        node_down_time = \
                            list(self.node_tweet_mapping[self.node_tweet_mapping["node_id"] == node_down]["pub_time"])[
                                0]
                        node_up_time = \
                            list(self.node_tweet_mapping[self.node_tweet_mapping["node_id"] == node_up]["pub_time"])[0]
                        if node_down - node_up > 1:
                            if node_up_time <= node_down_time:
                                appendTodf(
                                    expand(node_up + 1, node_down - 1, [
                                        self.node_tweet_mapping[
                                            self.node_tweet_mapping["node_id"] == node_up].squeeze(),
                                        self.node_tweet_mapping[
                                            self.node_tweet_mapping["node_id"] == node_down].squeeze()])
                                )
                            elif node_up_time > node_down_time:
                                appendTodf(
                                    expand(node_up + 1, node_down - 1, [
                                        self.node_tweet_mapping[
                                            self.node_tweet_mapping["node_id"] == node_down].squeeze(),
                                        self.node_tweet_mapping[
                                            self.node_tweet_mapping["node_id"] == node_up].squeeze()])
                                )
            except Exception as ex:
                print(str(ex))

    def getTweetNodeMap(self):
        test_res = self.mergeNodeTweet()
        test_news_tweet_mapping = self.getTweetDataset()

        self.mapEqualLenTweetNode(test_res, test_news_tweet_mapping)
        self.mapUnequalLenTweetNode(test_res, test_news_tweet_mapping)
        # get node_article_df
        # node_article_df contains the news_id node_id mapping
        node_article_df = self.getNewsNodeMapping()
        node_article_have_df = node_article_df.loc[
            node_article_df.apply(lambda x: x.news_article in self.all_label_news, axis=1)]
        node_article_have_df.reset_index(inplace=True, drop=True)

        self.fillMissingTweets(node_article_df, node_article_have_df)
        #get the missing nodes
        #get the total num of  nodes expected 
        len_total_nodes = len(getMissingTweetNodeIds(self.nodes_expect.loc[(slice(None),slice(None)),
                                                            (self.nodes_expect.columns)]["node_id"].tolist()))
        # check whether the tweet_node_mapping/df_missing_nodes/{}_missing_nodes.txt exist 
        # if not create it else leave it 
        # apppend self.missing missing nodes to the text file 
        dir = f"{self.config['init_dir_root']}/tweet_node_mapping/df_missing_nodes"
        create_dir(dir)
        txt_file_path = "{}/{}_missing_nodes.txt".format(dir,self.config["dataset"][:3])
        
        with open(txt_file_path,"a") as txtfile:
            for node in list(set(self.missing_nodes)):
                txtfile.write(str(node)+"\n")

        print("{} number of nodes missing out of total of {}".format(len(set(self.missing_nodes)),
                                                                    len_total_nodes))
        return self.node_tweet_mapping
