import numpy as np
import pandas as pd
import json
from util.labeled_users import getLabeledUsers
from multiprocessing.pool import Pool
from tqdm import tqdm


class ExtractMentionGraphIndex:
    def __init__(self, config):
        self.config = config
        self.label = None

    def search_timeline(self, users):
        graph_index_news = np.array([[], []], dtype='int64')

        # in graph_index_news
        # 1st row has source users
        # 2nd row has destination users

        def match_link(dest_user_id):
            if dest_user_id in users[1:]:
                nonlocal graph_index_news
                graph_index_news = np.append(graph_index_news, values=[[users[0]], [dest_user_id]], axis=1)

        try:
            with open(
                    "{}/{}_{}_all/user_timeline_tweets/{}.json".format(
                        self.config["root_upfd"],self.config["dataset"],self.label, 
                        users[0]
                    ),
                    'r') as f:
                user_timeline_tweets = json.load(f)
            for tweet in user_timeline_tweets:
                user_mentions = tweet['entities']['user_mentions']
                if len(user_mentions) != 0:
                    for user_dict in user_mentions:
                        match_link(user_dict['id'])
        except Exception as e:
            pass
            # print(f'{users[0]} timeline is not available')
        if not np.array_equal(graph_index_news, np.array([[], []])):
            return graph_index_news.tolist()

    def search_news(self, news):
        user_arr = news
        result = np.array([[], []])

        # in graph_index_news
        # 1st row has source users
        # 2nd row has destination users

        def permut(i):
            head = [user_arr[i]]
            tail = np.delete(user_arr, i)
            return np.append(head, tail).tolist()

        timeline_search_list = [permut(i) for i in range(len(user_arr))]

        pool = Pool(self.config["num_process"])
        res = pool.map(self.search_timeline, iterable=timeline_search_list)
        pool.close()
        pool.join()

        for el in res:
            if el is not None:
                result = np.append(np.array(result), np.array(el), axis=1).astype(np.int64)
        return result.tolist()

    def getMentionGraphIndex(self):
        # get labeled users
        for label in self.config["label"]:
            self.label = label
            labeled_news_user_groups = getLabeledUsers(self.config["dataset"], label)
            mention_graph_indices = []
            for i in tqdm(range(len(labeled_news_user_groups))):
                mention_graph_indices.append(self.search_news(
                    list(labeled_news_user_groups.iloc[i]["user_id"]))
                )
            labeled_news_user_mentionGraphs = labeled_news_user_groups.copy()
            labeled_news_user_mentionGraphs["mention_graph_indices"] = mention_graph_indices
            # dump labeled_news_user_mentionGraphs
            labeled_news_user_mentionGraphs.to_csv("{}/{}_{}_news_mentionGraph.csv".format(self.config["dump_root"],
                                                                                           self.config["dataset"][:3],
                                                                                           label))
