# from embed_users import fill_users
#
#
# def search_timeline(users):
#     graph_index_news = np.array([[], []], dtype='int64')
#
#     # in graph_index_news
#     # 1st row has source users
#     # 2nd row has destination users
#
#     def match_link(dest_user_id):
#         if dest_user_id in users[1:]:
#             nonlocal graph_index_news
#             graph_index_news = np.append(graph_index_news, values=[[users[0]], [dest_user_id]], axis=1)
#
#     try:
#         with open(
#                 f"C:/Users/MSI/Fakenewsnet-hpc/FakeNewsNet-master/code/"
#                 f"upfd_dataset/gossipcop_users/real/user_timeline_tweets/{users[0]}.json",
#                 'r') as f:
#             user_timeline_tweets = json.load(f)
#         for tweet in user_timeline_tweets:
#             user_mentions = tweet['entities']['user_mentions']
#             if len(user_mentions) != 0:
#                 for user_dict in user_mentions:
#                     match_link(user_dict['id'])
#     except Exception as e:
#         pass
#         # print(f'{users[0]} timeline is not available')
#     if not np.array_equal(graph_index_news, np.array([[], []])):
#         return graph_index_news.tolist()
#
#
# def search_news(news):
#     user_arr = news
#     result = np.array([[], []])
#
#     # in graph_index_news
#     # 1st row has source users
#     # 2nd row has destination users
#
#     def permut(i):
#         head = [user_arr[i]]
#         tail = np.delete(user_arr, i)
#         return np.append(head, tail).tolist()
#
#     timeline_search_list = [permut(i) for i in range(len(user_arr))]
#
#     pool = Pool(6)
#     res = pool.map(search_timeline, iterable=timeline_search_list)
#
#     pool.close()
#     pool.join()
#
#     for el in res:
#         if el is not None:
#             result = np.append(np.array(result), np.array(el), axis=1).astype(np.int64)
#     print(result.tolist())
#
#
# def main():
#     node_user_news_mapping = pd.read_csv(
#         'C:/Users/MSI/FakeNewsNet-hpc/FakeNewsNet-master/utils/gos_node_user_news_mapping.csv')
#     news_user_groups = node_user_news_mapping.groupby('news_id', sort=False)['user_id'].apply(list)
#     search_news(list(set(news_user_groups[7])))
#
#
# if __name__ == "__main__":
#     fill_users()
import os
from TweetRetweetMerge import TweetRetweetMerge
from helpers.createUser_News_bow import createUser_News_bow
from fillMissingProfileTimeline import fillMissingProfileTimeline
from TweetNodeMapping import TweetNodeMapper
from ExtractMentionGraphIndex import ExtractMentionGraphIndex


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

