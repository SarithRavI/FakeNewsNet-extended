import pandas as pd
import json
from tqdm import tqdm
from multiprocessing import Pool
import os
import random


class TweetRetweetMerge:
    def __init__(self, config):
        self.config = config
        # gos_node_user_news_mapping.csv
        self.config['node_user_news_mapping_file'] = f'{self.config["init_dir_root"]}/node_user_mappings/{self.config["dataset"][:3]}_node_user_news_mapping.csv'


    # root_test1 = "{}/{}".format(config['root_utils'], config['news_file'])

    def collect_tweets_news(self, news_id):
        tweet_uid_ls = []
        try:
            tweet_files = os.listdir("{}/{}/tweets".format(self.config['news_file'], news_id))
            for tweet in tweet_files:
                with open("{}/{}/tweets/{}".format(self.config['news_file'], news_id, tweet), 'r') as f1:
                    tweet_j = json.load(f1)
                tweet_uid = tweet_j['user']['id']
                tweet_uid_ls.append(tweet_uid)
                if int(tweet_j['retweet_count']) > 0:
                    try:
                        with open("{}/{}/retweets/{}".format(self.config['news_file'], news_id, tweet)) as f2:

                            retweet_j = json.load(f2)
                        for i in range(len(retweet_j['retweets'])):
                            retweet_item = retweet_j['retweets'][i]
                            tweet_uid_ls.append(retweet_item['user']['id'])
                            # dump json version of this retweet in tweet file location using retweet id
                            try:
                                with open(
                                        "{}/{}/tweets/{}.json".format(self.config['news_file'], news_id,
                                                                      retweet_item["id"]),
                                        'x') as f3:
                                    json.dump(retweet_item, f3)
                            except Exception as ex:
                                # print('ex1', ex)
                                pass
                    except FileNotFoundError as ex:
                        # print(f'retweet file for tweet {tweet} from {news_id} cannot open')
                        pass

        except Exception as ex:
            # print(ex)
            pass
        finally:
            return tweet_uid_ls

    def get_missing_uid_ls(self, news_uid, news_user_group):
        return list((set(news_uid) | set(news_user_group)) - set(news_uid))

    def aggregate(self, news_ls):
        node_user_news_mapping = pd.read_csv("{}".format(self.config['node_user_news_mapping_file']))
        news_user_groups = node_user_news_mapping.groupby('news_id', sort=False)['user_id'].apply(list)

        pool = Pool(self.config['num_process'])
        pbar = tqdm(total=len(news_ls))
        pbar.set_description('Bringing retweets to main..')

        def update(arg):
            pbar.update()

        res = []
        for i in range(pbar.total):
            res_ = pool.map_async(self.collect_tweets_news, [news_ls[i]], callback=update)
            res.append(res_.get()[0])
        pool.close()
        pool.join()
        missing_uid_ls = [self.get_missing_uid_ls(res[n], news_user_groups[news_ls][n]) for n in range(len(news_ls))]
        return missing_uid_ls

    def generate_missing(self, news, missing_uids):
        try:
            tweets = os.listdir("{}/{}/tweets".format(self.config['news_file'], news))
            uids = missing_uids
            if len(uids) <= len(tweets):
                sample = random.sample(tweets, len(uids))  # to avoid sample larger than population or is negative
            else:
                sample = random.sample(tweets + tweets[:(len(uids) - len(tweets))], len(uids))
            for i in range(len(sample)):
                with open("{}/{}/tweets/{}".format(self.config['news_file'], news, sample[i])) as f:
                    tweet_j = json.load(f)
                    tweet_j['user']['id'] = uids[i]
                    tweet_j['user']['id_str'] = str(uids[i])
                    random_tweet_id = "r{}_{}".format(i, random.randint(100000, 200000))
                    try:
                        with open("{}/{}/tweets/{}.json".format(self.config['news_file'], news, random_tweet_id),
                                  'x') as f:
                            json.dump(tweet_j, f)
                    except Exception as e:
                        # print('pos2', str(e))
                        pass
        except Exception as e:
            # print(e)
            pass

    def generate(self, missing):
        pool = Pool(self.config['num_process'])
        pbar = tqdm(total=len(missing))
        pbar.set_description('Generating tweets for missing users..')

        def update(arg):
            pbar.update()

        for i in range(pbar.total):
            pool.apply_async(self.generate_missing, args=missing[i], callback=update)
        pool.close()
        pool.join()

