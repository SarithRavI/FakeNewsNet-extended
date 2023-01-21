from enum import Enum
from user_tweet_data_extraction import extract_tweet_data
from user_pg_data_extraction import extract_pg_data
from user_spacy_data_extraction import extract_spacy_data
import argparse

class Dataset(Enum):
    GOS = "gossipcop"
    POL = "politifact"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Initiating the postprocessing')
    parser.add_argument('--dataset', type=str, default=Dataset.GOS.value,
                        help='which dataset to be used: pol for politifact. gos for gossipcop')

    args = parser.parse_args()
    print(args)

    if 'gos' in args.dataset  :
        dataset =[Dataset.GOS.value]
    elif 'pol' in args.dataset  :
        dataset = [Dataset.POL.value]
    elif args.dataset == 'all':
        dataset = [Dataset.GOS.value, Dataset.POL.value]

    for ds in dataset:
        # spacy data transformation
        extract_spacy_data(ds)
        # tweet data transformation
        extract_tweet_data(ds)
        # mention graph data transformation
        extract_pg_data(ds)





