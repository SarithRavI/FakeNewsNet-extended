from user_tweet_data_extraction import extract_tweet_data,extract_spacy_data
from user_pg_data_extraction import extract_pg_data


if __name__ == '__main__':

    extract_spacy_data("gossipcop")

    extract_tweet_data("gossipcop")

    extract_pg_data("gossipcop")





