from TransformHelpers.PostprocessTextData import PostData
from user_tweet_data_extraction import extract_tweet_data,extract_spacy_data
from user_pg_data_extraction import extract_pg_data

def postProcessTextFeatures(ds):  # type is vis or spacy
    config = {"root_tweet_node_mapping": "../utils/tweet_node_mapping",
              "root_node_user_mapping": "../utils/node_user_mappings",
              "root_upfd_data": "../code/upfd_dataset",
              "dataset": ds,
              "num_process": 40,
              }
    config["dump_location_vis"] = "visualization_data/{}".format(config["dataset"])
    config["dump_location_model"] = "train_test_data/{}".format(config["dataset"])
    postProcessing = PostData(config)
    postProcessing.processVisData()

def postProcessSpacyEmbeddings(ds):  # type is vis or spacy
    config = {"root_tweet_node_mapping": "../utils/tweet_node_mapping",
              "root_node_user_mapping": "../utils/node_user_mappings",
              "root_upfd_data": "../code/upfd_dataset",
              "dataset": ds,
              "num_process": 40,
              }
    config["dump_location_vis"] = "visualization_data/{}".format(config["dataset"])
    config["dump_location_model"] = "train_test_data/{}".format(config["dataset"])
    postProcessing = PostData(config)
    postProcessing.processSpacy()
 



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # startup tweet extraction
    # postProcess("gossipcop")
    # extract_tweet_data("gossipcop")

    # startup pg extraction
    # postProcess("gossipcop")
    # extract_pg_data("gossipcop")

    # startup spacy data extraction
    postProcessSpacyEmbeddings("gossipcop")
    extract_spacy_data("gossipcop")


