from PostProcessData import PostData
from user_tweet_data_extraction import extract_tweet_data
from user_pg_data_extraction import extract_pg_data

def postProcess(ds):  # type is vis or spacy
    config = {"root_tweet_node_mapping": "../utils/tweet_node_mapping",
              "root_node_user_mapping": "../utils/node_user_mappings",
              "root_upfd_data": "../code/upfd_dataset",
              "dataset": ds,
              "num_process": 40,
              # "dump_location_vis": "visualization_data",
              # "dump_location_model": "",
              }
    config["dump_location_vis"] = "visualization_data/{}".format(config["dataset"])
    config["dump_location_model"] = "train_test_data/{}".format(config["dataset"])
    postProcessing = PostData(config)
    postProcessing.processVisData()

def postProcessSpacy(ds):  # type is vis or spacy
    config = {"root_tweet_node_mapping": "../utils/tweet_node_mapping",
              "root_node_user_mapping": "../utils/node_user_mappings",
              "root_upfd_data": "../code/upfd_dataset",
              "dataset": ds,
              "num_process": 40,
              # "dump_location_vis": "visualization_data",
              # "dump_location_model": "",
              }
    config["dump_location_vis"] = "visualization_data/{}".format(config["dataset"])
    config["dump_location_model"] = "train_test_data/{}".format(config["dataset"])
    postProcessing = PostData(config)
    postProcessing.processSpacy()
 



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    postProcess("gossipcop")
    # extract_tweet_data("gossipcop")
    extract_pg_data("gossipcop")


