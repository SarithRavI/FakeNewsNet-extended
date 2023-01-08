import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import ast 
import os

class PostData:
    def __init__(self,config):
        self.config = config
    
    def processMG(self):
        dataset = self.config['dataset']
        label = self.config['label']
        mg_features_df = pd.DataFrame()
        mg_raw_df = pd.read_csv(f"{self.config['root_mention_graphs']}/{dataset[:3]}_{label}_news_mentionGraph.csv")
        for _,row in tqdm(mg_raw_df.iterrows()):
            all_users = list(ast.literal_eval(row["user_id"]))
            graph_edge_index_ = ast.literal_eval(row["mention_graph_indices"])
            
            graph_edge_index= np.array(graph_edge_index_).T.tolist()
            diGraph = nx.DiGraph(graph_edge_index) 
            multiDiGraph = nx.MultiDiGraph(graph_edge_index)
            
            reversed_edge_index = [graph_edge_index_[1],graph_edge_index_[0]]
            reversed_edge_index= np.array(reversed_edge_index).T.tolist()
            reversed_diGraph =nx.DiGraph(reversed_edge_index)
            
            users_in_mg = list(set((graph_edge_index_[1])+graph_edge_index_[0]))
            
            for user in all_users:
                user_mg_feature =dict()
                if len(users_in_mg) != 0:
                    if user in users_in_mg:
                        in_degree = int(diGraph.in_degree(user))
                        out_degree = int(diGraph.out_degree(user))
                        w_in_degree = int(multiDiGraph.in_degree(user))
                        w_out_degree = int(multiDiGraph.out_degree(user))
                        shrt_path_map = nx.single_source_shortest_path_length(diGraph, user, cutoff=2)
                        shrt_path_map_rev = nx.single_source_shortest_path_length(reversed_diGraph, user, cutoff=2)
                        hop_2_in = 0
                        hop_2_out = 0
                        for key in list(shrt_path_map.keys()):
                            if shrt_path_map[key] ==2:
                                hop_2_in +=1
                        for key in list(shrt_path_map_rev.keys()):
                            if shrt_path_map_rev[key] == 2:
                                hop_2_out +=1
                        user_mg_feature = {"news_id":row["news_id"],
                                        "user_id":user,
                                        "in_degree":in_degree,
                                        "out_degree":out_degree,
                                        "weighted_in_degree":w_in_degree,
                                        "weighted_out_degree":w_out_degree,
                                        "hop_2_in":int(hop_2_in),
                                        "hop_2_out":int(hop_2_out)}
                    else:
                        user_mg_feature = {"news_id":row["news_id"],
                                        "user_id":user,
                                        "in_degree":0,
                                        "out_degree":0,
                                        "weighted_in_degree":0,
                                        "weighted_out_degree":0,
                                        "hop_2_in":0,
                                        "hop_2_out":0}
                else:
                    user_mg_feature = {"news_id":row["news_id"],
                                        "user_id":user,
                                        "in_degree":0,
                                        "out_degree":0,
                                        "weighted_in_degree":0,
                                        "weighted_out_degree":0,
                                        "hop_2_in":0,
                                        "hop_2_out":0}
                
                mg_features_df = mg_features_df.append(user_mg_feature,ignore_index=True)  

        target_root = self.config["dump_location"]
        target = os.path.join(target_root,"{}_{}_mg_features.csv".format(self.config["dataset"][:3], self.config['label']))
        mg_features_df.to_csv(target,index=False)
       