import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sp

def doImpute(df):
    imputer= SimpleImputer(strategy="median")
    imputer.fit(df)
    x = imputer.transform(df)
    return x
    
def addNan(df,missing,news_node):
    all_new_nodes = missing+news_node
    for new_node in all_new_nodes:
        df.loc[new_node] = [np.nan]*len(df.columns)
    df.sort_index(inplace=True)
    return df

def addNanSpacy(arr,missing,news_node):
    all_new_nodes = missing+news_node
    all_new_nodes.sort()
    # print(arr.shape[1])
    for new_node in all_new_nodes:
        if int(new_node) > arr.shape[0]:
            arr = np.append(arr,np.array([[0]*(arr.shape[1])]),axis=0)
        else:
            # print(int(new_node),arr.shape[0])
            arr = np.insert(arr,int(new_node),[0]*(arr.shape[1]),axis=0)
    return arr


def ScaleToRange(df):
    min_max_scaler = MinMaxScaler()
    #min_max_scaler.fit(df)
    return pd.DataFrame(min_max_scaler.fit_transform(df),
                       columns = df.columns,
                       index = df.index)
def saveDf(df,loc):
    x= df.to_numpy()
    sparse_matrix = sp.csr_matrix(x)
    sp.save_npz('{}.npz'.format(loc), sparse_matrix)
    df.to_csv("{}.csv".format(loc))

def saveCompound(df,dataset,loc):
    x = df.to_numpy()
    X_u = sp.load_npz("../../../UPFD/{}/new_profile_feature.npz".format(dataset)).todense().astype(np.float32)
    all_x = np.hstack((X_u,x))    
    # print(x.shape)
    # print(X_u.shape)
    sparse_matrix = sp.csr_matrix(all_x)
    sp.save_npz('{}.npz'.format(loc), sparse_matrix)

def saveSpacyCompound(arr,dataset,loc):
    X_u = sp.load_npz("../../../UPFD/{}/new_profile_feature.npz".format(dataset)).todense().astype(np.float32)
    all_x = np.hstack((X_u,arr)) 
    # print(x.shape)
    # print(X_u.shape)
    print("Here I print 2: ",all_x.shape)
    sparse_matrix = sp.csr_matrix(all_x)
    sp.save_npz('{}.npz'.format(loc), sparse_matrix)