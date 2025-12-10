import pandas as pd

def cluster_based_filter(df):

    """
    Filters and processes a dataframe based on cluster annotations, removing uncertain 
    classifications and duplicates.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        - the O-Ster dataset
    
    Returns:
    --------
    df: pandas.DataFrame
        - dataset based on 5-cluster
    
    Output Files:
    -------------
    - "./O-Ster dataset/preprocessed/data_on10.csv": Deduplicated dataset based on 
      10-cluster 
    - "./O-Ster dataset/preprocessed/data_on5.csv": Deduplicated dataset based on 
      5-cluster 
    
    Notes:
    ------
    - Print statements show dataset size reduction at each step
    """

    df = df.dropna(subset=["cluster_10_ann05"])
    df = df[df["cluster_10_nome_ann02"] !='None/Doubt']
    df = df[df["cluster_5_nome_ann02"] !='None/Doubt']


    print("Dataframe without empty clusters", df.shape)
    df_on10 = df.drop_duplicates(subset=["id", "cluster_10_nome_ann01", "cluster_10_nome_ann02", "cluster_10_nome_ann05"])
    print("Dataframe based on 10 clusters", df_on10.shape)
    df_on10.to_csv("./O-Ster dataset/preprocessed/data_on10.csv", index=False)


    df_on5 = df.drop_duplicates(subset=["id", "cluster_5_nome_ann01", "cluster_5_nome_ann02", "cluster_5_nome_ann05"])
    print("Dataframe based on 5 clusters", df_on5.shape)
    df_on5.to_csv("./O-Ster dataset/preprocessed/data_on5.csv", index=False)

    return df_on5