import sys
import pandas as pd
import numpy as np
import random
import os
import json
import pdb


np.random.seed(42)
random.seed(42)

# Divide the entire dataset into a training set and a test set.
def data_allocation(num_clients, diff_quantity, num_communication_rounds):
    num_clients = int(num_clients)
    diff_quantity = int(diff_quantity)
    df = pd.read_json("new-databricks-dolly-15k.json", orient='records')
    sorted_df = df.sort_values(by=['category'])
    grouped = sorted_df.groupby('category')
    sampled_df = grouped.apply(lambda x: x.sample(n=10))
    sampled_df = sampled_df.reset_index(level=0, drop=True)
    remaining_df = sorted_df.drop(index=sampled_df.index)

    sampled_df = sampled_df.reset_index().drop('index', axis=1)
    remaining_df = remaining_df.reset_index().drop('index', axis=1)
    data_path = os.path.join("data", str(num_clients))

    os.makedirs(data_path,exist_ok=True)

    remaining_df_dic = remaining_df.to_dict(orient='records')
    with open(os.path.join(data_path, "global_training.json"), 'w') as outfile:
        json.dump(remaining_df_dic, outfile)

    sampled_df_dic = sampled_df.to_dict(orient='records')
    with open(os.path.join(data_path, "global_test.json"), 'w') as outfile:
        json.dump(sampled_df_dic, outfile)

    # Partition the global training data into smaller subsets for each client's local training dataset

    if diff_quantity: #each client receives a balanced representation of the categories
        min_size = 0
        min_require_size = 40
        alpha = 0.5

        N = len(remaining_df)
        net_dataidx_map = {}
        category_uniques = remaining_df['category'].unique().tolist()
        while min_size < min_require_size:

            idx_partition = [[] for _ in range(num_clients)]
            for k in range(len(category_uniques)):
                category_rows_k = remaining_df.loc[remaining_df['category'] == category_uniques[k]]
                category_rows_k_index = category_rows_k.index.values
                np.random.shuffle(category_rows_k_index)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_partition)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(category_rows_k_index)).astype(int)[:-1]
                idx_partition = [idx_j + idx.tolist() for idx_j, idx in
                                    zip(idx_partition, np.split(category_rows_k_index, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_partition])

            print(min_size)


    else: 

        if num_communication_rounds <= 0:
            raise ValueError("num_communications should be greater than 0.")
        
        num_shards_per_clients = 2
        remaining_df_index = remaining_df.index.values
        shards = np.array_split(remaining_df_index, int(num_shards_per_clients * num_clients))
        random.shuffle(shards)
        shards = [shards[i:i + num_shards_per_clients] for i in range(0, len(shards), num_shards_per_clients)]
        idx_partition = [np.concatenate(shards[n]).tolist() for n in range(num_clients)]
        
        
        start_idx = 0
        num_records = len(remaining_df)
        records_per_round = num_records // num_communication_rounds

        for communication_round in range(num_communication_rounds):

            end_idx = start_idx + records_per_round

            if communication_round == num_communication_rounds - 1:
                end_idx = num_records

            for client_id, idx in enumerate(idx_partition):
                print("\n Generating the local training dataset of Client_{} in Communication Round {}".format(client_id, communication_round+1))
                sub_remaining_df = remaining_df.loc[idx]
                sub_remaining_df = sub_remaining_df.reset_index().drop('index', axis=1)
                sub_remaining_df_dic = sub_remaining_df.to_dict(orient='records')
                
                num_records = len(sub_remaining_df_dic)
                records_per_round = num_records // num_communication_rounds
                end_idx = start_idx + records_per_round

                with open(os.path.join(data_path, "local_training_{}_comm_{}.json".format(client_id, communication_round+1)), 'w') as outfile:
                    json_data_slice = sub_remaining_df_dic[start_idx:end_idx]
                    json.dump(json_data_slice, outfile)

            start_idx = end_idx








'''else: #randomly splits the data into shards and assigns them to different clients
        num_shards_per_clients = 2
        remaining_df_index = remaining_df.index.values
        shards = np.array_split(remaining_df_index, int(num_shards_per_clients * num_clients))
        random.shuffle(shards)

        shards = [shards[i:i + num_shards_per_clients] for i in range(0, len(shards), num_shards_per_clients)]
        idx_partition = [np.concatenate(shards[n]).tolist() for n in range(num_clients)]


    for client_id, idx in enumerate(idx_partition):
        print(
            "\n Generating the local training dataset of Client_{}".format(client_id)
        )
        sub_remaining_df = remaining_df.loc[idx]
        sub_remaining_df = sub_remaining_df.reset_index().drop('index', axis=1)
        sub_remaining_df_dic = sub_remaining_df.to_dict(orient='records')

        with open(os.path.join(data_path, "local_training_{}.json".format(client_id)), 'w') as outfile:
            json.dump(sub_remaining_df_dic, outfile) '''
