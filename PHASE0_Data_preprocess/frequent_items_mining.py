import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import os
import sys
import json
import numpy as np
import argparse
from glob import glob
__file__ = "frequent_itmes_mining.py"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def frequent_itemsets_generation(plan_out_data_name, random_out_data_name, min_sup, min_len, top_N):
    transactions = []
    num_elements = []
    unique_items = []
    file_list = glob("../Data/2021_SEQ/*.csv")
    target_list = []
    for file_name in file_list:
        str_list = file_name.split("_")
        if str_list[-1].split(".")[0] in ["valid", "test"]:
            target_list.append(file_name)

    for file in target_list:
        f = open(file, "r")
        lines = f.readlines()
        for line in lines:
            dic =eval(line)
            trans = dic["transaction"]
            eos = trans[-1]
            bos = trans[0]
            elements = [str(i) for i in trans if i not in [eos, bos]]
            num_elements.append(len(elements))
            transactions.append(elements)
            unique_items = unique_items + elements
            unique_items = list(set(unique_items))

    Trans_encoder = TransactionEncoder()
    te_ary = Trans_encoder.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=Trans_encoder.columns_)
    frequent_itemsets = fpgrowth(df, min_support=min_sup, use_colnames=True) # mimimun support value setting
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    filted_frequent_itemsets = frequent_itemsets[(frequent_itemsets['length'] >= min_len)].copy() # minimum lenght setting
    filted_frequent_itemsets.sort_values(inplace = True, by = "support", ascending= False)
    filted_frequent_itemsets.reset_index(inplace = True, drop = True)
    
    select_list = []
    for idx in range(len(filted_frequent_itemsets)):
        itemsets = filted_frequent_itemsets.loc[idx,"itemsets"]
        unique_list = []
        for sets in itemsets:
            set_dic = eval(sets)
            unique_list.append(tuple((set_dic['row'], set_dic["column"])))
            unique_list = list(set(unique_list))
        if len(itemsets) == len(unique_list):
            select_list.append(itemsets)
            
    json_dic = {}
    if top_N == len(select_list):
        top_N = len(select_list)

    for s_idx in range(top_N):
        item_set = select_list[s_idx]
        fi_list = list(item_set)
        temp = df[fi_list]
        all_True = temp.any(axis='columns')
        idx_list = all_True[all_True == True].index.tolist()
        select_trans = [transactions[idx] for idx in idx_list]
        json_dic[s_idx] = {
            "planned_item_sets" : fi_list,
            "transactions" : select_trans
        }

    with open("../Data/Dataset_usermodel/{}".format(plan_out_data_name), "w") as jsonfile:
        json.dump(json_dic,jsonfile)
    
    #generate random sequence item to train cases of reject.
    random_dic = {}
    for key in json_dic.keys():
        temp = {'transactions': []}
        for i in range(len(json_dic[key]['transactions'])):
            temp['transactions'].append([unique_items[index] for index in np.random.choice(len(unique_items),18)])
        random_dic[key] = temp
    with open("../Data/Dataset_usermodel/{}".format(random_out_data_name), "w") as ran_jsonfile:
        json.dump(random_dic,ran_jsonfile)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--minsup", type=float, default=0.0001)
    parser.add_argument("--len", type=int, default=3)
    parser.add_argument("--topN", type=int, default=30)
    args = parser.parse_args()

    plan_out_data_name = "planned_items_transaction_UserModel.json"
    random_out_data_name = "random_items_transaction_UserModel.json"

    frequent_itemsets_generation(plan_out_data_name, random_out_data_name , args.minsup, args.len, args.topN)
    print("Done")
