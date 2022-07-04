import tqdm
from map_generator import *
from sequence_generator import *
import numpy as np
import pandas as pd
from glob import glob
import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def preprocess_purchase_transaction(file_list, f_df, loc_df, DATA_STORAGE_PATH_TEMPORAL):

    count_idx_person = 0
    count_idx_korean = 0
    count_idx_japanese = 0
    count_idx_chinese = 0
    count_idx_western = 0

    for file in tqdm.tqdm(file_list, desc="Generating purchase transaction "):
        dic_inf = file.split("/")[7]
        df = pd.read_csv(file)
        df["BIZ_TYPE"] = df['BIZ_TYPE'].str.strip()
        remove1 = df[(df.SALE_QNT < 0) | (df.SALE_PRC < 100)]
        remove2 = df[(df.CUST_GUBUN == 2) & (df.BIZ_TYPE.isnull()
                                             == True) & (df.CUST_GRADE.isnull() == True)]
        remove3 = df[(df.CUST_GENDER != "M") & (df.CUST_GENDER != "F")]
        re1 = remove1[["SALE_DT", "TRXN_NO", "MEM_NUM",
                       "CUST_NO", "POS_TIME"]].drop_duplicates()
        re2 = remove2[["SALE_DT", "TRXN_NO", "MEM_NUM",
                       "CUST_NO", "POS_TIME"]].drop_duplicates()
        re3 = remove3[["SALE_DT", "TRXN_NO", "MEM_NUM",
                       "CUST_NO", "POS_TIME"]].drop_duplicates()
        re = pd.concat([re1, re2, re3])
        re.drop_duplicates(inplace=True)
        re.reset_index(inplace=True, drop=True)
        drop_idx = []

        for i in range(len(re)):
            temp_idx = df[(df.SALE_DT == re.SALE_DT[i]) & (df.TRXN_NO == re.TRXN_NO[i]) & (df.MEM_NUM == re.MEM_NUM[i]) & (
                df.POS_TIME == re.POS_TIME[i]) & (df.CUST_NO == re.CUST_NO[i])].index.tolist()
            drop_idx = drop_idx + temp_idx
        df = df.drop(drop_idx)
        new_df = pd.merge(df, f_df, on="PRODU_CODE", how="left")
        unique_trans = new_df[["SALE_DT", "TRXN_NO", "MEM_NUM",
                               "CUST_NO", "POS_TIME"]].drop_duplicates().reset_index(drop=True)

        for i in range(len(unique_trans)):
            temp = new_df[(new_df.SALE_DT == unique_trans.SALE_DT[i]) & (new_df.TRXN_NO == unique_trans.TRXN_NO[i]) & (
                new_df.MEM_NUM == unique_trans.MEM_NUM[i]) & (new_df.POS_TIME == unique_trans.POS_TIME[i]) & (new_df.CUST_NO == unique_trans.CUST_NO[i])]
            temp.reset_index(inplace=True, drop=True)
            path_list = []
            for j in range(len(temp)):
                good_loc = loc_df[(loc_df.대분류 == temp.LIG_NAME[j]) & (loc_df.중분류 == temp.MID_NAME[j]) & (
                    loc_df.소분류 == temp.SML_NAME[j])][["row", "column"]].values
                if len(good_loc) == 0:
                    pass
                else:
                    path_list.append(list(good_loc[0]))
            if len(temp) == len(path_list):
                temp_dic = {
                    "bio": {
                        "age": temp.loc[0, "CUST_GRADE"],
                        "sex": temp.loc[0, "CUST_GENDER"],
                        "type": temp.loc[0, "CUST_GUBUN"],
                        "business": temp.loc[0, "BIZ_TYPE"],
                        "CUST_NO": temp.loc[0, "CUST_NO"],
                        "MEM_NUM": temp.loc[0, "MEM_NUM"]
                    }
                }
                temp_transaction = []

                for a in range(len(temp)):
                    temp_transaction.append(
                        {
                            "상품명": temp.loc[a, "PRODU_NAME"],
                            "상품코드": temp.loc[a, "PRODU_CODE"],
                            "대분류": temp.loc[a, "LIG_CODE"],
                            "중분류": temp.loc[a, "MID_CODE"],
                            "소분류": temp.loc[a, "SML_CODE"],
                            "row": str(path_list[a][0]),
                            "column": str(path_list[a][1]),
                            "price": str(temp.loc[a, "SALE_PRC"])
                        }
                    )
                temp_dic["transaction"] = temp_transaction

                if str(temp.CUST_GUBUN[0]) == "1":
                    temp_dic["bio"]["business"] = "개인"
                    temp_dic["idx"] = count_idx_person
                    f = open(DATA_STORAGE_PATH_TEMPORAL +
                             "{}/Transaction_data_개인.csv".format(dic_inf), "a")
                    f.write(str(temp_dic))
                    f.write("\n")
                    f.close()
                    count_idx_person += 1

                elif str(temp.CUST_GUBUN[0]) == "2":
                    if str(temp.BIZ_TYPE[0]) in ["한식", "일식", "중식", "양식"]:
                        if temp.BIZ_TYPE[0] == "한식":
                            temp_dic["idx"] = count_idx_korean
                            count_idx_korean += 1
                        elif temp.BIZ_TYPE[0] == "일식":
                            temp_dic["idx"] = count_idx_japanese
                            count_idx_japanese += 1
                        elif temp.BIZ_TYPE[0] == "중식":
                            temp_dic["idx"] = count_idx_chinese
                            count_idx_chinese += 1
                        elif temp.BIZ_TYPE[0] == "양식":
                            temp_dic["idx"] = count_idx_western
                            count_idx_western += 1

                        f = open(DATA_STORAGE_PATH_TEMPORAL +
                                 "{0}/Transaction_data_{1}.csv".format(dic_inf, temp.BIZ_TYPE[0]), "a")
                        f.write(str(temp_dic))
                        f.write("\n")
                        f.close()
                else:
                    pass


def TSP_solver(sequence_trans, loc_df):
    Seq_Gen = Sequence_generation(loc_df)
    sequence = []
    for trans in sequence_trans:
        sequence.append((int(trans["row"]), int(trans["column"])))
    sub_list = [seq for seq in sequence if seq != 0]
    sort_idx_list = Seq_Gen.Grid_TSP_based_sequence_generation(
        (20, 0), sub_list, (14, 0))
    sort_idx_list.remove(len(sort_idx_list)-1)
    sort_idx_list.remove(0)
    sort_idx_list = list(np.array(sort_idx_list)-1)
    seq_list = [sequence_trans[idx] for idx in sort_idx_list]
    return seq_list


def preprocess_sequential_transaction(biz_type, max_length, s_idx, loc_df, DATA_STORAGE_PATH_TEMPORAL):
    f = open(DATA_STORAGE_PATH_TEMPORAL +
             "2021/Transaction_data_{}.csv".format(biz_type), "r")
    lines = f.readlines()
    total_size = int(len(lines)/20000)
    print(total_size)
    fw = open(DATA_STORAGE_PATH_TEMPORAL +
              "2021_SEQ/Transaction_data_{0}_{1}_{2}.csv".format(biz_type, max_length, s_idx), "a")
    for idx in tqdm.tqdm(range(s_idx*20000, s_idx*20000 + 20000)):
        if idx >= len(lines):
            break
        line = lines[idx]
        line = line.strip()
        line_dic = eval(line)
        transaction = line_dic["transaction"]
        temp_trans = TSP_solver(transaction, loc_df)
        temp_trans = [{"상품코드": "<bos>", "상품명": "<bos>",
                       "row": "20", "column": "0"}] + temp_trans
        if len(temp_trans) >= max_length-1:
            temp_trans = temp_trans[:max_length-1]
            line_dic["transaction"] = temp_trans + \
                [{"상품코드": "<eos>", "상품명": "<eos>", "row": "14", "column": "0"}]
        else:
            temp_trans = temp_trans + \
                [{"상품코드": "<eos>", "상품명": "<eos>", "row": "14", "column": "0"}]
            line_dic["transaction"] = temp_trans + \
                [{"상품코드": "<eos>", "상품명": "<eos>", "row": "14",
                    "column": "0"}]*(max_length - len(temp_trans))
        fw.write(str(line_dic))
        fw.write("\n")
    fw.close()


def preprocess_map_generation(biz_type, maxlen, view_range, sdx,  map_arr, loc_df, DATA_STORAGE_PATH_TEMPORAL):

    if not os.path.exists(DATA_STORAGE_PATH_TEMPORAL + "/2021_MAP"):
        os.mkdir(DATA_STORAGE_PATH_TEMPORAL + "/2021_MAP")
    Opt = OptPath(loc_df)
    f = open(DATA_STORAGE_PATH_TEMPORAL +
             "2021_SEQ/Transaction_data_{0}_{1}_{2}.csv".format(biz_type, maxlen, sdx), "r")
    lines = f.readlines()
    height = 32
    weight = 54
    channel = 3
    # 5D tensor : samples, frames, height, width, channel)
    DataTensor = np.empty((len(lines), maxlen, height, weight, channel))
    for index, line in enumerate(tqdm.tqdm(lines)):
        line = line.strip()
        line_dic = eval(line)
        transaction = line_dic["transaction"]
        start_point = (20, 0)
        end_point = (14, 0)
        past_pos = start_point

        tran_tensor = np.empty((maxlen, height, weight, channel))
        l_tensor = []  # np.empty((maxlen,1))
        for idx in range(len(transaction)):
            current_product = transaction[idx]
            current_pos = (int(transaction[idx]["row"]), int(
                transaction[idx]["column"]))
            if (current_pos == end_point) or (current_pos == start_point):
                path = [current_pos]
            else:
                #print("{0} -> {1}".format(past_pos, current_pos))
                path, point1, point2 = Opt.optimal_path(current_pos, past_pos)
                if len(path) == 0:
                    if point1 == point2:
                        path = [point1]
                past_pos = current_pos
            # maaping on pixel_world
            masking_map = mapping_on_gridworld(
                current_pos, path, map_arr, view_range)
            masking_map = np.expand_dims(masking_map, 0)

            tran_tensor[idx] = masking_map
            l_tensor.append(current_product)
        tran_tensor = np.expand_dims(tran_tensor, 0)
        DataTensor[index] = tran_tensor

        fw = open(DATA_STORAGE_PATH_TEMPORAL +
                  "/2021_MAP/Label_{}_{}_{}.csv".format(biz_type, maxlen, sdx), "a")
        for l_idx, l_ts in enumerate(l_tensor):
            if l_idx == 0:
                fw.write(str(l_ts))
            else:
                fw.write("\t"+str(l_ts))
        fw.write('\n')
        fw.close()

    DataTensor = DataTensor.astype(np.int16)
    np.save(DATA_STORAGE_PATH_TEMPORAL +
            "/2021_MAP/VFrame_{}_{}_{}.npy".format(biz_type, maxlen, sdx), DataTensor)

    #label = pd.DataFrame(LabelTensor)
    #label.to_csv(DATA_STORAGE_PATH_TEMPORAL + "/2021_MAP/Label_{}_{}_{}.csv".format(biz_type,maxlen, sdx), index = False)
    print("Done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--txG", type=str, default="n")
    parser.add_argument("--seqG", type=str, default="n")
    parser.add_argument("--mapG", type=str, default="n")
    parser.add_argument("--maxlen", type=int, default=20)
    parser.add_argument("--biztype", type=str, default="personal")
    parser.add_argument("--view", type=int, default=3)
    parser.add_argument("--sdx", type=str, default='train1')
    args = parser.parse_args()

    dirname = os.path.dirname(os.path.abspath((os.path.dirname(__file__))))

    DATA_SOURCE_PATH = dirname + '/Data/'
    DATA_STORAGE_PATH_TEMPORAL = dirname + '/Saved_file/'
    DATA_STORAGE_PATH = dirname + '/Data/PHASE0/'

    file_list = glob(DATA_SOURCE_PATH + 'TRXN/2021/*.csv')
    f_df = pd.read_csv(DATA_SOURCE_PATH + 'Product_info.csv')
    loc_df = pd.read_csv(
        DATA_SOURCE_PATH + "Store_POG_PixelWorld.csv", engine="python", encoding="euc-kr")
    map_arr = np.load(DATA_SOURCE_PATH+"mapping_array.npy")

    # 1. Generating purchase_trascation data (Data type : dic)
    if args.txG == "y":
        preprocess_purchase_transaction(
            file_list, f_df, loc_df, DATA_STORAGE_PATH_TEMPORAL)

    # 2. Generating sequence trasaction data (Data type : dic)

    if args.seqG == "y":
        preprocess_sequential_transaction(
            args.biztype, args.maxlen, args.sdx, loc_df, DATA_STORAGE_PATH_TEMPORAL)

    # 3. Generating video frame and label data
    if args.mapG == "y":
        preprocess_map_generation(
            args.biztype, args.maxlen, args.view, args.sdx, map_arr, loc_df, DATA_SOURCE_PATH)
