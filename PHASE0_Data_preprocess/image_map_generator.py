import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from glob import glob
import pandas as pd
import numpy as np
if __name__ == "__main__":
    df = pd.read_csv("../Data/다담_상품배치_좌표포함_보정본.csv", engine="python", encoding= "euc-kr")
    map_size = np.zeros((32,54), dtype=np.int16)
    aisle_map = np.zeros((32,54),dtype=np.int16)
    unique = []
    for i in range(len(df)):
        unique.append(tuple((df.loc[i,"통로구분"],df.loc[i,"중분류코드"])))
    unique = list(set(unique))

    for i in range(len(df)):
        aisle_idx = unique.index(tuple((df.loc[i,"통로구분"], df.loc[i,"중분류코드"])))
        #aisle_idx = np.power(df.loc[i,"통로구분"] + 1, 1.56).astype(int)
        map_size[df.loc[i,"row"], df.loc[i,"column"]] = (aisle_idx + 1) * 100
    
    np.save("../Data/mapping_array.npy", map_size)

