import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PHASE0_Data_preprocess.A_star import *
import numpy as np
import tqdm
class OptPath:
    def __init__(self, loc_data):
        self.loc_data = loc_data
        self.grid_map_size = (32,54)
        self.obs_array = self._generate_obstacle()

    def _generate_obstacle(self):
        obs_df = self.loc_data.copy()
        obs_df = obs_df[(obs_df.대분류 != "시작지점") & (obs_df.대분류 != "종료지점")]
        obs_df.reset_index(drop = True, inplace = True)
        obs_array = np.zeros(self.grid_map_size)
        for i in range(len(obs_df)):
            obs_array[obs_df.loc[i, "row"]][obs_df.loc[i, "column"]] = 1
        return obs_array

    def _check_forbid_location(self, pos):
        temp_loc = self.loc_data.copy()
        temp = temp_loc[(temp_loc.row == pos[0]) & (temp_loc.column == pos[1])]
        temp = temp[(temp.대분류 != "시작지점") & (temp.대분류 != "종료지점")]
        if len(temp) != 0:
            return True
        elif (0 > pos[0]) | (0 > pos[1]):
            return True
        elif (self.grid_map_size[0] <= pos[0]) | (self.grid_map_size[1] <= pos[1]):
            return True
        else:
            return False
    
    def map_adjust_position(self, point):
        if self._check_forbid_location((int(point[0]),int(point[1]))) == False:
            target_pos = point
        else:
            adjacent_list = [[1,0],[-1,0],[0,-1],[0,1]] # adjecent direction : up, down, left, right
            adjacent_loc_list = (np.array(adjacent_list) + np.array([int(point[0]),int(point[1])])).tolist()
            forbid_list = [self._check_forbid_location(loc) for loc in adjacent_loc_list]
            idx_list = [idx for idx, value in enumerate(forbid_list) if value == False]
            #이동 가능한 위치 저장
            possible_loc_list = [adjacent_loc_list[idx] for idx in idx_list]
            target_pos = tuple(possible_loc_list[0])
        return target_pos
    
    def optimal_path(self, start, end):
        point1 = self.map_adjust_position(start)
        point2 = self.map_adjust_position(end)
        path = aStar(self.obs_array, point1, point2)
        return path, point1, point2

def mapping_on_gridworld(target_pos, path, map_arr, view_range):
    '''
    input variable 
    target_pos : customer's location at timestep t
    masking_map : 32 by 54 array for pixel world which is emulate retail store.
    '''
    masking_map = np.zeros((32,54,1), dtype = np.uint16)
    
    # current location
    masking_map[int(target_pos[0])][int(target_pos[1])] = 1000
    
    #path map
    for pos in path:
        masking_map[pos[0]][pos[1]] = 100

    #view map
    view_map = custo_lens(target_pos, view_range)
  
    map = np.dstack((masking_map,view_map,np.expand_dims(map_arr, -1)))

    return map

def custo_lens(target_pos, view_range):
    min_x, min_y = np.array(target_pos) - view_range
    max_x, max_y = np.array(target_pos) + view_range
    min_x, min_y = lens_limit(min_x, min_y)
    max_x, max_y = lens_limit(max_x, max_y)
    view_map = np.zeros((32,54,1), dtype = np.uint16)
    for x in range(min_x,max_x):
        for y in range(min_y, max_y):
            view_map[x][y] = 1
    return view_map

def lens_limit(x, y):
    if x < 0: x = 0
    elif x > 32: x = 31
    if y < 0: y = 0
    elif y > 54: y = 53
    return x, y

