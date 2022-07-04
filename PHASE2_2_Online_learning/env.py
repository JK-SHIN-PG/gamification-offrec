import torch
import numpy as np
from utils import *


class Environment:
    def __init__(self, loc_data, map_arr, action_list, planned_product_list, view_range, epsilon):
        self.current_pos = (20, 0)
        self.start_point = np.array((20, 0))
        self.end_point = np.array((14, 0))
        self.loc_data = loc_data
        self.map_arr = map_arr
        self.grid_map_size = (32, 54)
        self.action_list = action_list
        #self.action_one_hot = [0]*len(self.action_list)
        self.current_basket = []
        self.labeled_current_basket_rec = (
            torch.ones((1, 20)) * int(3)).type(torch.LongTensor)
        self.pointer_lcbr = 0
        self.labeled_current_basket_user = (
            torch.ones((1, 18)) * int(3)).type(torch.LongTensor)
        self.pointer_lcbu = 0
        self.current_object = []
        self.planned_product_list = planned_product_list.copy()
        self.init_product_list = planned_product_list.copy()
        self.unplanned_product_list = []
        self.seq_p_item_set = None
        self.view_range = view_range
        self.past_path = []
        self.whole_past_path = []
        self.obs_array = self.generate_obstacle()
        self.current_path = self.generate_optimal_path()
        self.hidden = None
        self.terminal = False
        #self.add_to_current_basket_rec("{'상품코드': '<bos>', '상품명': '<bos>', 'row': '20', 'column': '0', 'price': '0'}")
        self.epsilon = epsilon

    def generate_obstacle(self):
        obs_df = self.loc_data.copy()
        obs_df = obs_df[(obs_df.대분류 != "시작지점") & (obs_df.대분류 != "종료지점")]
        obs_df.reset_index(drop=True, inplace=True)
        obs_array = np.zeros(self.grid_map_size)
        for i in range(len(obs_df)):
            obs_array[obs_df.loc[i, "row"]][obs_df.loc[i, "column"]] = 1
        return obs_array

    # 0. planned item set을 기반으로 상품 구매 순서 생성 및 바로 다음 상품으로 향하기 위한 동선 생성
    def generate_optimal_path(self):
        if len(self.planned_product_list) == 0:
            seq_p_item_set = TSP_solver(self.current_pos, [
                                        "{'상품코드': '<eos>', '상품명': '<eos>', 'row': '14', 'column': '0', 'price': '0'}"], self.loc_data)
        else:
            seq_p_item_set = TSP_solver(self.current_pos, self.planned_product_list + [
                                        "{'상품코드': '<eos>', '상품명': '<eos>', 'row': '14', 'column': '0', 'price': '0'}"], self.loc_data)
        self.seq_p_item_set = seq_p_item_set
        Opt = OptPath(self.loc_data)
        self.current_object = seq_p_item_set[0]
        trans = eval(seq_p_item_set[0])
        path, point1, point2 = Opt.optimal_path(
            self.current_pos, (int(trans["row"]), int(trans["column"])))
        return path

    # 1-1 given current location -> generate image -> pass through CNN
    def generate_map(self):
        masking_map = np.zeros((32, 54, 1), dtype=np.float32)
        # set weight as 1000 for current location of user model
        masking_map[int(self.current_pos[0])][int(self.current_pos[1])] = 1000
        for pos in self.past_path:
            # set weight as 100 for past path of user model
            masking_map[pos[0]][pos[1]] = 100
        view_map = custo_lens(
            self.current_pos, self.view_range)  # create viwe map
        map = np.dstack(
            (masking_map, view_map, np.expand_dims(self.map_arr, -1)))
        map = torch.from_numpy(map)
        map = map.permute(2, 0, 1).unsqueeze(0)
        return map

    def add_to_current_basket_rec(self, rec_inf):  # not use
        self.labeled_current_basket_rec[0][self.pointer_lcbr] = self.action_list.index(
            rec_inf)
        self.pointer_lcbr += 1

    def add_to_current_basket_user(self):
        # if basket is full, basket is updated in way of queue (first in first out)
        if self.pointer_lcbu == 18:
            temp = self.labeled_current_basket_user[0].clone().detach()
            for i in range(17):
                temp[i] = self.labeled_current_basket_user[0][i+1]
            temp[17] = self.action_list.index(self.current_object)
            self.labeled_current_basket_user[0] = temp
        else:
            self.labeled_current_basket_user[0][self.pointer_lcbu] = self.action_list.index(
                self.current_object)
            self.pointer_lcbu += 1

    # 2. given contexual vector, get recommender item by passing through GRU

    def get_recommended_item(self, Encoder, Decoder, device):
        map = self.generate_map()
        embedding_vector = Encoder(map.to(device))
        prediction, hidden = Decoder(
            embedding_vector.unsqueeze(1), self.hidden)  # hidden
        label_prediction = torch.argmax(prediction)

        return prediction, label_prediction, hidden

    # 3. Give recommender item to user model and listen user's decision (accept or reject)
    def decide_acceptance(self, rec_product, User_model, device):
        input = torch.cat((rec_product.unsqueeze(0).unsqueeze(
            0), self.labeled_current_basket_user.to(device)), dim=1)
        output = User_model(input)
        output = torch.round(output.squeeze(0)).item()
        if output == 1:
            return True
        else:
            if np.random.random(1) < self.epsilon:
                return True
            else:
                return False

    # 4 (1). if accept, add to unplanned item set and generate path and seqence in item list (planned item set + unplanned item set)
    # 4 (2). if reject, go to next step (상품순서, 다음 상품으로 향하기 위한 동선을 그대로 상속받아 다음 스텝으로 넘어감)
    def get_response(self, user_decision, rec_product, hidden):
        if user_decision == True:
            rec_inf = self.action_list[rec_product]
            if rec_inf not in self.unplanned_product_list:
                self.unplanned_product_list.append(rec_inf)
                self.planned_product_list.append(rec_inf)
                # self.add_to_current_basket_rec(rec_inf)
                self.hidden = hidden
                return True
            return False
        else:
            return False

    # 5. Move the user model one step to get the next item
    def move_customer(self, response_result):
        if response_result == True:
            self.current_path = self.generate_optimal_path()
            self.whole_past_path = self.whole_past_path + self.past_path
            self.past_path = []

        if len(self.current_path) == 0:
            if eval(self.current_object)["상품명"] == "<eos>":
                if (self.current_pos[0] == 14) and (self.current_pos[1] == 0):
                    self.terminal = True
                    self.whole_past_path = self.whole_past_path + self.past_path
                    # print("terminal!")
            else:
                self.current_basket.append(self.current_object)
                # self.add_to_current_basket_rec()
                self.add_to_current_basket_user()
                self.planned_product_list.remove(self.current_object)
                self.current_path = self.generate_optimal_path()
                self.whole_past_path = self.whole_past_path + self.past_path
                self.past_path = []
        else:
            self.current_pos = self.current_path.pop()
            self.past_path.append(self.current_pos)
