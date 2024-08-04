import numpy as np
import torch
from torch_geometric.data import DataLoader,Data
from torch.utils.data import DataLoader as Loader
import scipy.sparse as sp
from tqdm import tqdm
import lib.utils as utils
from torch.nn.utils.rnn import pad_sequence
import pdb


class ParseData(object):
    def __init__(self, dataset_path,args,suffix='_springs5',mode="interp"):
        self.dataset_path = dataset_path
        self.suffix = suffix
        self.mode = mode
        self.random_seed = args.random_seed
        self.args = args
        self.total_step = args.total_ode_step
        self.cutting_edge = args.cutting_edge
        self.num_pre = args.extrap_num

        self.max_loc = None
        self.min_loc = None
        self.max_vel = None
        self.min_vel = None

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

    def load_data(self,sample_percent,batch_size,data_type="train",cut_num=5000):
        self.batch_size = batch_size
        self.sample_percent = sample_percent

        if data_type == "train":
            # cut_num = 5000
            print("train_num: ",cut_num)
        else:
            # cut_num = 5000
            print("test_num: ",cut_num)

        # Loading Data
        loc = np.load(self.dataset_path + '/loc_' + data_type + self.suffix + '.npy', allow_pickle=True)[:cut_num]
        vel = np.load(self.dataset_path + '/vel_' + data_type + self.suffix + '.npy', allow_pickle=True)[:cut_num]
        edges = np.load(self.dataset_path + '/edges_' + data_type + self.suffix + '.npy', allow_pickle=True)[:cut_num]  # [500,5,5]
        times = np.load(self.dataset_path + '/times_' + data_type + self.suffix + '.npy', allow_pickle=True)[:cut_num]  # 【500，5]

        self.num_graph = loc.shape[0]
        self.num_atoms = loc.shape[1]
        self.feature = loc[0][0][0].shape[0] + vel[0][0][0].shape[0]
        print("number graph in   "+data_type+"   is %d" % self.num_graph)
        print("number atoms in   " + data_type + "   is %d" % self.num_atoms)

        if self.suffix == "_springs5" or self.suffix == "_charged5" or self.suffix == '_forced_spring5' or self.suffix == 'pendulum3' or self.suffix == '_springs_damped5':
            # Normalize features to [-1, 1], across test and train dataset
            if self.max_loc == None:
                loc, max_loc, min_loc = self.normalize_features(loc,
                                                                self.num_atoms)  # [num_sims,num_atoms, (timestamps,2)]
                vel, max_vel, min_vel = self.normalize_features(vel, self.num_atoms)
                self.max_loc = max_loc
                self.min_loc = min_loc
                self.max_vel = max_vel
                self.min_vel = min_vel
            else:
                loc = (loc - self.min_loc) * 2 / (self.max_loc - self.min_loc) - 1
                vel = (vel - self.min_vel) * 2 / (self.max_vel - self.min_vel) - 1

        else:
            self.timelength = 49

        # split data w.r.t interp and extrap, also normalize times
        if self.mode=="interp":
            loc_en,vel_en,times_en = self.interp_extrap(loc,vel,times,self.mode,data_type)
            loc_de = loc_en
            vel_de = vel_en
            times_de = times_en
        elif self.mode == "extrap":
            loc_en,vel_en,times_en,loc_de,vel_de,times_de = self.interp_extrap(loc,vel,times,self.mode,data_type)
            # pdb.set_trace()

        #Encoder dataloader
        if self.mode == "extrap":
            series_list_en = self.encoder_data(loc_en,vel_en,times_en)
        else:
            raise NotImplementedError
        encoder_data_loader = Loader(series_list_en, batch_size=self.batch_size * self.num_atoms, shuffle=False,
                                     collate_fn=lambda batch: self.collate_encoder_batch(batch))  # num_graph*num_ball [tt,vals,masks]

                # Graph Dataloader --USING NRI
        edges = np.reshape(edges, [-1, self.num_atoms ** 2])
        edges = np.array((edges + 1) / 2, dtype=np.int64)
        edges = torch.LongTensor(edges)
        # Exclude self edges
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((self.num_atoms, self.num_atoms)) - np.eye(self.num_atoms)),
            [self.num_atoms, self.num_atoms])

        edges = edges[:, off_diag_idx]
        graph_data_loader = Loader(edges, batch_size=self.batch_size)

        # Decoder Dataloader
        if self.mode == "extrap":
            series_list_de = self.decoder_data(loc_de,vel_de,times_de)
        else:
            raise NotImplementedError
        
        decoder_data_loader = Loader(series_list_de, batch_size=self.batch_size * self.num_atoms, shuffle=False,
                                     collate_fn=lambda batch: self.variable_time_collate_fn_activity(
                                         batch))  # num_graph*num_ball [tt,vals,masks]

        num_batch = len(decoder_data_loader)
        encoder_data_loader = utils.inf_generator(encoder_data_loader)
        graph_data_loader = utils.inf_generator(graph_data_loader)
        decoder_data_loader = utils.inf_generator(decoder_data_loader)

        return encoder_data_loader, decoder_data_loader, graph_data_loader, num_batch

    def interp_extrap(self,loc,vel,times,mode,data_type):
        loc_observed = np.ones_like(loc)
        vel_observed = np.ones_like(vel)
        times_observed = np.ones_like(times)
        if mode =="interp":
            if data_type== "test":
                # get ride of the extra nodes in testing data.
                for i in range(self.num_graph):
                    for j in range(self.num_atoms):
                        loc_observed[i][j] = loc[i][j][:-self.num_pre]
                        vel_observed[i][j] = vel[i][j][:-self.num_pre]
                        times_observed[i][j] = times[i][j][:-self.num_pre]

                return loc_observed,vel_observed,times_observed/self.total_step
            else:
                return loc,vel,times/self.total_step

        elif mode == "extrap":# split into 2 parts and normalize t seperately
            loc_observed = np.ones_like(loc)
            vel_observed = np.ones_like(vel)
            times_observed = np.ones_like(times)

            loc_extrap = np.ones_like(loc)
            vel_extrap = np.ones_like(vel)
            times_extrap = np.ones_like(times)

            if data_type == "test":
                for i in range(self.num_graph):
                    for j in range(self.num_atoms):
                        loc_observed[i][j] = loc[i][j][-self.num_pre-1]
                        vel_observed[i][j] = vel[i][j][-self.num_pre-1]
                        times_observed[i][j] = times[i][j][-self.num_pre-1]

                        loc_extrap[i][j] = loc[i][j][-self.num_pre:]
                        vel_extrap[i][j] = vel[i][j][-self.num_pre:]
                        times_extrap[i][j] = times[i][j][-self.num_pre:]
                times_observed = times_observed/self.total_step
                times_extrap = (times_extrap - self.total_step)/self.total_step
            else:
                for i in range(self.num_graph):
                    for j in range(self.num_atoms):
                        times_current = times[i][j]
                        times_current_mask = np.where(times_current<self.total_step//2,times_current,0)
                        num_observe_current = np.argmax(times_current_mask)+1

                        loc_observed[i][j] = loc[i][j][num_observe_current]
                        vel_observed[i][j] = vel[i][j][num_observe_current]
                        times_observed[i][j] = times[i][j][num_observe_current]

                        loc_extrap[i][j] = loc[i][j][num_observe_current:]
                        vel_extrap[i][j] = vel[i][j][num_observe_current:]
                        times_extrap[i][j] = times[i][j][num_observe_current:]

                times_observed = times_observed / self.total_step
                times_extrap = (times_extrap - self.total_step//2) / self.total_step

            return loc_observed,vel_observed,times_observed,loc_extrap,vel_extrap,times_extrap

    def split_data(self,loc,vel,times):
        loc_observed = np.ones_like(loc)
        vel_observed = np.ones_like(vel)
        times_observed = np.ones_like(times)

        # split encoder data
        loc_list = []
        vel_list = []
        times_list = []

        for i in range(self.num_graph):
            for j in range(self.num_atoms):
                loc_list.append(loc[i][j][1:])  # [2500] num_train * num_ball
                vel_list.append(vel[i][j][1:])
                times_list.append(times[i][j][1:])

        series_list = []
        odernn_list = []
        for i, loc_series in enumerate(loc_list):
            # for encoder data
            graph_index = i // self.num_atoms
            atom_index = i % self.num_atoms
            length = len(loc_series)
            preserved_idx = sorted(
                np.random.choice(np.arange(length), int(length * self.sample_percent), replace=False))
            loc_observed[graph_index][atom_index] = loc_series[preserved_idx]
            vel_observed[graph_index][atom_index] = vel_list[i][preserved_idx]
            times_observed[graph_index][atom_index] = times_list[i][preserved_idx]

            # for odernn encoder
            feature_observe = np.zeros((self.timelength, self.feature))  # [T,D]
            times_observe = -1 * np.ones(self.timelength)  # maximum #[T], padding -1
            mask_observe = np.zeros((self.timelength, self.feature))  # [T,D] 1 means observed

            times_observe[:len(times_list[i][preserved_idx])] = times_list[i][preserved_idx]
            feature_observe[:len(times_list[i][preserved_idx])] = np.concatenate(
                (loc_series[preserved_idx], vel_list[i][preserved_idx]), axis=1)
            mask_observe[:len(times_list[i][preserved_idx])] = 1

            tt_observe = torch.FloatTensor(times_observe)
            vals_observe = torch.FloatTensor(feature_observe)
            masks_observe = torch.FloatTensor(mask_observe)

            odernn_list.append((tt_observe, vals_observe, masks_observe))

            # for decoder data, padding and mask
            feature_predict = np.zeros((self.timelength, self.feature))  # [T,D]
            times_predict = -1 * np.ones(self.timelength)  # maximum #[T], padding = 0, if have initial, then padding -1
            mask_predict = np.zeros((self.timelength, self.feature))  # [T,D] 1 means observed

            times_predict[:len(times_list[i])] = times_list[i]
            feature_predict[:len(times_list[i])] = np.concatenate((loc_series, vel_list[i]), axis=1)
            mask_predict[:len(times_list[i])] = 1

            tt = torch.FloatTensor(times_predict)
            vals = torch.FloatTensor(feature_predict)
            masks = torch.FloatTensor(mask_predict)

            series_list.append((tt, vals, masks))

        return series_list, loc_observed, vel_observed, times_observed

    def decoder_data(self, loc, vel, times):
        # split decoder data
        loc_list = []
        vel_list = []
        times_list = []

        for i in range(self.num_graph):
            for j in range(self.num_atoms):
                loc_list.append(loc[i][j])  # [2500] num_train * num_ball
                vel_list.append(vel[i][j])
                times_list.append(times[i][j])

        series_list = []
        for i, loc_series in enumerate(loc_list):
            # for decoder data, padding and mask
            feature_predict = np.zeros((self.timelength, self.feature))  # [T,D]
            times_predict = -1 * np.ones(self.timelength)  # maximum #[T], padding = 0, if have initial, then padding -1
            mask_predict = np.zeros((self.timelength, self.feature))  # [T,D] 1 means observed

            times_predict[:len(times_list[i])] = times_list[i]
            feature_predict[:len(times_list[i])] = np.concatenate((loc_series, vel_list[i]), axis=1)
            mask_predict[:len(times_list[i])] = 1

            tt = torch.FloatTensor(times_predict)
            vals = torch.FloatTensor(feature_predict)
            masks = torch.FloatTensor(mask_predict)

            series_list.append((tt, vals, masks))

        return series_list

    def encoder_data(self, loc, vel, times):
        # split decoder data
        # feature_list = []
        # vel_list = []
        # times_list = []
        series_list = []
        for i in range(self.num_graph):
            for j in range(self.num_atoms):
                series_list.append((torch.FloatTensor(np.concatenate([loc[i][j], vel[i][j]])), times[i][j]))  # [2500] num_train * num_ball
                # vel_list.append(vel[i][j])
                # times_list.append(times[i][j])

        return series_list

    def collate_encoder_batch(self,batch):
        combined_vals = torch.zeros([len(batch),batch[0][0].shape[0]])
        combined_t = torch.zeros(len(batch))

        for b, (vals, t) in enumerate(batch):
            combined_vals[b] = vals
            combined_t[b] = t

        combined_t = combined_t.float()

        data_dict = {
            "data": combined_vals,
            "time_steps": combined_t - 1
            }
        return data_dict
    
    def variable_time_collate_fn_activity(self,batch):
        """
        Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
            - record_id is a patient id
            - tt is a 1-dimensional tensor containing T time values of observations.
            - vals is a (T, D) tensor containing observed values for D variables.
            - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise. Since in human dataset, it join the data of four tags (belt, chest, ankles) into a single time series
            - labels is a list of labels for the current patient, if labels are available. Otherwise None.
        Returns:
            combined_tt: The union of all time observations.
            combined_vals: (M, T, D) tensor containing the observed values.
            combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
        """
        D = self.feature
        combined_tt, inverse_indices = torch.unique(torch.cat([ex[0] for ex in batch]), sorted=True,
                                                    return_inverse=True) #【including 0 ]
        offset = 0
        combined_vals = torch.zeros([len(batch), len(combined_tt), D])
        combined_mask = torch.zeros([len(batch), len(combined_tt), D])
        combined_frt = torch.zeros(len(batch))

        for b, ( tt, vals, mask) in enumerate(batch):
            indices = inverse_indices[offset:offset + len(tt)]

            offset += len(tt)

            combined_vals[b, indices] = vals
            combined_mask[b, indices] = mask
            combined_frt[b] = tt[0]

        # get rid of the padding timepoint
        combined_tt = combined_tt[1:]
        combined_vals = combined_vals[:,1:,:]
        combined_mask = combined_mask[:,1:,:]

        combined_tt = combined_tt.float()

        data_dict = {
            "data": combined_vals,
            "time_steps": combined_tt,
            "mask": combined_mask,
            "time_first": combined_frt,
            }
        return data_dict

    def normalize_features(self,inputs, num_balls):
        '''
        :param inputs: [num-train, num-ball,(timestamps,2)]
        :return:
        '''
        value_list_length = [balls[i].shape[0] for i in range(num_balls) for balls in inputs]  # [2500] num_train * num_ball
        self.timelength = max(value_list_length)
        value_list = [torch.tensor(balls[i]) for i in range(num_balls) for balls in inputs]
        value_padding = pad_sequence(value_list,batch_first=True,padding_value = 0)
        max_value = torch.max(value_padding).item()
        min_value = torch.min(value_padding).item()

        # Normalize to [-1, 1]
        inputs = (inputs - min_value) * 2 / (max_value - min_value) - 1
        return inputs,max_value,min_value

    def convert_sparse(self,graph):
        graph_sparse = sp.coo_matrix(graph)
        edge_index = np.vstack((graph_sparse.row, graph_sparse.col))
        edge_attr = graph_sparse.data
        return edge_index, edge_attr
