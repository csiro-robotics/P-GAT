import json
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset

from attentional_graph.utils.comm import get_world_size

class normalization:
    def __init__(self, params=None) -> None:
        if params is not None:
            self.mean = params[0]
            self.std = params[1]

    def fit(self, data, masks=None):
        '''
        Calculate the data's mean and std

        Input:
            data: array
                Data need to normalize
        '''
        data_dim = data.shape[-1]
        data = data.reshape(-1, data_dim)
        if masks is not None:
            masks = torch.flatten(masks)
            data = data[~masks]
        self.mean = torch.mean(data)
        self.std = torch.std(data)

    def transform(self, data, masks=None):
        data = (data - self.mean) / self.std
        if masks is not None:
            masks = torch.unsqueeze(masks, dim=1)
            masks = torch.swapaxes(masks, 1, 2)
            data = data * (~masks).int()
        return data

def data_normalize(data, need_fit, sample_index=None, cfg=None):
    normalizer = normalization()
    if need_fit:
        normalization_info = {}
        features = data['features'][sample_index[0], :, :]
        poses = data['poses'][sample_index[0], :, :]
        masks = data['masks'][sample_index[0], :]

        normalizer.fit(features, masks=masks)
        print('features data mean: %f and std: %f'%(
            normalizer.mean, 
            normalizer.std
        ))
        normalization_info['features'] = {
            'mean': normalizer.mean.item(), 
            'std': normalizer.std.item()
        }
        data['features'] = normalizer.transform(
            data['features'], 
            masks=data['masks'],
        )


        normalizer.fit(poses, masks=masks)
        print('poses data mean: %f and std: %f'%(
            normalizer.mean, 
            normalizer.std
        ))
        normalization_info['poses'] = {
            'mean': normalizer.mean.item(), 
            'std': normalizer.std.item()
        }
        data['poses'] = normalizer.transform(
            data['poses'], 
            masks=data['masks'],
        )
        # save the normalization info
        with open(cfg.OUTPUT_DIR + cfg.TRAIN.NORMAL_PARAM, 'w') as normal_param:
            json.dump(normalization_info, normal_param)
    else:
        # load fitted normalization parameters
        with open(cfg.OUTPUT_DIR + cfg.TEST.NORMAL_PARAM, 'r') as normal_param:
            normalization_param = json.load(normal_param)

        # normalize the features
        normalizer.mean = normalization_param['features']['mean']
        normalizer.std = normalization_param['features']['std']
        print('features data mean: %f and std: %f'%(
            normalizer.mean, 
            normalizer.std
        ))
        for idx in range(len(data['features'])): 
            data['features'][idx] = normalizer.transform(
                data['features'][idx],
                masks=data['masks'][idx],
            )

        # normalize the poses
        normalizer.mean = normalization_param['poses']['mean']
        normalizer.std = normalization_param['poses']['std']
        print('poses data mean: %f and std: %f'%(
            normalizer.mean, 
            normalizer.std
        ))
        for idx in range(len(data['poses'])):
            data['poses'][idx] = normalizer.transform(
                data['poses'][idx],
                masks=data['masks'][idx],
            )
    return data

def load_from_file(file_path, stride, is_train=False):
    original_data = torch.load(file_path)
    if is_train:
        data_with_stride = original_data
    else:
        data_with_stride = original_data[::stride]
    return data_with_stride

def build_dataset(cfg, dataset_id, is_train=True):
    '''
    Arguments:
        cfg: config file
        dataset_id: int
            indicate which dataset is loading
        is_train (boolean): whether to setup the dataset for training or testing
    '''
    if is_train:
        feature_list = cfg.DATASETS.FEATURE_TRAIN
        pose_list = cfg.DATASETS.POSE_TRAIN
        run_id_list = cfg.DATASETS.RUN_ID_TRAIN
        mask_list = cfg.DATASETS.MASK_TRAIN
        nodes_info_list = cfg.DATASETS.NODEINFO_TRAIN
        score_file = cfg.DATASETS.SCORE
        switch_file = cfg.DATASETS.SWITCH
        paired = cfg.DATASETS.PAIR_TRAIN
    else:
        feature_list = cfg.DATASETS.FEATURE_TEST
        pose_list = cfg.DATASETS.POSE_TEST
        run_id_list = cfg.DATASETS.RUN_ID_TEST
        mask_list = cfg.DATASETS.MASK_TEST
        nodes_info_list = cfg.DATASETS.NODEINFO_TEST
        score_file = cfg.DATASETS.SCORE


    if not isinstance(feature_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(feature_list))
    if not isinstance(pose_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(pose_list))
    if not isinstance(run_id_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(run_id_list))
    if not isinstance(mask_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(mask_list))

    feature_length = len(feature_list)
    pose_length = len(pose_list)
    mask_length = len(mask_list)
    if feature_length != pose_length:
        raise RuntimeError(
            "The length of feature and position must be same,"
            " got different length!"
        )
    if feature_length != mask_length:
        raise RuntimeError(
            "The length of feature and mask must be same,"
            " got different length!"
        )

    feature = []
    pose = []
    run_id = []
    mask = []
    nodes_info = []
    for index in range(feature_length):
        feature.append(load_from_file(
            cfg.DATA_DIR[dataset_id] + feature_list[index],
            cfg.TEST.STRIDE,
            is_train=is_train,
        ))
        pose.append(load_from_file(
            cfg.DATA_DIR[dataset_id] + pose_list[index],
            cfg.TEST.STRIDE,
            is_train=is_train,
        ))
        run_id.append(load_from_file(
            cfg.DATA_DIR[dataset_id] + run_id_list[index],
            cfg.TEST.STRIDE,
            is_train=is_train,
        ))
        mask.append(load_from_file(
            cfg.DATA_DIR[dataset_id] + mask_list[index],
            cfg.TEST.STRIDE,
            is_train=is_train,
        ))
        nodes_info.append(load_from_file(
            cfg.DATA_DIR[dataset_id] + nodes_info_list[index],
            cfg.TEST.STRIDE,
            is_train=is_train,
        ))
    score =  torch.load(cfg.DATA_DIR[dataset_id] + score_file)
    if is_train:
        with open(cfg.DATA_DIR[dataset_id] + paired, 'r') as pair_json:
            paired_info = json.load(pair_json)
        switch =  torch.load(cfg.DATA_DIR[dataset_id] + switch_file)

        return {
            'features': torch.cat(feature), 
            'poses': torch.cat(pose), 
            'scores': score,
            'masks': torch.cat(mask),
            'nodes_info': torch.cat(nodes_info),
            'switches': switch,
            'paired_info': paired_info,
            'run_id': torch.cat(run_id),
        }
    else:
        # Don't concatenate because the first element is training data
        # and the second element is the testing data
        return {
            'features': feature, 
            'poses': pose, 
            'scores': score,
            'masks': mask,
            'nodes_info': nodes_info,
            'run_id': run_id,
        }