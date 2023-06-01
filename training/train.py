import json
import os
from datetime import datetime
import sys
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import tqdm

from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from attentional_graph.config_init import cfg
from attentional_graph.engine.trainer import do_train
from attentional_graph.modeling.pose_gat import PoseGAT
from attentional_graph.utils.comm import get_rank
from attentional_graph.utils.logger import setup_logger
from attentional_graph.utils.miscellaneous import mkdir, save_config
from datasets import build_dataset, data_normalize

# TODO: This is just a place holder to add more params to stats
def print_stats(stats, phase):
    if 'num_pairs' in stats:
        # For batch hard contrastive loss
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   Pairs per batch (all/non-zero pos/non-zero neg): {:.1f}/{:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_pairs'],
                       stats['pos_pairs_above_threshold'], stats['neg_pairs_above_threshold']))
    elif 'num_triplets' in stats:
        # For triplet loss
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   Triplets per batch (all/non-zero): {:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_triplets'],
                       stats['num_non_zero_triplets']))
    elif 'num_pos' in stats:
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   #positives/negatives: {:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['avg_embedding_norm'], stats['num_pos'], stats['num_neg']))

    s = ''
    l = []
    if 'mean_pos_pair_dist' in stats:
        s += 'Pos dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}   Neg dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}'
        l += [stats['min_pos_pair_dist'], stats['mean_pos_pair_dist'], stats['max_pos_pair_dist'],
              stats['min_neg_pair_dist'], stats['mean_neg_pair_dist'], stats['max_neg_pair_dist']]
    if 'pos_loss' in stats:
        if len(s) > 0:
            s += '   '
        s += 'Pos loss: {:.4f}  Neg loss: {:.4f}'
        l += [stats['pos_loss'], stats['neg_loss']]
    if len(l) > 0:
        print(s.format(*l))

def create_index(splitted_run_index, run_per_subgraph, paired_info):
    '''
    generate the training and validation subgraphs' index
    update the paired_info
    '''
    training_runs = set(splitted_run_index[0])
    validation_runs = set(splitted_run_index[1])
    training_index = []
    validation_index = []
    for idx in range(len(run_per_subgraph)):
        if run_per_subgraph.tolist()[idx] in training_runs:
            training_index.append(idx)
            paired_info[str(idx)] = update_pairing_list(paired_info[str(idx)], training_runs)
        elif run_per_subgraph.tolist()[idx] in validation_runs:
            validation_index.append(idx)
            paired_info[str(idx)] = update_pairing_list(paired_info[str(idx)], validation_runs)
        #else:
        #    print('Run %i is not in the lists'%run_per_subgraph[idx])
    return [training_index, validation_index], paired_info

def update_pairing_list(paired, run_idx):
    updated_paired_info = []
    for idx in run_idx:
        updated_paired_info += paired[idx]
    return updated_paired_info

def flatten_list(arr):
    flattened = []
    for i in range(len(arr)):
        flattened += arr[i]
    return flattened

def index_split(dataset_input, use_all=False):
    '''
    - Split the index of subgraphs into training and validation
    - Update the paired_info of each subgraph according which dataset (training or validation)
        this subgraph from
    '''
    run_per_subgraph = dataset_input['run_id']
    total_runs_num = torch.max(run_per_subgraph)
    print('The dataset has %i runs'%total_runs_num)
    if use_all:
        splitted_index, dataset_input['paired_info'] = create_index(
            [[run_idx for run_idx in range(total_runs_num + 1)], []],
            run_per_subgraph,
            dataset_input['paired_info'],
        )
    else:
        splitted_run_index = train_test_split(range(total_runs_num + 1), test_size=0.2)
        splitted_index, dataset_input['paired_info'] = create_index(
            splitted_run_index, 
            run_per_subgraph, 
            dataset_input['paired_info'],
        )
    return splitted_index, dataset_input

def main():
    # ---------------------------------------------------------------------------- #
    # model configuration
    # ---------------------------------------------------------------------------- #
    parser = argparse.ArgumentParser(
        description='Train graph net embeddings using SuperGlue'
    )
    parser.add_argument(
        "--config_file", 
        default="configs/training_config.yml", 
        metavar="FILE", 
        help="path to config file", 
        type=str
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    # Check if gpu acceleration is available
    print("is gpu available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("attentional graph", output_dir, get_rank())
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    # ---------------------------------------------------------------------------- #
    # Data loading
    # ---------------------------------------------------------------------------- #
    datasets = {}
    num_datasets = len(cfg.DATA_DIR)
    num_nodes = 0
    subgraph_index = []
    subgraph_dataset_id = []
    normalized_data = {}
    for dataset_id in range(num_datasets):
        dataset_input = build_dataset(
            cfg,
            dataset_id=dataset_id,
            is_train=True,
        )
        splitted_index, dataset_input = index_split(dataset_input, use_all=True)
        for key in dataset_input:
            if key not in normalized_data:
                normalized_data[key] = [dataset_input[key]]
            else:
                normalized_data[key].append(dataset_input[key])
        num_nodes = max(num_nodes, dataset_input['masks'].shape[-1])
        subgraph_index.append(splitted_index[0])
        subgraph_dataset_id.append([dataset_id for _ in range(len(splitted_index[0]))])
    normalized_data['dataset_subgraph_info'] = subgraph_index

    train_index_loader = DataLoader(
        TensorDataset(
            torch.tensor(flatten_list(subgraph_index)),
            torch.tensor(flatten_list(subgraph_dataset_id)),
        ), 
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_size=cfg.SOLVER.BATCHSIZE,
        shuffle=True,
    )
    datasets['train'] = train_index_loader

    # ---------------------------------------------------------------------------- #
    # Model definition
    # ---------------------------------------------------------------------------- #
    super_glue_model = PoseGAT(
        pose_dim=cfg.MODEL.ATTENTION_GRAPH.POSE_DIM, 
        feature_dim=cfg.MODEL.ATTENTION_GRAPH.FEATURE_DIM, 
        keypoint_enc_hidden_dim=cfg.MODEL.ATTENTION_GRAPH.KEYPOINT_HIDDEN_DIM,
        include_pose=cfg.MODEL.ATTENTION_GRAPH.INCLUDE_POSE,
        num_heads=cfg.MODEL.ATTENTION_GRAPH.NUM_HEADS,
        num_layers=cfg.MODEL.ATTENTION_GRAPH.NUM_LAYERS,
        dropout=cfg.MODEL.ATTENTION_GRAPH.DROPOUT,
    ).to(device)
    # loss function and optimizer
    criterion = torch.nn.BCELoss(reduction='none')
    optimizer = torch.optim.Adam(
        super_glue_model.parameters(), 
        lr=cfg.SOLVER.LR,
    )
    if cfg.TRAIN.MODEL_PARAM is not None:
        model_param = cfg.OUTPUT_DIR + cfg.TRAIN.MODEL_PARAM
        super_glue_model.load_state_dict(torch.load(model_param))

    # ---------------------------------------------------------------------------- #
    # Model training
    # ---------------------------------------------------------------------------- #  
    # Initialize TensorBoard writer
    now = datetime.now()
    logdir = os.path.join("tf_logs", now.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(logdir)

    if cfg.TRAIN.HAS_VALIDATION:
        phases = ['train', 'val']
        stats = {'train': [], 'val': []}
        with open(cfg.OUTPUT_DIR + "loss_curve.txt", "w") as loss_file:
            loss_file.write("epoch, train_loss, val_loss\n")
    else:
        phases = ['train']
        stats = {'train': []}
        with open(cfg.OUTPUT_DIR + "loss_curve.txt", "w") as loss_file:
            loss_file.write("epoch, train_loss\n")

    for epoch in tqdm.tqdm(range (1, cfg.SOLVER.EPOCH+1)):

        running_stats = [] # running stats for the current epoch
        for phase in phases:
            running_stats = do_train(
                cfg, 
                super_glue_model, 
                normalized_data, 
                train_index_loader,
                num_nodes,
                optimizer, 
                criterion, 
                device,
            )

            # Compute mean stats for the epoch
            epoch_stats = {}
            # running_stats = running_stats_
            for key in running_stats[0].keys():
                temp = [e[key] for e in running_stats]
                epoch_stats[key] = np.mean(temp)

            stats[phase].append(epoch_stats)
            print_stats(epoch_stats, phase)

        # ******* EPOCH END *******   

        if cfg.TRAIN.HAS_VALIDATION:
            with open(cfg.OUTPUT_DIR + "loss_curve.txt", "a") as loss_file:
                loss_file.write("%i, %0.6f, %0.6f\n"%(
                    epoch,
                    stats['train'][-1]['loss'],
                    stats['val'][-1]['loss'],
                ))
        else:
            with open(cfg.OUTPUT_DIR + "loss_curve.txt", "a") as loss_file:
                loss_file.write("%i, %0.6f\n"%(
                    epoch,
                    stats['train'][-1]['loss'],
                ))

        loss_metrics = {'train': stats['train'][-1]['loss']}
        if 'train' in phases:
            loss_metrics['train'] = stats['train'][-1]['loss']
        writer.add_scalars('Loss', loss_metrics, epoch)
        if 'val' in phases:
            loss_metrics['val'] = stats['val'][-1]['loss']
        writer.add_scalars('Loss', loss_metrics, epoch)

        if epoch % cfg.TRAIN.SAVE_INTERVAL == 0:
            torch.save(
                super_glue_model.state_dict(), 
                cfg.OUTPUT_DIR + "attentional_graph" + str(epoch) + ".pt"
            )
        # TODO: we need to move toward dynamic batch size to make the training speed more efficient
        # if 'num_triplets' in stats['train'][-1]:
        #     nz_metrics = {'train': stats['train'][-1]['num_non_zero_triplets']}
        #     if 'val' in phases:
        #         nz_metrics['val'] = stats['val'][-1]['num_non_zero_triplets']
        #     writer.add_scalars('Non-zero triplets', nz_metrics, epoch)

        # elif 'num_pairs' in stats['train'][-1]:
        #     nz_metrics = {'train_pos': stats['train'][-1]['pos_pairs_above_threshold'],
        #                   'train_neg': stats['train'][-1]['neg_pairs_above_threshold']}
        #     if 'val' in phases:
        #         nz_metrics['val_pos'] = stats['val'][-1]['pos_pairs_above_threshold']
        #         nz_metrics['val_neg'] = stats['val'][-1]['neg_pairs_above_threshold']
        #     writer.add_scalars('Non-zero pairs', nz_metrics, epoch)

    if cfg.SAVE_MODEL:
        torch.save(
            super_glue_model.state_dict(), 
            cfg.OUTPUT_DIR + "attentional_graph.pt"
        )

if __name__ == '__main__':
    main()
