from matplotlib.lines import lineStyles
from matplotlib import pyplot as plt
import os
import json
import argparse
import numpy as np
from yacs.config import CfgNode as CN

def default_config():
    '''
    The one-stop reference point for all configurable options

    **overrides the options should use YAML configuration files**
    '''
    cfg = CN()
    # The path to the repeated test folders
    cfg.TEST_PATH = './'
    # Folder name including the path from the cfg.TEST_PATH/repeat* 
    # to the folder which has the evaluation.json
    cfg.TEST_FOLDER = "oxford_test/"
    # The AR@K json file
    cfg.FILE_NAME = "evaluation_result.json"
    # The output plot filename
    cfg.PLOT_NAME = "ar@k.png"
    return cfg

def plot_recall(cfg):
    fig1, ax1 = plt.subplots(dpi=150)
    plt.style.use('ggplot')
    x = [i+1 for i in range(25)]
    for it in os.scandir(cfg.TEST_PATH):
        if it.is_dir():
            trial_name = os.path.split(it.path)[1]
            json_path = it.path + "/" + cfg.TEST_FOLDER + cfg.FILE_NAME
            f = open(json_path)
            recalls = json.load(f)
            recalls = recalls["recall@k"]
            ax1.plot(x,
                recalls,
                'o-', label = trial_name)

    ax1.set_ylabel('AR@K')
    ax1.set_xlabel('N - number of top candidates')
    fig1.legend(loc='lower right')
    print(cfg.TEST_PATH + '/' + cfg.PLOT_NAME)
    fig1.savefig(cfg.TEST_PATH + '/' + cfg.PLOT_NAME)

def plot_recall_stats(cfg):
    fig2, ax2 = plt.subplots(dpi=150)
    plt.style.use('ggplot')
    x = [i+1 for i in range(25)]
    recalls_total = []
    for it in os.scandir(cfg.TEST_PATH):
        if it.is_dir():
            json_path = it.path + "/" + cfg.TEST_FOLDER + cfg.FILE_NAME
            f = open(json_path)
            recalls = json.load(f)
            recalls = recalls["recall@k"]
            recalls_total.append(recalls)
    np.concatenate(recalls_total)

    # Compute mean and std
    mean = np.mean(recalls_total, axis=0)
    std = np.std(recalls_total, axis=0)

    # ax2.plot(x,
    #     mean,
    #     'o-', label = "P-GAT")
    ax2.errorbar(x, mean*100, std*100, fmt='-bo', ecolor= "blue", capsize=3, label = "P-GAT")
    ax2.set_ylabel('mAR@K')
    ax2.set_xlabel('N - number of top candidates')
    fig2.legend(loc='lower right')
    print(cfg.TEST_PATH + '/stat_' + cfg.PLOT_NAME)
    fig2.savefig(cfg.TEST_PATH + '/stat_' + cfg.PLOT_NAME)



def main():

    parser = argparse.ArgumentParser(
        description= 'Plot recal curves'
    )
    parser.add_argument(
        "--config_file",
        default = "configs/plot_config.yml",
        metavar= "FILE",
        help="path to config file",
        type=str

    )
    args = parser.parse_args()

    cfg = default_config()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    plot_recall(cfg)
    plot_recall_stats(cfg)
    print("done!")


    
if __name__ == "__main__":
    main()