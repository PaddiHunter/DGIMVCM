import random
import sys
sys.path.append('.')
import numpy as np
import torch.optim
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

from utils.datasets import *
from models.baseModels import *
from torch.nn.functional import normalize
import torch.nn as nn
from utils.loss import *
from utils.util import *
from config import *
import time
import torch.optim
from utils.util import *
from utils.datasets import *
import collections
import warnings
from utils.graph_adjacency import *
from config import get_config
from utils.std_utils import *
from models.DGIMVCM import *



def main():
    # prepare
    test_time = 5

    for flag in [0]:
        config = get_config(flag)
        
        if config is None:
            print(f"Warning: get_config({flag}) returned None, skipping this configuration.")
            continue
            
        config['print_num'] = 100

        # logger
        logger, plt_name = get_logger(config)
        logger.info('Dataset:' + str(config['dataset']))

        # Load data
        X_list, Y_list = load_data(config, train_dir=True)

        for ms in [0,0.1,0.3,0.5,0.7]:

            for k in range(20, 21):
                setup_seed(42)
                topk=config['topk']
                logger.info(f'K neighbors {topk}')
                config['missing_rate'] = ms
                logger.info(f'missing rate {ms}')
                for name, value in config.items():
                    if isinstance(value, dict):
                        logger.info(str(name))
                        for item_name, item_value in value.items():
                            logger.info(f'{item_name} :{item_value}')
                    else:
                        logger.info(f'{name} :{value}')

                fold_acc, fold_nmi, fold_ari = [], [], []
                init_data_seed = config['training']['data_seed']
                accumulated_metrics = collections.defaultdict(list)
                # mask the data
                mask = get_mask(X_list[0].shape[0], config['missing_rate'], config['view_num'])
                X_list_miss = []
                for i in range(config['view_num']):
                    X_list_miss.append(np.multiply(X_list[i], mask[:, i][:, np.newaxis]))
                # data and mask to device
                X_list_train = []
                for i in range(len(X_list_miss)):
                    X_list_train.append(torch.from_numpy(X_list_miss[i]).float().to(device))
                mask = torch.from_numpy(mask).long().to(device)

                for data_seed in range(init_data_seed, test_time + init_data_seed):
                    logger.info(
                        f'---------------------------------------------------------------------start train {data_seed}-----------------------------------------------------------------------------')
                    setup_seed(data_seed)

                    # build model
                    model = MyModel(config)
                    model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr = config['training']['lr'])
                    
                    y_pred = model.run_train(X_list=X_list_train,mask=mask,optimizer=optimizer,Y_list=Y_list,logger=logger,ms=ms)
                    
                    scores = evaluation(y_pred=y_pred, y_true=Y_list[0], accumulated_metrics=accumulated_metrics)
                    logger.info(f"Final scores: {str(scores)}")
                    acc,nmi,ari=accumulated_metrics['acc'][-1], accumulated_metrics['nmi'][-1], accumulated_metrics['ARI'][-1]
                    fold_acc.append(acc)
                    fold_nmi.append(nmi)
                    fold_ari.append(ari)
                    
                logger.info(
                    '------------------------------------------------Training over----------------------------------------------------')
                logger.info(f"All fold results - ACC: {fold_acc}")
                logger.info(f"All fold results - NMI: {fold_nmi}")
                logger.info(f"All fold results - ARI: {fold_ari}")
                acc, nmi, ari = cal_std(logger, fold_acc, fold_nmi, fold_ari)
                logger.info(f'Final results - acc: {acc}, nmi: {nmi}, ari: {ari}')
        logger.handlers.clear()


if __name__ == '__main__':
    main()