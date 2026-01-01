import os
import warnings
warnings.filterwarnings("ignore")
import os
import torch
import methods
import argparse
import pandas as pd
from conf import cfg
from core.model import build_model
from core.data import build_loader
from core.utils import seed_everything, save_df, set_logger
from tqdm import tqdm
from setproctitle import setproctitle
from loguru import logger as log

def eval(cfg):
    model = build_model(cfg).cuda()
    tta_model = getattr(methods, cfg.ADAPTER.NAME).setup(model, cfg)
    tta_model.cuda()
    loader, processor = build_loader(cfg)
    new_results = {
        "method": cfg.ADAPTER.NAME,
        'dataset':cfg.CORRUPTION.DATASET,
        'model': cfg.MODEL.ARCH,
        'batch_size':cfg.TEST.BATCH_SIZE,
        'seed': cfg.SEED,
        'note': cfg.NOTE,
        'order': cfg.CORRUPTION.ORDER_NUM,
        'Avg': 0
    }
    log.info(new_results)
    rounds = 15
    for _round in range(rounds):
        new_results[f'round_{_round}'] = 0
        all_y_test = torch.tensor([]).cuda()
        for batch_id, data_package in enumerate(loader):
            x_test, y_test, domain = data_package["image"], data_package['label'], data_package['domain']
            if len(y_test) == 1:
                continue
            x_test, y_test = x_test.cuda(), y_test.cuda()
            all_y_test = torch.cat((all_y_test, y_test))
            output = tta_model(x_test)
            predict = torch.argmax(output, dim=1)
            accurate = (predict == y_test)
            processor.process(accurate, domain)

        processor.calculate()
        _results = processor.info()

        acc = _results['Avg']
        new_results[f'round_{_round}'] += acc
        _error = 100 - acc
        log.info(f"[round_{_round}]: Acc {acc:.2f} || Error {_error:.2f}")

    new_results['Avg'] = sum([new_results[f'round_{_round}'] for _round in range(rounds)]) / rounds
    log.info(f"[Avg ]: Acc {new_results['Avg']:.2f} || Error {100-new_results['Avg']:.2f}")
    save_df(new_results,f'./results/PTTA_round_{cfg.CORRUPTION.DATASET}_{cfg.CORRUPTION.ORDER_NUM}.csv')

    
def main():
    parser = argparse.ArgumentParser("Pytorch Implementation for Continual Test Time Adaptation!")
    parser.add_argument(
        '-acfg',
        '--adapter-config-file',
        metavar="FILE",
        default="",
        help="path to adapter config file",
        type=str)
    parser.add_argument(
        '-dcfg',
        '--dataset-config-file',
        metavar="FILE",
        default="",
        help="path to dataset config file",
        type=str)
    parser.add_argument(
        '-ocfg',
        '--order-config-file',
        metavar="FILE",
        default="",
        help="path to order config file",
        type=str)

    parser.add_argument(
        'opts',
        help='modify the configuration by command line',
        nargs=argparse.REMAINDER,
        default=None)

    args = parser.parse_args()
    if len(args.opts) > 0:
        args.opts[-1] = args.opts[-1].strip('\r\n')

    cfg.merge_from_file(args.adapter_config_file)
    cfg.merge_from_file(args.dataset_config_file)
    if args.order_config_file != "":
        cfg.merge_from_file(args.order_config_file)

    cfg.merge_from_list(args.opts)
    cfg.freeze()

    seed_everything(cfg.SEED)
    set_logger(cfg.LOG_DIR,cfg.ADAPTER.NAME)
    current_file_name = os.path.basename(__file__)
    setproctitle(f"{current_file_name}:{cfg.CORRUPTION.DATASET}:{cfg.ADAPTER.NAME}")    

    log.info(
        f"Loaded configuration file: \n"
        f"\tadapter: {args.adapter_config_file}\n"
        f"\tdataset: {args.dataset_config_file}\n"
        f"\torder: {args.order_config_file}"
    )
    eval(cfg)

if __name__ == "__main__":
    main()
