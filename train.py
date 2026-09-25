import os
import sys
sys.path.extend(['./models', './data_loader'])
import torch
import logging
import importlib
from datetime import datetime
from opt import parse_args
import numpy as np


def setlogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def create_file_and_logger(args):
    file_name = f'[{args.source}]To[{args.target}]_'+ f'[{args.random_state}]' 
    # ex) base_model_bs16_ep30_lr1e-3
    lr_str = f"{args.lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    hyperparam_suffix = f"_bs{args.batch_size}_ep{args.max_epoch}_lr{lr_str}"
    model_dir_name = args.model_name + hyperparam_suffix
    if args.model_name == 'DAREGRAM':
        args.run_tag = f'align_scale{args.align_scale:g}_seed{args.random_state}'
        model_dir_name += '_' + args.run_tag
        file_name += '_' + args.run_tag
    
    save_dir = os.path.join(args.save_dir, model_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir
    args.save_path = os.path.join(save_dir, file_name)

    logger = setlogger(args.save_path + '.log')

    logging.info("==== Runtime Args ====")
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")

    return logger, args


if __name__ == '__main__':
    os.environ['NUMEXPR_MAX_THREADS'] = '8'
    args = parse_args()

    # fix seed
    if args.random_state is not None:
        os.environ['PYTHONHASHSEED'] = str(args.random_state)
        np.random.seed(args.random_state)
        torch.manual_seed(args.random_state)
        torch.cuda.manual_seed(args.random_state)
        torch.cuda.manual_seed_all(args.random_state)
        torch.backends.cudnn.deterministic = True

    args.source_condition = args.source
    args.target_condition = args.target

    # file/logger
    logger, args = create_file_and_logger(args)

    # load model trainer (models/{model_name}.py inside Trainset class)
    trainer = importlib.import_module(f"models.{args.model_name}").Trainset(args)

    try:
        if args.load_path:
            logging.info(f"Load weights from: {args.load_path}")
            trainer.load_model()
            trainer.test()
            if not args.save:
                try: os.remove(args.save_path + '.log')
                except OSError: pass
        else:
            trainer.train()
            if args.save:
                trainer.save_model()
            # else:
            #     try: os.remove(args.save_path + '.log')
            #     except OSError: pass
    finally:
        logger.handlers.clear()
