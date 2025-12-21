import os
import math
import torch
import logging
import numpy as np
import pandas as pd
from torch import optim
from data_loader import data_utils
from data_loader import aug


class InitTrain(object):

    def __init__(self, args):
        self.args = args
        if args.cuda_device:
            self.device = torch.device("cuda:" + args.cuda_device)
            logging.info('using {} / {} gpus'.format(len(args.cuda_device.split(',')), torch.cuda.device_count()))
        else:
            self.device = torch.device("cpu")
            logging.info('using cpu')

    def _get_lr_scheduler(self, optimizer):
        args = self.args
        valid = ['step', 'exp', 'stepLR', 'fix', 'cos']
        assert args.lr_scheduler in valid, f"lr scheduler should be one of {valid}, but got {args.lr_scheduler}"

        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=args.gamma)

        elif args.lr_scheduler == 'exp':
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=steps, gamma=args.gamma)

        elif args.lr_scheduler == 'fix':
            lr_scheduler = None

        elif args.lr_scheduler == 'cos':
            # cosine decay up to max_epoch
            T_max  = getattr(args, 't_max', getattr(args, 'max_epoch', getattr(args, 'epochs', 100)))
            eta_min = getattr(args, 'eta_min', 0.0)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min
            )

        return lr_scheduler

    def _get_optimizer(self, model):
        args = self.args
        if isinstance(model, list):
            par = [{'params': md.parameters()} for md in model]
        else:
            par = model.parameters()
        assert args.opt in ['sgd', 'adam'], f"optimizer should be 'sgd' or 'adam', but got {args.opt}"
        if args.opt == 'sgd':
            optimizer = optim.SGD(par, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            optimizer = optim.Adam(par, lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
        return optimizer

    def _get_tradeoff(self, tradeoff_list, epoch=None):
        tradeoff = []
        for item in tradeoff_list:
            if item == 'exp':
                tradeoff.append(2 / (1 + math.exp(-10 * (epoch-1) / (self.args.max_epoch-1))) - 1)
            elif isinstance(item, (float, int)):
                tradeoff.append(item)
            else:
                raise Exception(f"unknown trade-off type {item}")
        return tradeoff

    # ---------- NPZ loader (train/val split) ----------
    def _to_list_samples(self, X):
        """
        (N,C,L) ndarray or object-array -> List[np.ndarray]
        """
        if isinstance(X, np.ndarray) and X.dtype != object:
            return [X[i] for i in range(X.shape[0])]
        return list(X)


    def _load_split_npz(self, npz_dir: str, condition: str, split: str):
        """
        load samples(List[ndarray]), labels(np.float32) from {npz_dir}/{split}_{condition}.npz
        """
        fname = f"{split}_{condition}.npz"
        path = os.path.join(npz_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"NPZ not found: {path}")
        data = np.load(path, allow_pickle=True)
        samples = self._to_list_samples(data["samples"])
        labels = data["labels"].astype(np.float32)
        return samples, labels

    # ---------- Normalize(fixed channel statistics) ----------
    def _build_transforms(self, fixed_stats):
        return {
            'train': aug.Compose([aug.NormalizeFixed(fixed_stats), aug.Retype()]),
        }

    def _compute_norm_stats(self, data_series, eps: float = 1e-8):
        """
        data_series: iterable of arrays
        각 원소 shape = (C, H, W) 
        통계: channel-wise z-score
        """
        arrs = []

        for x in data_series:
            x = np.asarray(x, dtype=np.float32)
            if x.ndim != 3:
                raise ValueError(f"Expected (C,H,W), got shape={x.shape}")
            arrs.append(x)

        X = np.stack(arrs, axis=0)   # (N, C, H, W)

        # channel-wise statistics
        mean = X.mean(axis=(0, 2, 3))          # (C,)
        std  = X.std(axis=(0, 2, 3)) + eps      # (C,)

        return {
            "type": "zscore",
            "mean": mean.astype(np.float32),
            "std":  std.astype(np.float32),
        }



    # ---------- initialize data  ----------
    def _init_data(self):
        args = self.args
        self.datasets = {}

        npz_dir   = getattr(args, "npz_dir", "dataset")
        norm_type = getattr(args, "norm_type", "zscore")

        # ----- SOURCE (train) -----
        print(f"Loading SOURCE from NPZ: {args.source_condition}")
        src_tr_X, src_tr_y = self._load_split_npz(npz_dir, args.source_condition, "train")

        df_src_train = pd.DataFrame({"data": src_tr_X, "labels": src_tr_y})

        # normalize statistics are estimated only from source-train → fixed applied to all splits/domains
        fixed_stats = self._compute_norm_stats(df_src_train["data"])
        transforms  = self._build_transforms(fixed_stats)


        self.datasets['source_train'] = data_utils.dataset(df_src_train, transform=transforms['train'])
        logging.info(f"source training set: {len(self.datasets['source_train'])}")

        # ----- TARGET (train) -----
        print(f"Loading TARGET from NPZ: {args.target_condition}")
        tgt_tr_X, tgt_tr_y = self._load_split_npz(npz_dir, args.target_condition, "target")
        tgt_ul_tr_X, tgt_ul_tr_y = self._load_split_npz(npz_dir, args.target_condition, "unlabeled")

        df_tgt_train = pd.DataFrame({"data": tgt_tr_X, "labels": tgt_tr_y})
        df_tgt_ul_train = pd.DataFrame({"data": tgt_ul_tr_X, "labels": tgt_ul_tr_y})
    
        self.datasets['target_unlabeled'] = data_utils.dataset(df_tgt_ul_train, transform=transforms['train'])
        self.datasets['target_test'] = data_utils.dataset(df_tgt_train, transform=transforms['train'])
        logging.info(f"target unlabeled set: {len(self.datasets['target_unlabeled'])}")
        logging.info(f"target test set: {len(self.datasets['target_test'])}")

        # -----------------
        # Dataloaders
        # -----------------
        dataset_keys = ['source_train', 'target_unlabeled', 'target_test']
        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.datasets[x],
                batch_size=args.batch_size,
                shuffle=(x.endswith('train') or x == 'target_unlabeled'),
                num_workers=args.num_workers,
                drop_last=(x.endswith('train') or x == 'target_unlabeled'),  # for training, drop_last=True
                pin_memory=(self.device.type == 'cuda')
            )
            for x in dataset_keys
        }
        # create iterator for training only (target_test is used directly in test method)
        self.iters = {x: iter(self.dataloaders[x]) for x in ['source_train', 'target_unlabeled']}
