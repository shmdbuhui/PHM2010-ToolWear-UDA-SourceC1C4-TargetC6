import torch
import logging
from tqdm import tqdm
import torch.nn.functional as F
import utils
from train_utils import InitTrain
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from networks.resnet import ResNet18



class Trainset(InitTrain):
    def __init__(self, args):
        super(Trainset, self).__init__(args)

        self.model = nn.Module()
        # feature extractor: (B,6,128,128) -> (B,512)
        self.model.feature_extractor = ResNet18().to(self.device)

        # regressor: 512 -> 1
        self.model.regressor = nn.Sequential(
            nn.Linear(512,1)
        ).to(self.device)

        self._init_data()

    def save_model(self):
        torch.save({'model': self.model.state_dict()}, self.args.save_path + '.pth')
        logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))

    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])

    def train(self):
        args = self.args
        self.optimizer = torch.optim.Adam([
            {"params": self.model.feature_extractor.parameters(), "lr": args.lr},
            {"params": self.model.regressor.parameters(), "lr": args.lr},
        ])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        for epoch in range(1, args.max_epoch + 1):
            logging.info('-' * 5 + f'Epoch {epoch}/{args.max_epoch}' + '-' * 5)
            if self.lr_scheduler is not None:
                logging.info(f'current lr: {self.lr_scheduler.get_last_lr()}')

            self.model.train()
            running_loss = 0.0

            num_iter = len(self.dataloaders['source_train'])
            for _ in tqdm(range(num_iter), ascii=True):
                source_data, source_labels = utils.get_next_batch(
                    self.dataloaders, self.iters, 'source_train', self.device
                )
                source_labels = source_labels.float().unsqueeze(1)  # (B,1)
                print(source_data.shape, source_labels.shape)

                self.optimizer.zero_grad()
                feat = self.model.feature_extractor(source_data)   # (B,512)
                pred = self.model.regressor(feat)                  # (B,1)
                loss = F.mse_loss(pred, source_labels)
                running_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            logging.info(f"Train-MSE Loss: {running_loss / num_iter:.4f}")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        self.test()

    def test(self):
        self.model.eval()
        args = self.args

        def eval_loader(dataloader, name="", visualize=False):
            y_true, y_pred = [], []
            with torch.no_grad():
                for xb, yb in dataloader:
                    xb = xb.to(self.device)                         # (B,6,128,128)
                    yb = yb.to(self.device).float().unsqueeze(1)    # (B,1)
                    feat = self.model.feature_extractor(xb)         # (B,512)
                    pred = self.model.regressor(feat)               # (B,1)
                    y_true.extend(yb.cpu().numpy())
                    y_pred.extend(pred.cpu().numpy())

            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

            mae  = mean_absolute_error(y_true, y_pred)
            mse  = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2   = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / (np.maximum(np.abs(y_true), 1e-8))))

            logging.info(f"[{name}] MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}, MAPE={mape*100:.2f}%")

            if visualize:
                save_dir = os.path.join("visualization", args.model_name)
                os.makedirs(save_dir, exist_ok=True)

                passes = np.arange(1, len(y_true) + 1)
                np.savetxt(os.path.join(save_dir, f"{args.source_condition}_tgt-{args.target_condition}.csv"),
                           np.column_stack((passes, y_true, y_pred)), delimiter=",",
                           header="cut,y_true,y_pred", comments="")
                plt.figure(figsize=(7, 4))
                plt.plot(passes, y_true, label="True VB", linewidth=2)
                plt.plot(passes, y_pred, label="Predicted VB", linestyle="--", linewidth=2)
                plt.xlabel("Pass index")
                plt.ylabel("Flank wear (VB)")
                plt.title(
                    f"Source={args.source_condition.upper()} → Target={args.target_condition.upper()}\n"
                    f"MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}, MAPE={mape*100:.2f}%"
                )
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.7)
                plt.tight_layout()

                out_path = os.path.join(save_dir, f"{args.source_condition}_tgt-{args.target_condition}.png")
                plt.savefig(out_path, dpi=300)
                plt.close()
                logging.info(f"Saved visualization: {out_path}")

            return mae, rmse, r2, mape

        eval_loader(self.dataloaders['target_test'], name="target_test", visualize=True)
