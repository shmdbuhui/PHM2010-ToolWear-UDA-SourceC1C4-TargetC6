import torch
import logging
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
import utils
from train_utils import InitTrain
import numpy as np        
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import hashlib
from networks.resnet import ResNet18
class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        self.feature_extractor = ResNet18().to(self.device)
        self.regressor = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256,1)
        ).to(self.device)
        digest = hashlib.sha256()
        for module in (self.feature_extractor, self.regressor):
            for name, tensor in sorted(module.state_dict().items()):
                digest.update(name.encode('utf-8'))
                digest.update(tensor.detach().cpu().numpy().tobytes())
        logging.info(f'Initial model SHA256: {digest.hexdigest()}')
        self._init_data()

    # ==================== DAR-RSD LOSS ====================
    def DARE_GRAM_LOSS(self,feat_s, feat_t):    
        b,p = feat_s.shape

        A = torch.cat((torch.ones(b,1).to(self.device), feat_s), 1)
        B = torch.cat((torch.ones(b,1).to(self.device), feat_t), 1)

        cov_A = (A.t()@A)
        cov_B = (B.t()@B) 

        _,L_A,_ = torch.linalg.svd(cov_A)
        _,L_B,_ = torch.linalg.svd(cov_B)
        

        den = L_A.detach().sum()
        den2 = L_B.detach().sum()
        eigen_A = torch.cumsum(L_A.detach(), dim=0)/den
        eigen_B = torch.cumsum(L_B.detach(), dim=0)/den2

        if(eigen_A[1]>0.9):
            T = eigen_A[1].detach()
        else:
            T = 0.9
            
        index_A = torch.argwhere(eigen_A.detach()<=T)[-1]

        if(eigen_B[1]>0.9):
            T = eigen_B[1].detach()
        else:
            T = 0.9

        index_B = torch.argwhere(eigen_B.detach()<=T)[-1]
        
        k = max(index_A, index_B)[0]

        A = torch.linalg.pinv(cov_A ,rtol = (L_A[k]/L_A[0]).detach())
        B = torch.linalg.pinv(cov_B ,rtol = (L_B[k]/L_B[0]).detach())
        
        cos_sim = nn.CosineSimilarity(dim=0,eps=1e-6)
        cos = torch.dist(torch.ones((p+1)).to(self.device),(cos_sim(A,B)),p=1)/(p+1)
        return 0.1*(cos) + 0.01*torch.dist((L_A[:k]),(L_B[:k]))/k


    def save_model(self):
        torch.save({
            'feature_extractor': self.feature_extractor.state_dict(),
            'regressor': self.regressor.state_dict()
            }, self.args.save_path + '.pth')
        logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))
    
    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path, map_location=self.device)
        self.feature_extractor.load_state_dict(ckpt['feature_extractor'])
        self.regressor.load_state_dict(ckpt['regressor'])
        
    def train(self):
        args = self.args
        self.optimizer = torch.optim.Adam([
            {"params": self.feature_extractor.parameters(), "lr": args.lr * 0.5},
            {"params": self.regressor.parameters(), "lr": args.lr},
        ])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        for epoch in range(1, args.max_epoch+1):
            logging.info('-'*5 + f'Epoch {epoch}/{args.max_epoch}' + '-'*5)
            
            if self.lr_scheduler is not None:
                logging.info(f'current lr: {self.lr_scheduler.get_last_lr()}')
   
            self.feature_extractor.train()
            self.regressor.train()
            running_loss = 0.0
            epoch_loss = defaultdict(float)
            tradeoff = self._get_tradeoff(args.tradeoff, epoch)
            num_iter = len(self.dataloaders['source_train'])               
            source_order_hash = hashlib.sha256()
            target_order_hash = hashlib.sha256()

            for batch_idx in tqdm(range(num_iter), ascii=True):
                # source / target batch
                source_data, source_labels = utils.get_next_batch(
                    self.dataloaders, self.iters, 'source_train', self.device
                )
                source_labels = source_labels.float().unsqueeze(1)  # (B,1)

                target_data, _ = utils.get_next_batch(
                    self.dataloaders, self.iters, 'target_unlabeled', self.device
                )
                source_order_hash.update(source_data.detach().cpu().numpy().tobytes())
                target_order_hash.update(target_data.detach().cpu().numpy().tobytes())

                # forwards
                self.optimizer.zero_grad()
                
                feat_s = self.feature_extractor(source_data)              
                feat_t = self.feature_extractor(target_data)            

                # regressor
                y_s = self.regressor(feat_s)             
                # supervised regression loss (source only)
                loss_c = F.mse_loss(y_s, source_labels)

                # DARE-GRAM domain alignment loss
                loss_gram = self.DARE_GRAM_LOSS(feat_s, feat_t)

                # total loss
                alignment_term = args.align_scale * tradeoff[0] * loss_gram
                loss = loss_c + alignment_term

                if epoch == 2 and batch_idx == 0:
                    grad_s, grad_t = torch.autograd.grad(
                        alignment_term, (feat_s, feat_t), retain_graph=True
                    )
                    logging.info(f'Alignment gradient norm, source={grad_s.norm().item():.9g}, '
                                 f'target={grad_t.norm().item():.9g}')

                running_loss += loss.item()
                epoch_loss['Regressor(MSE)'] += loss_c.item()
                epoch_loss['DARE-GRAM'] += loss_gram.item()
                epoch_loss['Weighted alignment'] += alignment_term.item()

                # backward
                loss.backward()


                self.optimizer.step()
                
            logging.info(f"Train-Total Loss: {running_loss/num_iter:.4f}")
            logging.info(f"Train-Regressor(MSE): {epoch_loss['Regressor(MSE)']/num_iter:.4f}")
            logging.info(f"Train-DARE-GRAM Loss: {epoch_loss['DARE-GRAM']/num_iter:.4f}")
            logging.info(f"Train-Weighted Alignment: {epoch_loss['Weighted alignment']/num_iter:.6f}")
            logging.info(f"Batch order SHA256: source={source_order_hash.hexdigest()} "
                         f"target={target_order_hash.hexdigest()}")
            logging.info(f"Backbone BN batches tracked: "
                         f"{self.feature_extractor.backbone.bn1.num_batches_tracked.item()}")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
 
        self.test()
            
    def test(self):
        self.feature_extractor.eval()
        self.regressor.eval()
        args = self.args

        def eval_loader(dataloader, name="", visualize=False):
            y_true, y_pred = [], []
            with torch.no_grad():
                for xb, yb in dataloader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device).float().unsqueeze(1)  
                    feat = self.feature_extractor(xb)
                    pred = self.regressor(feat)
                    y_true.extend(yb.cpu().numpy())
                    y_pred.extend(pred.cpu().numpy())

            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

            mae  = mean_absolute_error(y_true, y_pred)
            mse  = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2   = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / (np.maximum(np.abs(y_true), 1e-8))))

            logging.info(
                f"[{name}] MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}, MAPE={mape*100:.2f}%"
            )

            if visualize:
                save_dir = os.path.join("visualization", f"{args.model_name}_{args.run_tag}")
                os.makedirs(save_dir, exist_ok=True)

                passes = np.arange(1, len(y_true) + 1)
                file_stem = f"{args.source_condition}_tgt-{args.target_condition}_{args.run_tag}"
                np.savetxt(os.path.join(save_dir, file_stem + ".csv"),
                           np.column_stack((passes, y_true, y_pred)), delimiter=",",
                           header="sample_index,y_true_vb,y_pred_vb", comments="")
                plt.figure(figsize=(7, 4))
                plt.plot(passes, y_true, label="True VB", linewidth=2)
                plt.plot(passes, y_pred, label="Predicted VB", linestyle="--", linewidth=2)
                plt.xlabel("Sample index")
                plt.ylabel("Flank wear (VB)")  
                plt.title(
                    f"Source={args.source_condition.upper()} → Target={args.target_condition.upper()}\n"
                    f"MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}, MAPE={mape*100:.2f}%"
                )
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.7)
                plt.tight_layout()

                out_path = os.path.join(save_dir, file_stem + ".png")
                plt.savefig(out_path, dpi=300)
                plt.close()
                logging.info(f"Saved visualization: {out_path}")

            return mae, rmse, r2, mape

        eval_loader(self.dataloaders['target_test'], name="Target test", visualize=True)
