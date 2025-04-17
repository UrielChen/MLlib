import torch
from tqdm import tqdm
import numpy as np
from .build import TRAINER_REGISTRY
from .build import device
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score, mean_squared_error, mean_absolute_error
import time


class Trainer:

    def train(self, data_loader, model, loss_func, optimizer, epoch_idx):
        pass

    def valid(self, data_loader, model, loss_func, epoch_idx):
        pass

    def test(self, data_loader, model):
        pass


@TRAINER_REGISTRY.register()
class SimpleRegressionTrainer(Trainer):
    """Trainer for regression tasks"""
    def __init__(self, cfg, collector, logger):
        self.cfg = cfg
        self.clt = collector
        self.logger = logger
        self.tb_writer = SummaryWriter(cfg.OUTPUT_DIR)
        self.device = device
        print(f"Training on device: {device}")
        if device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(device)
            print(f"GPU model: {gpu_name}")
        else:
            print("Using CPU")

    def train(self, data_loader, model, loss_func, optimizer, epoch_idx):
        lr = optimizer.param_groups[0]['lr']
        self.logger.info(f"Training: learning rate:{lr}")
        model.train()
        loss_list = []
        mae_list = []
        mse_list = []
        for i, data in enumerate(data_loader):
            inputs, targets = self.data_fmt(data)
            preds = model(*inputs).squeeze()
            optimizer.zero_grad()
            loss = loss_func(preds, targets)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            mae_list.append(mean_absolute_error(preds.cpu().detach().numpy(), targets.cpu().detach().numpy()))
            mse_list.append(mean_squared_error(preds.cpu().detach().numpy(), targets.cpu().detach().numpy()))
            if i % self.cfg.LOG_INTERVAL == self.cfg.LOG_INTERVAL - 1:
                self.logger.info(
                    f"Train: Epoch[{epoch_idx+1:03d}/{self.cfg.MAX_EPOCH}] "
                    f"Batch[{i+1:03d}/{len(data_loader)}] -- LOSS: {loss.item():.10f}"
                )
            self.tb_writer.add_scalar("epo_loss", np.mean(loss_list), epoch_idx)
            self.clt.record_train_loss(loss_list)
            self.clt.record_train_mae(mae_list)
            self.clt.record_train_mse(mse_list)

    def valid(self, data_loader, model, loss_func, epoch_idx):
        model.eval()
        loss_batch_list, mae_batch_list, mse_batch_list = [], [], []
        all_preds, all_targets = [], []
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inputs, targets = self.data_fmt(data)
                preds = model(*inputs).squeeze()
                loss = loss_func(preds, targets)
                loss_batch_list.append(loss.item())

                mae_batch = mean_absolute_error(preds.cpu().detach().numpy(), targets.cpu().detach().numpy())
                mae_batch_list.append(mae_batch)

                mse_batch = mean_squared_error(preds.cpu().detach().numpy(), targets.cpu().detach().numpy())
                mse_batch_list.append(mse_batch)

                # all_preds.append(preds.cpu())  # phil_todo, here i store the validation result
                # all_targets.append(targets.cpu())
            mae_epo_mean = np.array(mae_batch_list).mean()
            mse_epo_mean = np.array(mae_batch_list).mean()
            self.tb_writer.add_scalar("valid_mae", mae_epo_mean, epoch_idx)
            self.tb_writer.add_scalar("valid_mse", mse_epo_mean, epoch_idx)

        self.clt.record_valid_loss(loss_batch_list)
        self.clt.record_valid_mae(mae_batch_list)
        self.clt.record_valid_mse(mse_batch_list)

        # all_preds = torch.cat(all_preds).numpy()
        # all_targets = torch.cat(all_targets).numpy()
        # mae = mean_absolute_error(all_preds, all_targets)  # todo, why？ can be changed to other type, loss func?
        # mse = mean_squared_error(all_preds, all_targets)


        if mse_epo_mean < self.clt.best_valid_mse:
            self.clt.update_best_mse(mse_epo_mean)  # still uses the same interface
            self.clt.update_model_save_flag(1)
        else:
            self.clt.update_model_save_flag(0)

        self.logger.info(
            "Valid: Epoch[{:0>3}/{:0>3}] Train Mean_MSE: {:.10f} Valid MSE:{:.10f} \n".
            format(
                epoch_idx + 1, self.cfg.MAX_EPOCH,
                float(self.clt.epoch_train_mse),
                float(self.clt.epoch_valid_mse),
            )
        )


    def test(self, data_loader, model):
        model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for data in tqdm(data_loader):
                inputs, targets = self.data_fmt(data)
                preds = model(*inputs).squeeze()
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        total_preds = torch.cat(all_preds).numpy()
        total_targets = torch.cat(all_targets).numpy()

        mae = np.mean(np.abs(total_preds - total_targets))
        mse = np.mean((total_preds - total_targets) ** 2)

        return round(mae, 4), round(mse, 4)

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        data_in, targets = data["data"], data["targets"]
        return (data_in,), targets


@TRAINER_REGISTRY.register()
class SimpleTrainer(Trainer):
    """base trainer for bi-modal input"""
    def __init__(self, cfg, collector, logger):
        self.cfg = cfg
        self.clt = collector
        self.logger = logger
        self.tb_writer = SummaryWriter(cfg.OUTPUT_DIR)
        self.device = device
        # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, data_loader, model, loss_func, optimizer, epoch_idx):
        lr = optimizer.param_groups[0]['lr']
        self.logger.info(f"Training: learning rate:{lr}")
        # self.tb_writer.add_scalar("lr", lr, epoch_idx)
        model.train()
        acc_batch_list, loss_list = [], []
        for i, data in enumerate(data_loader):
            inputs, labels = self.data_fmt(data)
            outputs = model(*inputs)
            outputs = torch.squeeze(outputs)
            optimizer.zero_grad()
            loss = loss_func(outputs.cpu(), labels.cpu())
            loss.backward()
            optimizer.step()
            acc_batch = (outputs.argmax(dim=1) == labels).float().mean().item()
            acc_batch_list.append(acc_batch)
            loss_list.append(loss.item())

            # print loss and training info for an interval
            if i % self.cfg.LOG_INTERVAL == self.cfg.LOG_INTERVAL - 1:
                self.logger.info(
                    "Train: Epoch[{:0>3}/{:0>3}] Batch[{:0>3}/{:0>3}] -- LOSS: {:.4f} ACC:{:.4f} ".format(
                        epoch_idx + 1, self.cfg.MAX_EPOCH,   # Epo
                        i + 1, len(data_loader),                     # Iter
                        float(loss.item()), float(acc_batch),  # LOSS ACC ETA
                    )
                )

        self.tb_writer.add_scalar("epo_loss", np.array(loss_list).mean(), epoch_idx)
        self.clt.record_train_loss(loss_list)
        self.clt.record_train_acc(acc_batch_list)

    def valid(self, data_loader, model, loss_func, epoch_idx):
        model.eval()
        with torch.no_grad():
            loss_batch_list = []
            acc_batch_list = []
            acc_epoch = []
            for i, data in enumerate(data_loader):
                inputs, labels = self.data_fmt(data)
                outputs = model(*inputs)
                outputs = torch.squeeze(outputs)
                # phil_todo, you just calculate the metrics, so here you do not store the validation result?
                loss = loss_func(outputs.cpu(), labels.cpu())
                loss_batch_list.append(loss.item())

                acc_batch = (outputs.argmax(dim=1) == labels).float().mean().item()
                acc_batch_list.append(acc_batch)
            acc_epo_mean = np.array(acc_batch_list).mean()  # ocean acc on all valid images
            self.tb_writer.add_scalar("valid_acc", acc_epo_mean, epoch_idx)
        self.clt.record_valid_loss(loss_batch_list)
        self.clt.record_valid_acc(acc_batch_list)  # acc over batches
        if acc_epo_mean > self.clt.best_valid_acc:
            self.clt.update_best_acc(acc_epo_mean)
            self.clt.update_model_save_flag(1)
        else:
            self.clt.update_model_save_flag(0)

        self.logger.info(
            "Valid: Epoch[{:0>3}/{:0>3}] Train Mean_Acc: {:.2%} Valid Mean_Acc:{:.2%} \n".
            format(
                epoch_idx + 1, self.cfg.MAX_EPOCH,
                float(self.clt.epoch_train_acc),
                float(self.clt.epoch_valid_acc),
            )
        )

    def test(self, data_loader, model):
        # mse_func = torch.nn.MSELoss(reduction="none")
        model.eval()
        with torch.no_grad():
            acc_epo = []
            label_list = []
            pred_list = []
            for data in tqdm(data_loader):
                inputs, labels = self.data_fmt(data)
                outputs = model(*inputs)  # for more then one input data

                outputs = outputs.cpu().detach()
                outputs = torch.squeeze(outputs)
                pred = outputs.argmax(dim=1)
                labels = labels.cpu().detach()
                pred_list.append(pred)
                label_list.append(labels)
                acc_batch = (pred == labels).float().mean().item()
                acc_epo.append(acc_batch)
            test_acc = np.array(acc_epo).mean()
        total_label = torch.cat(label_list, dim=0).numpy()
        total_pred = torch.cat(pred_list, dim=0).numpy()
        uar = balanced_accuracy_score(total_label, total_pred)
        acc_avg_rand = np.round(test_acc.astype("float64"), 4)
        uar_rand = np.round(uar.astype("float64"), 4)
        # self.tb_writer.add_scalar("test_acc", ocean_acc_avg_rand)
        return acc_avg_rand, uar_rand

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        data_in, labels = data["data"], data["label"]
        return (data_in,), labels


@TRAINER_REGISTRY.register()
class ConvTrainer(SimpleTrainer):

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        data_in, labels = data["data"], data["label"]
        data_in = data_in.reshape((-1, 1, 1, 194))
        return (data_in,), labels


@TRAINER_REGISTRY.register()
class TransformerTrainer(SimpleTrainer):

    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        data_in, labels = data["data"], data["label"]
        bs, dim = data_in.shape
        tmp = torch.zeros(bs, 200).to(self.device)
        tmp[:, :194] = data_in
        data_in = tmp.reshape(bs, 1, 200)
        return (data_in,), labels


@TRAINER_REGISTRY.register()
class PhilNNTrainer(SimpleRegressionTrainer):
    def data_fmt(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        data_in, target = data["data"], data["target"]
        return (data_in,), target
