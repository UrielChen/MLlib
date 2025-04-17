import numpy as np
from src.tools.draw import plot_line


class TrainSummary:
    def __init__(self):
        self.batch_loss = {"train": [], "valid": []}
        self.batch_acc = {"train": [], "valid": []}
        self.epoch_loss = {"train": [], "valid": []}
        self.epoch_acc = {"train": [], "valid": []}
        self.epoch_mae = {"train": [], "valid": []}
        self.epoch_mse = {"train": [], "valid": []}
        self._best_acc = [0]
        self._best_epoch = [0]
        self._best_mae = [0]
        self._best_mse = [0]
        self.model_save_flag = [0]
        self.valid_info = {"ocean_acc": []}

    def update_best_acc(self, acc):
        self._best_acc.append(acc)

    def update_best_mae(self, acc):
        self._best_mae.append(acc)

    def update_best_mse(self, acc):
        self._best_mse.append(acc)

    def update_best_epoch(self, epo):
        self._best_epoch.append(epo)

    def update_model_save_flag(self, flag):
        """
        mark whether to save model weights
        args:
            flag (int 0 / 1): 0, mark as not save, otherwise save

        """
        self.model_save_flag.append(flag)

    def record_valid_ocean_acc(self, ocean):
        self.valid_info["ocean_acc"].append(ocean)

    def record_train_loss(self, loss_train):
        self.batch_loss["train"].append(loss_train)
        self.update_epoch_train_loss(loss_train)

    def record_train_acc(self, acc_train):
        self.batch_acc["train"].append(acc_train)
        self.update_epoch_train_acc(acc_train)

    def record_train_mae(self, mae_train):
        self.batch_loss["train"].append(mae_train)
        self.update_epoch_train_mae(mae_train)

    def record_train_mse(self, mse_train):
        self.batch_loss["train"].append(mse_train)
        self.update_epoch_train_mse(mse_train)

    def record_valid_loss(self, loss_valid):
        self.batch_loss["valid"].append(loss_valid)
        self.update_epoch_valid_loss(loss_valid)

    def record_valid_acc(self, acc_valid):
        self.batch_acc["valid"].append(acc_valid)
        self.update_epoch_valid_acc(acc_valid)

    def record_valid_mae(self, mae_valid):
        self.batch_loss["train"].append(mae_valid)
        self.update_epoch_valid_mae(mae_valid)

    def record_valid_mse(self, mse_valid):
        self.batch_loss["train"].append(mse_valid)
        self.update_epoch_valid_mse(mse_valid)

    @property
    def model_save(self):
        return self.model_save_flag[-1] > 0

    @property
    def best_valid_acc(self):
        return self._best_acc[-1]

    @property
    def best_valid_mae(self):
        return self._best_mae[-1]

    @property
    def best_valid_mse(self):
        return self._best_mse[-1]

    @property
    def best_epoch(self):
        return self._best_epoch[-1]

    @property
    def epoch_train_acc(self):
        return self.epoch_acc["train"][-1]

    @property
    def epoch_valid_acc(self):
        return self.epoch_acc["valid"][-1]

    @property
    def epoch_train_mae(self):
        return self.epoch_mae["train"][-1]

    @property
    def epoch_valid_mae(self):
        return self.epoch_mae["valid"][-1]

    @property
    def epoch_train_mse(self):
        return self.epoch_mse["train"][-1]

    @property
    def epoch_valid_mse(self):
        return self.epoch_mse["valid"][-1]

    @property
    def epoch_train_loss(self):
        return self.epoch_loss["train"][-1]

    @property
    def epoch_valid_loss(self):
        return self.epoch_loss["valid"][-1]

    @property
    def valid_ocean_acc(self):
        return self.valid_info["ocean_acc"][-1]

    def update_epoch_train_loss(self, loss_list):
        """record mean loss value of each training epoch"""
        self.epoch_loss["train"].append(np.mean(np.array(loss_list)))

    def update_epoch_train_acc(self, acc_list):
        """record mean loss value of each training epoch"""
        self.epoch_acc["train"].append(np.mean(np.array(acc_list)))

    def update_epoch_train_mae(self, mae_list):
        """record mean loss value of each training epoch"""
        self.epoch_mae["train"].append(np.mean(np.array(mae_list)))

    def update_epoch_train_mse(self, mse_list):
        """record mean loss value of each training epoch"""
        self.epoch_mse["train"].append(np.mean(np.array(mse_list)))

    def update_epoch_valid_loss(self, loss_list):
        """record mean loss value of each validation epoch"""
        self.epoch_loss["valid"].append(np.mean(np.array(loss_list)))

    def update_epoch_valid_acc(self, acc_list):
        """record mean loss value of each validation epoch"""
        self.epoch_acc["valid"].append(np.mean(np.array(acc_list)))

    def update_epoch_valid_mae(self, mae_list):
        """record mean loss value of each validation epoch"""
        self.epoch_mae["valid"].append(np.mean(np.array(mae_list)))

    def update_epoch_valid_mse(self, mse_list):
        """record mean loss value of each validation epoch"""
        self.epoch_mse["valid"].append(np.mean(np.array(mse_list)))

    def draw_epo_info(self, epochs=None, log_dir=None):
        if not epochs:
            epochs = len(self.epoch_loss["train"])
        plt_x = np.arange(0, epochs)
        plot_line(
            plt_x, self.epoch_loss["train"],
            plt_x, self.epoch_loss["valid"],
            mode="loss", out_dir=log_dir
        )
        plot_line(
            plt_x, self.epoch_acc["train"],
            plt_x, self.epoch_acc["valid"],
            mode="acc", out_dir=log_dir
        )

    def draw_batch_info(self, log_dir):
        plt_x_batch = np.arange(1, len(self.batch_loss["train"]) + 1)
        plot_line(
            plt_x_batch, self.batch_loss["train"],
            plt_x_batch, self.batch_acc["train"],
            mode="batch info", out_dir=log_dir
        )
