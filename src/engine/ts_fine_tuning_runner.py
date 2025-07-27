import os
from pathlib import Path
from datetime import datetime
from src.data.build import build_dataloader
from src.modeling.network.build import build_model
from src.modeling.loss.build import build_loss_func
from src.modeling.solver.build import build_solver, build_scheduler
from src.engine.build import build_trainer
from src.evaluate.summary import TrainSummary
from src.checkpoint.save import resume_training, load_model, load_pretrained, save_ts_model, save_ts_pred
from src.tools.logger import make_logger
import numpy as np
import pandas as pd
from utils.phil_test import phil_test_main


class FTConstructor:
    """
    construct certain experiment by the following template
    step 1: prepare dataloader
    step 2: prepare model and loss function
    step 3: select optimizer for gradient descent algorithm
    step 4: prepare trainer for typical training in pytorch manner
    """
    def __init__(self, cfg):
        """ run exp from config file

        arg:
            cfg_file: config file of an experiment
        """
        self.cfg = cfg
        self.data_loader = None
        self.model = None
        self.loss_f = None
        self.optimizer = None
        self.scheduler = None
        self.collector = None
        self.trainer = None

    def build(self):
        self.model = self.build_model()
        self.data_loader = self.build_dataloader()
        self.loss_f = self.build_loss_function()
        self.optimizer = self.build_solver()
        self.scheduler = self.build_scheduler()
        self.collector = TrainSummary()
        self.trainer = self.build_trainer()

    def build_dataloader(self):
        return build_dataloader(self.cfg)

    def build_model(self):
        return build_model(self.cfg)

    def build_loss_function(self):
        return build_loss_func(self.cfg)

    def build_solver(self):
        return build_solver(self.cfg, self.model)

    def build_scheduler(self):
        return build_scheduler(self.cfg, self.optimizer)

    def build_trainer(self):
        return build_trainer(self.cfg, self.collector, self.logger)


class TSFineTuningRunner(FTConstructor):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.logger, self.log_dir = make_logger(cfg.TRAIN.OUTPUT_DIR)
        self.pred_dir = Path(self.log_dir, "pred_dir")
        self.model_dir = Path(self.log_dir, "model_dir")
        self.build()
        self.logger.info(cfg)
        self.total_pred, self.total_targets = None, None

    def before_train(self, cfg, date):
        if cfg.RESUME:
            self.model, self.optimizer, epoch = resume_training(cfg.RESUME, self.model, self.optimizer)
            if cfg.RESUME_EPOCH == "resume":
                cfg.START_EPOCH = epoch
            elif cfg.RESUME_EPOCH == "restart":  # this is for pretraining and fine-tuning schema
                cfg.START_EPOCH = 0
            else:
                raise ValueError(cfg.RESUME_EPOCH)  # phil_todo

            self.logger.info(f"resume training from {cfg.RESUME}")
        if cfg.LOAD_PRETRAIN:  # phil_todo test it!
            self.model = load_pretrained(cfg.PRETRAINING_PATH, self.model)

        if self.cfg.SOLVER.RESET_LR:
            # self.logger.info("change learning rate form [{}] to [{}]".format(
            #     self.optimizer.param_groups[0]["lr"],
            #     self.cfg.SOLVER.LR_INIT,
            # ))
            # self.optimizer.param_groups[0]["lr"] = self.cfg.SOLVER.LR_INIT

            self.logger.info("Resetting LR and scheduler")

            # Manually set LR
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.cfg.SOLVER.LR_INIT

            # Rebuild scheduler
            self.scheduler = build_scheduler(self.cfg, self.optimizer)


        self.data_loader["train"].dataset.update(date)
        if cfg.IS_VALIDATE:
            self.data_loader["valid"].dataset.update(date)

    def train_epochs(self, cfg, date):
        for epoch in range(cfg.START_EPOCH, cfg.MAX_EPOCH):
            self.trainer.train(self.data_loader["train"], self.model, self.loss_f, self.optimizer, epoch)

            if cfg.IS_VALIDATE and epoch % cfg.VALID_INTERVAL == 0:
                self.trainer.valid(self.data_loader["valid"], self.model, self.loss_f, epoch)

            self.scheduler.step()
            if self.collector.model_save and epoch % cfg.VALID_INTERVAL == 0:
                self.model_dir.mkdir(parents=True, exist_ok=True)

                save_ts_model(epoch, self.collector.best_valid_acc, self.model, self.optimizer, self.model_dir, date)
                self.collector.update_best_epoch(epoch)
            if self.cfg.TRAIN.SAVE_LAST and epoch == (cfg.MAX_EPOCH - 1):
                self.model_dir.mkdir(parents=True, exist_ok=True)
                save_ts_model(epoch, self.collector.best_valid_acc, self.model, self.optimizer, self.model_dir, date)

    def after_train(self, cfg, date):
        # cfg = self.cfg.TRAIN
        # self.collector.draw_epo_info(log_dir=self.log_dir)
        self.logger.info(
            "Training on {} done at {}, best mse: {} in :{}".format(
                date,
                datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
                self.collector.best_valid_mse,
                self.collector.best_epoch,
            )
        )

    def before_test(self, cfg, date):
        ##############
        # Load Model #
        ##############
        self.logger.info(f"On Test, date: {date}")
        if cfg.LOAD_TODAY_MODEL:

            weight_file = os.path.join(self.model_dir, f"{date}.pkl")
            self.logger.info(f"test with model on date {date} at {weight_file}")
            self.model = load_model(self.model, weight_file)
        elif cfg.WEIGHT:
            self.model = load_model(self.model, cfg.WEIGHT)
        else:
            try:
                weights = [
                    file for file in os.listdir(self.log_dir)
                    if file.endswith(".pkl") and ("last" not in file)
                ]
                weights = sorted(weights, key=lambda x: int(x[11:-4]))
                weight_file = os.path.join(self.log_dir, weights[-1])
            except IndexError:
                weight_file = os.path.join(self.log_dir, "checkpoint_last.pkl")
            self.logger.info(f"test with model {weight_file}")
            self.model = load_model(self.model, weight_file)

        self.pred_dir.mkdir(parents=True, exist_ok=True)
        ####################
        # Update Test Data #
        ####################
        self.data_loader["test"].dataset.update(date)
        return

    def on_test(self, cfg, date):
        curr_preds, curr_targets = self.trainer.test(
            self.data_loader["test"], self.model
        )
        save_ts_pred(curr_preds, curr_targets, self.pred_dir, date)

        if self.total_pred is None:
            self.total_pred = curr_preds
            self.total_targets = curr_targets
        else:
            self.total_pred = np.concatenate([self.total_pred, curr_preds], axis=0)  # todo wrong direction
            self.total_targets = np.concatenate([self.total_targets, curr_targets], axis=0)

        mae = np.mean(np.abs(curr_preds - curr_targets))
        mse = np.mean((curr_preds - curr_targets) ** 2)
        self.logger.info(f"Test on date: {date} mae: {mae} ; mse: {mse}")
        return

    def after_test(self, cfg, date):

        return

    def train(self, date):
        cfg = self.cfg.TRAIN
        self.before_train(cfg, date)
        self.train_epochs(cfg, date)
        self.after_train(cfg, date)

    def test(self, date, weight=None):
        cfg = self.cfg.TEST
        # cfg.WEIGHT = weight if weight else cfg.WEIGHT
        self.before_test(cfg, date)
        self.on_test(cfg, date)
        self.after_test(cfg, date)
        return

    def run(self, train_only=False, test_only=False):
        date_list = np.loadtxt(self.cfg.TS_FINE_TUNING.DATE_LIST, dtype=str, delimiter=',')
        fine_tuning_start = self.cfg.TS_FINE_TUNING.START
        fine_tuning_end = self.cfg.TS_FINE_TUNING.END
        date_list = date_list[(date_list >= fine_tuning_start) & (date_list <= fine_tuning_end)]
        for date in date_list:
            if not train_only and not test_only:
                self.train(date)
                self.test(date)
            elif train_only and not test_only:
                self.train(date)
            elif not train_only and test_only:
                self.test(date)
            else:
                raise ValueError()

            self.clean_up()

            if self.cfg.TRAIN.LOAD_PRETRAIN:
                self.model = load_pretrained(self.cfg.PRETRAINING_PATH, self.model)
            else:
                self.model = build_model(self.cfg)

            self.optimizer = build_solver(self.cfg, self.model)
            self.scheduler = build_scheduler(self.cfg, self.optimizer)

        np.save(Path(self.log_dir, "total_pred.npy"), self.total_pred)
        np.save(Path(self.log_dir, "total_targets.npy"), self.total_targets)
        phil_test_main(self.log_dir, self.total_pred, self.total_targets, 49)

    def clean_up(self):
        import torch, gc
        torch.cuda.empty_cache()
        gc.collect()
        self.collector = TrainSummary()
        self.trainer = self.build_trainer()
