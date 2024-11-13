import argparse
import csv
import logging
import pathlib
import time
from dataclasses import dataclass
import optuna 

from torch import manual_seed, Tensor, log
from torch.nn import Module, modules, MSELoss, L1Loss, KLDivLoss, HuberLoss, SmoothL1Loss
from torch.optim import Adam, Optimizer, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from pathlib import Path

# from postprocessing.visualization import visualizations
from processing.networks.unet import UNet
from processing.networks.model import weights_init
from processing.networks.unet3d import weights_init_3d

class KLD_log():
    def __init__(self):
        self.kld = KLDivLoss()

    def __call__(self, prediction: Tensor, label: Tensor):
        return self.kld(log(prediction), label)

@dataclass
class Solver(object):
    model: Module
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    loss_func: modules.loss._Loss = MSELoss()
    learning_rate: float = 1e-4
    opt: Optimizer = Adam
    finetune: bool = False
    best_model_params: dict = None
    metrics: dict = None

    def __post_init__(self):
        # self.opt = self.opt(self.model.parameters(), self.learning_rate, weight_decay=1e-4) # already done in optuna case
        # contains the epoch and learning rate, when lr changes
        self.lr_schedule = {0: self.opt.param_groups[0]["lr"]}
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.opt, patience=10, cooldown=10, factor=0.5, min_lr=1e-6)

        if not self.finetune:
            if self.model.__class__.__name__=="UNet3d":
                self.model.apply(weights_init_3d)
            else:
                self.model.apply(weights_init) 
        
        self.metrics: dict = {"MSE": MSELoss(), "MAE": L1Loss()}#, "KLD": KLD_log(), "Huber": HuberLoss(), "SmoothL1": SmoothL1Loss()}

    def train(self, trial, args: dict):
        manual_seed(0)
        start_time = time.perf_counter()
        # initialize tensorboard
        writer = SummaryWriter(args["destination"])
        device = args["device"]
        # writer.add_graph(self.model, next(iter(self.train_dataloader))[0].to(device))

        epochs = tqdm(range(args["epochs"]), desc="epochs", disable=False)
        for epoch in epochs:
            try:
                # Set lr according to schedule
                if epoch in self.lr_schedule.keys():
                    self.opt.param_groups[0]["lr"] = self.lr_schedule[epoch]

                # Training
                self.model.train()
                train_epoch_loss, other_losses_train = self.run_epoch(self.train_dataloader, device)

                # Validation
                # if epoch % 10 == 0:
                self.model.eval()
                val_epoch_loss, other_losses_val = self.run_epoch(self.val_dataloader, device)
                # if epoch % 10 == 0:
                for metric_name, metric_value in other_losses_val.items():
                    writer.add_scalar(f"val {metric_name}", metric_value, epoch)
                for metric_name, metric_value in other_losses_train.items():
                        writer.add_scalar(f"train {metric_name}", metric_value, epoch)

                # Logging
                writer.add_scalar("train_loss", train_epoch_loss, epoch)
                writer.add_scalar("val_loss", val_epoch_loss, epoch)
                writer.add_scalar("learning_rate", self.opt.param_groups[0]["lr"], epoch)
                epochs.set_postfix_str(f"train loss: {train_epoch_loss:.2e}, val loss: {val_epoch_loss:.2e}, lr: {self.opt.param_groups[0]['lr']:.1e}")
                
                # Keep best model
                if self.best_model_params is None or val_epoch_loss < self.best_model_params["loss"]:
                    self.best_model_params = {
                        "epoch": epoch,
                        "loss": val_epoch_loss,
                        "train loss": train_epoch_loss,
                        "val RMSE": val_epoch_loss**0.5, # TODO only true if loss_func == MSELoss()
                        "train RMSE": train_epoch_loss**0.5, #TODO only true if loss_func == MSELoss()
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.opt.state_dict(),
                        "parameters": self.model.parameters(),
                        "training time in sec": (time.perf_counter() - start_time),
                    }

                    if False:
                        self.model.save(args["destination"], model_name=f"best_model_e{epoch}.pt")
                        with open(Path.cwd() / "runs" / args["destination"] / f"best_model_e{epoch}.yaml", "w") as f:
                            f.write(f"epoch: {self.best_model_params['epoch']}\n")
                            f.write(f"val_epoch_loss: {self.best_model_params['loss']}\n")
                            f.write(f"train_epoch_loss: {self.best_model_params['train loss']}\n")
                            f.write(f"training time in sec: {self.best_model_params['training time in sec']}\n")

                
                trial.report(val_epoch_loss, epoch)

                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                self.lr_scheduler.step(val_epoch_loss)

            except KeyboardInterrupt:
                try:
                    new_lr = float(input("\nNew learning rate: "))
                except ValueError as e:
                    print(e)
                else:
                    for g in self.opt.param_groups:
                        g["lr"] = new_lr
                    self.lr_schedule[epoch] = self.opt.param_groups[0]["lr"]

        # Apply best model params to model
        self.model.load_state_dict(self.best_model_params["state_dict"]) #self.model = 
        self.opt.load_state_dict(self.best_model_params["optimizer"]) #self.opt =
        print(f"Best model was found in epoch {self.best_model_params['epoch']}.")

        return val_epoch_loss

    def run_epoch(self, dataloader: DataLoader, device: str):
        epoch_loss = 0.0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            if self.model.training:
                self.opt.zero_grad()

            y_pred = self.model(x)
            required_size = y_pred.shape[2:]
            
            #3d
            start_pos = ((y.shape[2] - required_size[0])//2, (y.shape[3] - required_size[1])//2, (y.shape[4] - required_size[2])//2)
            y_reduced = y[:, :, start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1], start_pos[2]:start_pos[2]+required_size[2]]
            
            #start_pos = ((y.shape[2] - required_size[0])//2, (y.shape[3] - required_size[1])//2)
            #y_reduced = y[:, :, start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]]

            loss = self.loss_func(y_pred, y_reduced)

            if self.model.training:
                loss.backward()
                self.opt.step()

            epoch_loss += loss.detach().item()
        epoch_loss /= len(dataloader)

        # Calculate metrics
        metric_values = {}
        for metric_name, metric in self.metrics.items():
            metric_values[metric_name] = metric(y_pred, y_reduced).detach().item()
            
        return epoch_loss, metric_values

    def save_lr_schedule(self, path: str):
        """ save learning rate history to csv file"""
        with open(path, "w") as f:
            logging.info(f"Saving lr-schedule to {path}.")
            for epoch, lr in self.lr_schedule.items():
                f.write(f"{epoch},{lr}\n")

    def load_lr_schedule(self, path: pathlib.Path, case_2hp:bool=False):
        """ read lr-schedule from csv file"""
        # check if path contains lr-schedule, else use default one
        if not path.exists():
            logging.warning(f"Could not find lr-schedule at {path}. Using default lr-schedule instead.")
            path = pathlib.Path.cwd() / "processing" / "lr_schedules"
            lr_schedule_file = "default_lr_schedule.csv" if not case_2hp else "default_lr_schedule_2hp.csv"
            path = path / lr_schedule_file

        with open(path, "r") as f:
            for line in f:
                epoch, lr = line.split(",")
                self.lr_schedule[int(epoch)] = float(lr)

    def save_metrics(self, destination: pathlib.Path, no_params:int, max_epochs:int, training_time:float, device: str = "cpu"):
        csv_file = open(destination.parent / "measurements_all_metrics.csv", "a")
        # no. parameters, max. epochs, training time
        csv_writer = csv.writer(csv_file)

        self.model.eval()
        train_epoch_loss, other_losses_train = self.run_epoch(self.train_dataloader, device)
        val_epoch_loss, other_losses_val = self.run_epoch(self.val_dataloader, device)

        other_losses_train_list = []
        for _, val in other_losses_train.items():
            other_losses_train_list.append(val)
        other_losses_val_list = []
        for _, val in other_losses_val.items():
            other_losses_val_list.append(val)

        print(destination.name, self.best_model_params["epoch"], train_epoch_loss, val_epoch_loss)
        row = [destination.name, self.best_model_params["epoch"], train_epoch_loss, val_epoch_loss]
        row.extend(other_losses_train_list)
        row.extend(other_losses_val_list)
        row.extend([no_params, max_epochs, training_time])
               
        csv_writer.writerow(row)
        csv_file.close()