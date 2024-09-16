import torch
from torch.nn import MSELoss,L1Loss, modules
from torch.utils.data import DataLoader
import os
import time
import yaml
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt

from networks.unet import UNet
from networks.unet3d import UNet3d
from processing.solver import Solver
from data_stuff.utils import SettingsTraining

def measure_len_width_1K_isoline(data: Dict[str, "DataToVisualize"]):
    ''' 
    function (for paper23) to measure the length and width of the 1K-isoline;
    prints the values for usage in ipynb
    '''
    
    lengths = {}
    widths = {}
    T_gwf = 10.6

    _, axes = plt.subplots(4, 1, sharex=True)
    for index, key in enumerate(["t_true", "t_out"]):
        plt.sca(axes[index])
        datapoint = data[key]
        datapoint.data = torch.flip(datapoint.data, dims=[1])
        left_bound, right_bound = 1280, 0
        upper_bound, lower_bound = 0, 80
        if datapoint.data.max() > T_gwf + 1:
            levels = [T_gwf + 1] 
            CS = plt.contour(datapoint.data.T, levels=levels, cmap='Pastel1', extent=(0,1280,80,0))

            # calc maximum width and length of 1K-isoline
            for level in CS.allsegs:
                for seg in level:
                    right_bound = max(right_bound, seg[:,0].max())
                    left_bound = min(left_bound, seg[:,0].min())
                    upper_bound = max(upper_bound, seg[:,1].max())
                    lower_bound = min(lower_bound, seg[:,1].min())
        lengths[key] = max(right_bound - left_bound, 0)
        widths[key] = max(upper_bound - lower_bound, 0)
        print(f"lengths_{key[2:]}.append({lengths[key]})")
        print(f"widths_{key[2:]}.append({widths[key]})")
        print(f"max_temps_{key[2:]}.append({datapoint.data.max()})")
        plt.sca(axes[index+2])
    plt.close("all")
    return lengths, widths

def measure_loss(model: UNet|UNet3d, dataloader: DataLoader, device: str, loss_func: modules.loss._Loss = MSELoss()):

    try:
        norm = dataloader.dataset.norm
    except AttributeError:
        norm = dataloader.dataset.dataset.norm
    model.eval()
    mse_loss = 0.0
    mse_closs = 0.0
    mae_loss = 0.0
    mae_closs = 0.0

    for x, y in dataloader: # batchwise
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x).to(device)
        mse_loss += loss_func(y_pred, y).detach().item()
        mae_loss += torch.mean(torch.abs(y_pred - y)).detach().item()

        y = torch.swapaxes(y, 0, 1)
        y_pred = torch.swapaxes(y_pred, 0, 1)
        y = norm.reverse(y.detach().cpu(),"Labels")
        y_pred = norm.reverse(y_pred.detach().cpu(),"Labels")
        mse_closs += loss_func(y_pred, y).detach().item()
        mae_closs += torch.mean(torch.abs(y_pred - y)).detach().item()
        
    mse_loss /= len(dataloader)
    mse_closs /= len(dataloader)
    mae_loss /= len(dataloader)
    mae_closs /= len(dataloader)

    return {"mean squared error": mse_loss, "mean squared error in [°C^2]": mse_closs, 
            "mean absolute error": mae_loss, "mean absolute error in [°C]": mae_closs}

def measure_additional_losses(model: UNet|UNet3d, dataloader: DataLoader, summed_error_pic: torch.Tensor = None, settings: SettingsTraining = None):
    #based on: https://github.com/JuliaPelzer/Heat-Plume-Prediction/blob/main/postprocessing/measurements.py
    try:
        norm = dataloader.dataset.norm
    except AttributeError:
        norm = dataloader.dataset.dataset.norm
    model.eval()
    results = {}
    var = 0.0

    for x, y in dataloader: # batchwise
        x = x.to(settings.device).detach()
        y = y.to(settings.device).detach()
        y_pred = model(x).to(settings.device).detach()
        y = torch.swapaxes(y, 0, 1)
        y_pred = torch.swapaxes(y_pred, 0, 1)
        y = norm.reverse(y.cpu(),"Labels")
        y_pred = norm.reverse(y_pred.cpu(),"Labels")
        y = torch.swapaxes(y, 0, 1)
        y_pred = torch.swapaxes(y_pred, 0, 1)

        var += torch.mean(torch.pow(abs((y_pred - y) - summed_error_pic),2))
             
    var /= len(dataloader)
    std = torch.sqrt(var)

    results = {"variance in [°C]": var, "standard deviation in [°C]": std}

    return results

def measure_losses(model: UNet|UNet3d, dataloader: DataLoader, settings: SettingsTraining):
    '''
    function to measure losses: mse_loss, mae_closs, rmse_closs, r2_closs, pbt_threshold
    based on https://github.com/JuliaPelzer/Heat-Plume-Prediction/blob/allin1_preproc_v2/postprocessing/measurements.py
    '''
    pbt_threshold = 0.1 # [°C]
    
    device = settings.device
    if settings.problem == "allin1":
        norm = dataloader.dataset.norm
    elif settings.problem in ["1hp", "2stages", "3d"]:
        norm = dataloader.dataset.dataset.norm
    model.eval()
    results = {}

    mse_loss = torch.Tensor([0.0,] )
    mae_closs = torch.Tensor([0.0,] )
    rmse_closs = torch.Tensor([0.0,] )
    r2_closs = torch.Tensor([0.0,] )
    pbt_closs = torch.Tensor([0.0,] )

    for x, y in dataloader:
        x = x.to(device).detach() # B,C,X,Y,Z
        y = y.to(device).detach()
        y_pred = model(x).to(device).detach()

        # normed losses
        mse_loss += MSELoss(reduction="sum")(y_pred, y).item()

        # reverse norm -> values in original units and scales
        y = torch.swapaxes(y, 0, 1) # for norm, C needs to be first
        y_pred = torch.swapaxes(y_pred, 0, 1)
        y = norm.reverse(y.cpu(),"Labels")
        y_pred = norm.reverse(y_pred.cpu(),"Labels")
        y = torch.swapaxes(y, 0, 1)
        y_pred = torch.swapaxes(y_pred, 0, 1)

        # losses in Celsius
        mae_closs += L1Loss(reduction="sum")(y_pred, y).item()
        rmse_closs += MSELoss(reduction="sum")(y_pred, y).item()
        r2_closs -= (MSELoss()(y_pred, y) / MSELoss()(y,torch.mean(y)) * y_pred.shape[0]).item()
        
        # count all pixels where the difference is bigger than 0.1°C, then average over number datapoints and number of cells
        pbt_closs += (torch.sum(torch.abs(y_pred - y) > pbt_threshold)).item()

    # average over all datapoints
    no_datapoints = len(dataloader.dataset)
    if settings.problem=="3d":
        no_cells = y_pred.shape[2] * y_pred.shape[3]*y_pred.shape[4]
    else:
        no_cells = y_pred.shape[2] * y_pred.shape[3]

    print(f"Number of cells: {no_cells}")
    print(f"Number of datapoints: {no_datapoints}")
    
    mse_loss /= (no_datapoints * no_cells)
    mae_closs /= (no_datapoints * no_cells)
    rmse_closs /=  (no_datapoints * no_cells)
    rmse_closs = torch.sqrt(rmse_closs)
    r2_closs /= no_datapoints
    r2_closs += 1
    pbt_closs /= (no_datapoints * no_cells) 
    pbt_closs *= 100 # value close to 0% is good, close to 100% is bad

    results = {"MSE [-] ": "{:.2e}".format(mse_loss.item()), f"MAE [°C] ": "{:.2e}".format(mae_closs.item()), f"RMSE [°C] ": "{:.2e}".format(rmse_closs.item()), f"R2 based on °C ": "{:.2e}".format(r2_closs.item())} 
        
    results["PBT (percentage bigger than threshold 0.1C in [%])"] = pbt_closs.item()

    return results


def save_all_measurements(settings:SettingsTraining, len_dataset, times, solver:Solver=None, errors:Dict={}):
    with open(Path.cwd() / "runs" / settings.destination / f"measurements_{settings.case}.yaml", "w") as f:
        for key, value in times.items():
            f.write(f"{key}: {value}\n")
        f.write(f"timestamp of end: {time.ctime()}\n")
        f.write(f"duration of whole process excluding visualisation and measurements in seconds: {(times['time_end']-times['time_begin'])}\n")
        if settings.case in ["train", "finetune"]: 
            f.write(f"duration of initializations in seconds: {times['time_initializations']-times['time_begin']}\n")
            f.write(f"duration of training in seconds: {times['time_training']-times['time_initializations']}\n")
        f.write(f"input params: {settings.inputs}\n")
        f.write(f"dataset location: {settings.dataset_prep.parent}\n")
        f.write(f"dataset name: {settings.dataset_raw}\n")
        f.write(f"case: {settings.case}\n")
        f.write(f"number of test datapoints: {len_dataset}\n")
        f.write(f"name_destination_folder: {settings.destination}\n")
        if settings.case in ["train", "finetune"]:
            f.write(f"number epochs: {settings.epochs}\n")

        for key, value in errors.items():
            f.write(f"{key}: {value}\n")
            print(f"{key}: {value}")
        if settings.case in ["test", "finetune"]:
            f.write(f"path to pretrained model: {settings.model}\n")
        if settings.case in ["train", "finetune"]: 
            f.write(f"best model found after epoch: {solver.best_model_params['epoch']}\n")
            f.write(f"best model found with val loss: {solver.best_model_params['loss']}\n")
            f.write(f"best model found with train loss: {solver.best_model_params['train loss']}\n")
            f.write(f"best model found with val RMSE: {solver.best_model_params['val RMSE']}\n")
            f.write(f"best model found with train RMSE: {solver.best_model_params['train RMSE']}\n")
            f.write(f"best model found after training time in seconds: {solver.best_model_params['training time in sec']}\n")

    if settings.case in ["train", "finetune"]:  
        with open(Path.cwd() / "runs" / settings.destination.parent / f"measurements_all_runs.csv", "a") as f:
            #name, settings.epochs, epoch, val loss, train loss, val rmse, train rmse, train time
            f.write(f"{settings.destination.name},{settings.epochs},{solver.best_model_params['epoch']}, {round(solver.best_model_params['loss'],6)}, {round(solver.best_model_params['train loss'],6)}, {round(solver.best_model_params['val RMSE'],6)}, {round(solver.best_model_params['train RMSE'],6)}, {round(solver.best_model_params['training time in sec'],6)}\n")

    print(f"Measurements saved\n")