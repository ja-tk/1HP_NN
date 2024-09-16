import time
from dataclasses import dataclass, field
from math import inf
from typing import Dict

# import matplotlib as mpl
# mpl.use('pgf')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import DataLoader

from data_stuff.transforms import NormalizeTransform
from networks.unet import UNet
from networks.unet3d import UNet3d

# mpl.rcParams.update({'figure.max_open_warning': 0})
# plt.rcParams['figure.figsize'] = [16, 5]

# TODO: look at vispy library for plotting 3D data

@dataclass
class DataToVisualize:
    data: np.ndarray
    name: str
    extent_highs :tuple = (1280,100) # x,y in meters
    imshowargs: Dict = field(default_factory=dict)
    contourfargs: Dict = field(default_factory=dict)
    contourargs: Dict = field(default_factory=dict)

    def __post_init__(self):
        extent = (0,int(self.extent_highs[0]),int(self.extent_highs[1]),0)

        self.imshowargs = {"cmap": "RdBu_r", 
                           "extent": extent}

        self.contourfargs = {"levels": np.arange(10.4, 16, 0.25), 
                             "cmap": "RdBu_r", 
                             "extent": extent}
        
        T_gwf = 10.6
        T_inj_diff = 5.0
        self.contourargs = {"levels" : [np.round(T_gwf + 1, 1)],
                            "cmap" : "Pastel1", 
                            "extent": extent}

        if self.name == "Liquid Pressure [Pa]":
            self.name = "Pressure in [Pa]"
        elif self.name == "Material ID":
            self.name = "Position of the heatpump in [-]"
        elif self.name == "Permeability X [m^2]":
            self.name = "Permeability in [m$^2$]"
        elif self.name == "SDF":
            self.name = "SDF-transformed position in [-]"
    
def visualizations(model: UNet | UNet3d, dataloader: DataLoader, device: str, amount_datapoints_to_visu: int = inf, plot_path: str = "default", pic_format: str = "png"):
    print("Visualizing...", end="\r")

    if amount_datapoints_to_visu > len(dataloader.dataset):
        amount_datapoints_to_visu = len(dataloader.dataset)

    norm = dataloader.dataset.dataset.norm
    info = dataloader.dataset.dataset.info
    model.to(device)
    model.eval()
    settings_pic = {"format": pic_format,
                    "dpi": 600,}

    current_id = 0
    for inputs, labels in dataloader:
        len_batch = inputs.shape[0]
        for datapoint_id in range(len_batch):
            name_pic = f"{plot_path}_{current_id}"
            y_label="x [m]"

            x = torch.unsqueeze(inputs[datapoint_id].to(device), 0)
            y = labels[datapoint_id]
            y_out = model(x)

            x, y, y_out = reverse_norm_one_dp(x, y, y_out, norm)

            if y.dim()==3:
                # visualization for xy
                # 2d data comes in transposed (for xy) , 3d data not (look at ReduceTo2DTransform)
                dict_to_plot=prepare_data_to_plot(x.transpose(1,3), y.transpose(0,2), y_out.transpose(0,2), info) # order (z,y,x)
                name_pic = f"{plot_path}_{current_id}_xy"
                plot_datafields(dict_to_plot, name_pic, settings_pic, y_label)
                # visualization for zy
                name_pic = f"{plot_path}_{current_id}_zy"
                y_label="z [m]"

            dict_to_plot = prepare_data_to_plot(x, y, y_out, info)
            plot_datafields(dict_to_plot, name_pic, settings_pic, y_label)
            # plot_isolines(dict_to_plot, name_pic, settings_pic)
            # measure_len_width_1K_isoline(dict_to_plot)

            if current_id >= amount_datapoints_to_visu-1:
                return None
            current_id += 1

def reverse_norm_one_dp(x: torch.Tensor, y: torch.Tensor, y_out:torch.Tensor, norm: NormalizeTransform):
    # reverse transform for plotting real values
    x = norm.reverse(x.detach().cpu().squeeze(0), "Inputs")
    y = norm.reverse(y.detach().cpu(),"Labels")[0]#[0] wie squeeze(0): 1,64,256,64 -> 64,256,64
    y_out = norm.reverse(y_out.detach().cpu()[0],"Labels")[0]
    return x, y, y_out

def prepare_data_to_plot(x: torch.Tensor, y: torch.Tensor, y_out:torch.Tensor, info: dict):
    # prepare data of temperature true, temperature out, error, physical variables (inputs)
    temp_max = max(y.max(), y_out.max())
    temp_min = min(y.min(), y_out.min())

    if y.dim() ==3:
        position=get_hp_position_from_info(x, info)[0]

        y=remove_dimension(y, position)
        y_out=remove_dimension(y_out, position)
        x=remove_dimension(x, position)

    extent_highs = (np.array(info["CellsSize"][:2]) * x.shape[-2:])
    

    dict_to_plot = {
        "t_true": DataToVisualize(y, "Label: Temperature in [°C]", extent_highs, {"vmax": temp_max, "vmin": temp_min}),
        "t_out": DataToVisualize(y_out, "Prediction: Temperature in [°C]", extent_highs, {"vmax": temp_max, "vmin": temp_min}),
        "error": DataToVisualize(torch.abs(y-y_out), "Absolute error in [°C]", extent_highs),
    }
    inputs = info["Inputs"].keys()
    for input in inputs:
        index = info["Inputs"][input]["index"]
        dict_to_plot[input] = DataToVisualize(x[index], input, extent_highs)

    return dict_to_plot

def plot_datafields(data: Dict[str, DataToVisualize], name_pic: str, settings_pic: dict, y_label="x [m]"):
    # plot datafields (temperature true, temperature out, error, physical variables (inputs))

    num_subplots = len(data)
    fig, axes = plt.subplots(num_subplots, 1, sharex=True)
    fig.set_figheight(num_subplots)
    
    for index, (name, datapoint) in enumerate(data.items()):
        plt.sca(axes[index])
        plt.title(datapoint.name)
        # if name in ["t_true", "t_out"]:  
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("ignore")

        #         CS = plt.contour(torch.flip(datapoint.data, dims=[1]).T, **datapoint.contourargs)
        #     plt.clabel(CS, inline=1, fontsize=10)
        plt.imshow(datapoint.data.T, **datapoint.imshowargs)
        plt.gca().invert_yaxis()

        plt.ylabel(y_label)
        _aligned_colorbar()

    plt.sca(axes[-1])
    plt.xlabel("y [m]")
    plt.tight_layout()
    plt.savefig(f"{name_pic}.{settings_pic['format']}", **settings_pic)

def plot_isolines(data: Dict[str, DataToVisualize], name_pic: str, settings_pic: dict):
    # plot isolines of temperature fields
    num_subplots = 3 if "Original Temperature [C]" in data.keys() else 2
    fig, axes = plt.subplots(num_subplots, 1, sharex=True)
    fig.set_figheight(num_subplots)

    for index, name in enumerate(["t_true", "t_out", "Original Temperature [C]"]):
        try:
            plt.sca(axes[index])
            data[name].data = torch.flip(data[name].data, dims=[1])
            plt.title("Isolines of "+data[name].name)
            plt.contourf(data[name].data.T, **data[name].contourfargs)
            plt.ylabel("x [m]")
            _aligned_colorbar(ticks=[11.6, 15.6])
        except:
            pass

    plt.sca(axes[-1])
    plt.xlabel("y [m]")
    plt.tight_layout()
    plt.savefig(f"{name_pic}_isolines.{settings_pic['format']}", **settings_pic)

def infer_all_and_summed_pic(model: UNet | UNet3d, dataloader: DataLoader, device: str):
    '''
    sum inference time (including reverse-norming) and pixelwise error over all datapoints
    '''
    
    norm = dataloader.dataset.dataset.norm
    model.to(device)
    model.eval()

    current_id = 0
    avg_inference_time = 0
    summed_error_pic = torch.zeros_like(torch.Tensor(dataloader.dataset[0][0][0])).cpu()

    for inputs, labels in dataloader:
        len_batch = inputs.shape[0]
        for datapoint_id in range(len_batch):
            # get data
            x = inputs[datapoint_id].to(device)
            x = torch.unsqueeze(x, 0)

            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            y_out = model(x)
            end.record()
            torch.cuda.synchronize()
            avg_inference_time += start.elapsed_time(end)/1000
            
            y = labels[datapoint_id]

            # reverse transform for plotting real values
            x = norm.reverse(x.cpu().detach().squeeze(), "Inputs")
            y = norm.reverse(y.cpu().detach(),"Labels")[0]
            y_out = norm.reverse(y_out.cpu().detach()[0],"Labels")[0]
            summed_error_pic += abs(y-y_out)

            current_id += 1

    avg_inference_time /= current_id
    summed_error_pic /= current_id
    return avg_inference_time, summed_error_pic

def plot_avg_error_cellwise(dataloader, summed_error_pic, settings_pic: dict):
    # plot avg error cellwise AND return time measurements for inference

    info = dataloader.dataset.dataset.info
    extent_highs = (np.array(info["CellsSize"][:2]) * (dataloader.dataset[0][0][0].shape)[-2:])
    extent = (0,int(extent_highs[0]),int(extent_highs[1]),0)

    if summed_error_pic.dim()==3:
        #transpose for xy
        summed_error_pic.transpose(0,2)#z,y,x

        #slice in z-direction
        #position=get_hp_position(summed_error_pic)[0]
        #summed_error_pic=remove_dimension(summed_error_pic, position)#y,x
        
        #keep the highest values in z-direction
        summed_error_pic=torch.max(summed_error_pic,0)[0]

        #mean in z-direction
        #summed_error_pic_2d=torch.mean(summed_error_pic,0)


    plt.figure()
    plt.imshow(summed_error_pic.T, cmap="RdBu_r", extent=extent)
    plt.gca().invert_yaxis()
    plt.ylabel("x [m]")
    plt.xlabel("y [m]")
    plt.title("Cellwise averaged error [°C]")
    _aligned_colorbar()

    plt.tight_layout()
    plt.savefig(f"{settings_pic['folder']}/avg_error.{settings_pic['format']}", format=settings_pic['format'])

def _aligned_colorbar(*args, **kwargs):
    cax = make_axes_locatable(plt.gca()).append_axes(
        "right", size=0.3, pad=0.05)
    plt.colorbar(*args, cax=cax, **kwargs)

def get_hp_position(tensor: torch.tensor):#intended for summed_error_pic, as error is highest at hp position
    '''
    Returns the hp position by searching for the maximum tensor value.
    Only works for 3D input data (X,Y,Z)
    '''     
    assert tensor.dim()==3 , f"Tensor has wrong dimension: {tensor.dim()} instead of 3"
    
    loc_hp = np.array(np.where(tensor == tensor.max())).squeeze()
    return loc_hp

def get_hp_position_from_info(tensor: torch.tensor, info: dict):
    '''
    Returns the hp position by reading the corresponding channel index from info
    and then searching for the maximum tensor value in this channel.
    Only works for 4D input data with first entry "channels" (C,X,Y,Z)
    '''     
    assert tensor.dim()==4 , f"Tensor has wrong dimension: {tensor.dim()} instead of 4"
    try:
        idx = info["Inputs"]["Material ID"]["index"]
    except:
        idx = info["Inputs"]["SDF"]["index"]
    loc_hp = np.array(np.where(tensor[idx] == tensor[idx].max())).squeeze()
    
    return loc_hp

def remove_dimension(tensor: torch.tensor, position: int):
    '''
    Removes first dimension at given position.
    Works for 3D tensors (X,Y,Z) or 4D tensors with first entry "channels" (C,X,Y,Z)
    '''
    if tensor.dim()==3:
        assert position <= tensor.shape[0], "position is larger than data dimension 0"
        tensor=tensor[position,:,:]
        tensor.squeeze(0)
    else:
        assert tensor.dim()==4 , f"Tensor has wrong dimension: {tensor.dim()} instead of 4"
        assert position <= tensor.shape[1], "position is larger than data dimension 1"
        tensor=tensor[:,position,:,:]
        tensor.squeeze(1)
    return tensor

