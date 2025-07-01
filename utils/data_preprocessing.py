# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 11:36:24 2023

@author: Torres
"""

#%% imports

import matplotlib.pyplot as plt
import numpy as np
import torch 
import os

#%% Scaling methods

class Norm_z_score():
    def __init__(self,data):
        self.data = data
        self.data_normalized = (self.data - self.data.mean(dim=0))/self.data.std(dim=0)
    def __call__(self):
        return self.data_normalized
     
class Norm_linear_scalling():
    def __init__(self, data, lower_value, upper_value):
        self.data = data
        self.lower_value = lower_value
        self.upper_value = upper_value
        self.min = torch.min(self.data)
        self.max = torch.max(self.data)
        self.data_normalized = ((self.upper_value-self.lower_value)*(self.data - self.min)/(self.max - self.min))+self.lower_value        
    def __call__(self):
        return self.data_normalized, self.min, self.max
    
    
class DeNorm_linear_scalling():
    def __init__(self, data, lower_value, upper_value, minimum, maximum):
        self.data = data
        self.lower = lower_value
        self.upper = upper_value
        self.min = minimum
        self.max = maximum
        self.data_normalized = ((self.max-self.min)*(self.data - self.lower)/(self.upper - self.lower))+self.min        
    def __call__(self):
        return self.data_normalized

class Norm_log_scalling():
    def __init__(self,data):
        self.data = data
        self.data_normalized = torch.log(self.data)
    def __call__(self):
        return self.data_normalized

#%% Normalization

class Normalization():
    def __init__(self,data, save_path="", normalize="Z score",low_value_normalization=0, upper_value_normalization=1,plot_normalized_dataset = True):
        self.DATA=data
        self.normalize = normalize
        self.low_value_normalization = low_value_normalization
        self.upper_value_normalization = upper_value_normalization
        self.plot_normalized_dataset = plot_normalized_dataset
        self.save_path = save_path
        self.time = torch.arange(0,self.DATA.shape[0],1)
        
        # Data normalization
        self.method()
        
    
    def method(self):
        
        self.dataset= torch.tensor(())
        for i in range(self.DATA.shape[1]):
            sensor = self.DATA[:,i]
            
            if self.normalize == "Z score":
                dataset_z_score_norm = Norm_z_score(sensor)
                series = dataset_z_score_norm.data_normalized
                self.plot_normalization(self.time, series, title="Normalized Z Score - sensor_"+str(i), run=i, plotting=self.plot_normalized_dataset)  
            
            if self.normalize == "linear scalling":
                dataset_linear_scalling_norm = Norm_linear_scalling(sensor,self.low_value_normalization,self.upper_value_normalization)
                series = dataset_linear_scalling_norm.data_normalized
                self.plot_normalization(self.time, series, title="Normalized linear scalling sensor_"+str(i), run=i, plotting=self.plot_normalized_dataset)  
            
            if self.normalize == "log scalling":
                dataset_log_scalling_norm = Norm_log_scalling(sensor)
                series = dataset_log_scalling_norm.data_normalized
                self.plot_normalization(self.time, series, title="Normalized log scalling sensor_"+str(i), run=i, plotting=self.plot_normalized_dataset)  
            
            series=series.reshape(len(series),1)
            self.dataset=torch.cat((self.dataset,series),1)
            
        return self
   
    def __call__(self):
        return self.dataset

    def plot_normalization(self, time, series, title, run, plotting=False, format="-", start=0, end=None):
            
            plt.style.use("default")
            fig = plt.figure(figsize=(5, 3))
            plt.plot(time[start:end], series[start:end], format)
            plt.title(title)
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.grid(True)
            
            path=self.save_path
            Datapath=os.path.join(path,"02_Plots Dataset Generation"+"/"+ "run" +str(run))
            if not os.path.exists(Datapath):
                os.makedirs(Datapath)
            plt.savefig(Datapath+"/"+title+".jpg",bbox_inches="tight")
            
            if (plotting == True):
                plt.show()
            else:
                plt.close()
            
            return fig
  
#%% Windowing
            
class PreprocessingDataset(torch.utils.data.Dataset):
    def __init__(self, train, data,save=True,save_path="", splittingdataset = False, reshapedata=False, window_size=25, shift=25, selected_channels=(tuple(range(3)))):
       
        self.save = save
        self.save_path = save_path
        self.window_size=window_size
        self.shift=shift
        self.window_split = 0
        self.selected_channels=selected_channels
        self.sensors=data.shape[1]
        self.DATA=data
        self.LABEL=torch.zeros(0)
        
        
        #get train or test data
        if train:
            self.DATA= self.DATA[0:(int(self.DATA.shape[0]*0.8))]
        else:
            self.DATA= self.DATA[(int(self.DATA.shape[0]*0.8))::]
        
        self.moving_window()
        self.select_channels()
        if reshapedata:
            self.catall()
            
    def splitdata(self):   
        while (self.DATA.shape[0]%self.split!=0):
            self.split=self.split+1
    
        print("the dataset will be split in {} parts of {} rows".format(self.split,self.DATA.shape[0]/(self.split)))
        self.size_window=int(self.DATA.shape[0]/(self.split))
        self.split_completeDS=int(self.DATA.shape[0]/self.size_window)
        
        self.datasplit = torch.zeros(self.split_completeDS,self.DATA.shape[1],self.size_window) 
        
        for j in range (self.sensors): #number of channels for the columns
            k=0
            for i in range(0, self.DATA.shape[0], self.size_window):
                self.datasplit[k,j] = self.DATA[i:i+self.size_window,j]
                k=k+1
            
    def moving_window(self):
        
        self.window_split = int((self.DATA.shape[0]-self.window_size)/self.shift)+1
        self.datasplit = torch.zeros([self.window_split,self.sensors,self.window_size]) 
        
        k=0
        for i in range(self.window_split):
            self.datasplit[i]=self.DATA[k:k+self.window_size].permute(1,0)
            k=k+self.shift
            
    def select_channels(self):
        
        self.selected_dataset = torch.tensor([])#zeros(self.split_completeDS,len(self.selected_channels),self.size_window)
        for i in (self.selected_channels):
            self.selected_dataset=torch.cat((self.selected_dataset, self.datasplit[:,i:i+1]), dim=1)
   
    def catall(self):
        #selected_dataset = 110,2,96 
        #reshape selected_dataset to 1 channel: 220, 1, 96 
        self.selected_dataset = torch.reshape(self.selected_dataset, (self.selected_dataset.shape[0]*self.selected_dataset.shape[1],1,self.selected_dataset.shape[2]))
        
    def __len__(self):
        return self.selected_dataset.shape[0]
    def __getitem__(self, idx):
        return self.selected_dataset[idx]
