# -*- coding: utf-8 -*-%
"""
Created on %(date)s

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import os
import yaml
import torch

from model.transformer_GRU_model import GRUModel, TransformerEncoder
# from dataset.dataloader_single import TimeSeriesDataset
from utils.data_storage import DataStorage
from utils.generate_folder import GenerateFolder
from utils.config_plots import configure_plt
from utils.plots import plot_losses
# from utils.plots import plot_thermo

# %% config

with open('config//config.yaml', encoding='utf-8') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)
device = config['general']['device']

# %% trials

configure_plt()

# %% create dataloader

# train_set = TimeSeriesDataset(path='./resources', window_length = 5, window_step = 1, mode = 'train')
# test_set = TimeSeriesDataset(path='./resources', window_length = 5, window_step = 1, mode = 'test')

train_set = torch.load('./resources/train_set.pt')
test_set = torch.load('./resources/test_set.pt')

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True, num_workers=0)

mean, std = train_set.mean[[7]], train_set.std[[7]]
mean_temp, std_temp = train_set.mean[[4]], train_set.std[[4]]


# %%

for i in range(10):

    # %%
    
    folder = GenerateFolder(GenerateAll=False)
    train_folder = folder.GenerateTrainFolder(generate=True)
    datafolder, imgfolder, netfolder, tablefolder = folder.GenerateDataFolder(generate=True, location=train_folder)
    
    # %% model
    
    # model = generate_sequence(**config['network']).cuda()
    
    transformer_encoder = TransformerEncoder(input_dim = 480, d_model = 480, nhead = 4, 
                                         num_encoder_layers = 2, dim_feedforward = 480)
    gru_model = GRUModel(input_dim = 480, hidden_dim = 480, output_dim = 192, num_layers = 2)
    model = torch.nn.Sequential(transformer_encoder, gru_model).cuda()
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma=0.1)
    
    storage = DataStorage(['Epochs', 'Batch', 'loss', 'testloss'], show=2, line=100, header=500, precision=5)
    
    # %% testing
    
    buffer = 0.
    for index, (data, label) in enumerate(test_loader):
        with torch.no_grad():
            data, label = data.cuda(), label.cuda()[...,0]
            prediction = model(data.permute(0,2,1))
            buffer +=  criterion(prediction, label)
    testloss = buffer / (index + 1)
    
    # %% trainign 
    
    batch = 0
    testloss = torch.tensor([0.])
    for epoch in range(20):
        
        for _, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            
            data, label = data.cuda(), label.cuda()[...,0]
            
            prediction = model(data.permute(0,2,1))
            
            loss = criterion(prediction, label)
            
            loss.backward()
            optimizer.step()
            
            batch += 1
            
            storage.Store([epoch, batch, loss.item(), testloss.item()])
            
            # if batch % 50 == 0:
            #     for i in range(16):
            #         prediction_ = prediction[i].cpu().detach() * std + mean
            #         target_ = label[i].cpu().detach() * std + mean
            #         fig = plot_thermo(prediction_, target_)
            
            if batch % 100 == 0:
                buffer = 0.
                for index, (data, label) in enumerate(test_loader):
                    with torch.no_grad():
                        data, label = data.cuda(), label.cuda()[...,0]
                        prediction = model(data.permute(0,2,1))
                        buffer +=  criterion(prediction, label)
                testloss = buffer / (index + 1)
                
        scheduler.step()
        
        
    # %% save data
    torch.save(model.state_dict(), os.path.join(folder.netfolder, 'model_weights.pth'))
    torch.save(model, os.path.join(folder.netfolder, 'model.pt'))
    torch.save(storage, os.path.join(folder.datafolder, 'train_storage.pt'))
    
    # %% plot
    
    fig = plot_losses(storage)
    fig.savefig(os.path.join(train_folder, 'loss_plot.png'), dpi = 300, bbox_inches='tight')
    
    # %%
    
    



