import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import json
import os
import csv
import io
from csv import writer
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score




class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()



        self.encoder = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        

        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )


        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2)
        )


        
    def forward(self, x):

        enc1 = self.encoder(x)

        mid = self.middle(enc1)

        dec1 = self.decoder(mid)
        return dec1




def NormalizeData(my_data):
    md = np.asarray(my_data, dtype = 'double')


    Xr = (((md - np.min(md))) / (np.max(md)+.00025 - np.min(md)))     

    Xr = np.asarray(Xr, dtype = 'double')

    #Xr = 2*Xr - 1  # If we want the output to be normalized between -1 and 1
    return Xr






tensors = torch.tensor(X_np_ten_arry, dtype=torch.float32)



my_model=UNet()
my_model.load_state_dict(torch.load('/home/..../my_model.pth'))





for idx in range(len(Y_data)):
    tensorss = tensors[idx].view(1, 12,64,48)
    predicted_values = my_model(tensorss)

   
   


	


