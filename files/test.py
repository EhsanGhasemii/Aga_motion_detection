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


# Tensor Dim : 12 X 64 X 48


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

    #Xr = 2*Xr - 1    #  If we want the output to be normalized between -1 and 1
    return Xr


X_data = []
Y_data = []



X_np_ten_arry = np.array(X_data)

Y_np_flat_arry = np.array(Y_data)






tensors = torch.tensor(X_np_ten_arry, dtype=torch.float32)

labels = torch.tensor( np.asarray(Y_np_flat_arry, dtype = 'double'), dtype=torch.float32)

dataset = TensorDataset(tensors, labels)



train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


model = UNet()



criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

      
# Training loop
num_epochs = 60
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        fet_vect = outputs.reshape(-1,outputs.size(dim = 1)*outputs.size(dim = 2)*outputs.size(dim = 3))
       
        t1 =torch.zeros((outputs.size(dim = 0),outputs.size(dim = 1)*outputs.size(dim = 2)*outputs.size(dim = 3)))
        
        targy = targets.view(-1, 1)
        t1[:,0:1] = targy
        loss = criterion(fet_vect, t1)  
        
        loss.backward()
        optimizer.step()
        


    model.eval()
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_outputs = model(val_inputs)
            val_fet_vect = val_outputs.reshape(-1,val_outputs.size(dim = 1)*val_outputs.size(dim = 2)*val_outputs.size(dim = 3))
        
            vt1 =torch.zeros((val_outputs.size(dim = 0),val_outputs.size(dim = 1)*val_outputs.size(dim = 2)*val_outputs.size(dim = 3)))
        
            vtargy = val_targets.view(-1, 1)
            vt1[:,0:1] = vtargy
            
            
            val_loss = criterion(val_fet_vect, vt1)

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    
    
 
 
torch.save(model.state_dict(),'/home/.../my_model.pth')


