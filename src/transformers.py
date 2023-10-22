import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


class Debbuger:
    @staticmethod
    def log(level, verbose, *values: object):
        if verbose >= level:
            print(values)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
    
    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class TemporalSerie(Dataset):
    def __init__(self, serie, window_length):
        self.serie = torch.FloatTensor(serie).squeeze()
        self.window_length = window_length
    def __len__(self):
        return self.serie.shape[0] - self.window_length
    def __getitem__(self, idx):
        return (self.serie[idx:idx+self.window_length], self.serie[idx+self.window_length])


class PositionEncoder(nn.Module):
    def __init__(self, d_modelo, max_len=5000):
        super(PositionEncoder, self).__init__()

        pe = torch.zeros(max_len, d_modelo)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_modelo, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_modelo))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class Transformer(nn.Module):
    def __init__(self, 
                 d_model, 
                 nhead, 
                 num_encoder_layers, 
                 num_decoder_layers, 
                 dim_feedforward=64, 
                 dropout=0.3):
        super().__init__()
        self.position_encoder = PositionEncoder(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=torch.tanh)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.position_encoder(x)
        x = self.transformer(x, x)
        x = self.fc(x.permute(1, 0, 2))
        return x[:, -1, :]
    

class TransformerModel():
    def __init__(self, 
                 d_model, 
                 nhead, 
                 num_encoder_layers, 
                 num_decoder_layers, 
                 dim_feedforward=64, 
                 dropout=0.3, 
                 loss_metric=nn.MSELoss()):
        self.model = Transformer(
            d_model, 
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout)
        self.loss_metric = loss_metric

    def __trainer_one_epoch(self, data, optimizer):
        self.model.train()
        losses = []
        for sequence, target in data:
            sequence = sequence.unsqueeze(-1)
            target = target.unsqueeze(-1)

            output = self.model(sequence)
            loss = self.loss_metric(output, target)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return sum(losses)/len(losses)

    def __validate(self, data):
        self.model.eval()
        losses = []
        for sequence, target in data:
            sequence = sequence.unsqueeze(-1)
            target = target.unsqueeze(-1)

            output = self.model(sequence)
            loss = self.loss_metric(output, target)
            losses.append(loss.item())
        
        return sum(losses)/len(losses)

    def fit(self,
            serie: TemporalSerie,
            epochs,
            learning_rate,
            batch_size=16,
            verbose=0):
        data = DataLoader(serie, batch_size=batch_size)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            for sequence, target in data:
                sequence = sequence.unsqueeze(-1)
                target = target.unsqueeze(-1)

                output = self.model(sequence)
                loss = self.loss_metric(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            Debbuger.log(1, verbose, f"Epoch: {epoch} | Loss: {loss.item()}")

    def fit_with_es(self,
                    train: TemporalSerie,
                    validation: TemporalSerie,
                    epochs,
                    es: EarlyStopper,
                    learning_rate=0.001,
                    batch_size=16,
                    verbose=0):
        data_train = DataLoader(train, batch_size=batch_size)
        data_validation = DataLoader(validation, batch_size=batch_size)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

        train_losses = []
        validation_losses = []

        for epoch in range(epochs):
            train_loss = self.__trainer_one_epoch(data_train, optimizer)
            validation_loss = self.__validate(data_validation)

            train_losses.append(train_loss)
            validation_losses.append(validation_loss)

            Debbuger.log(1, verbose, f'Epoch: {epoch} | Train loss: {train_loss} | Validation loss: {validation_loss}')

            if(es.early_stop(validation_loss)):
                break

        return {
            'train_loss': train_losses,
            'validation_loss': validation_losses
        }



    def predict(self, series: TemporalSerie, batch_size = 16):
        self.model.eval()
        predicts = []
        
        with torch.no_grad():
            for sequence, _ in DataLoader(series, batch_size=batch_size):
                sequence = sequence.unsqueeze(-1)
                output = self.model(sequence)
                predicts.append(output.squeeze(-1).numpy())

        return np.concatenate(predicts).reshape(-1, 1)


