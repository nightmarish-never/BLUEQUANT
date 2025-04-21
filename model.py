import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, dropout_rate=0.3, output_size=5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)  # 输出5个预测值
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # 取最后时间步的输出送入全连接层


class Trainer:
    def __init__(self, model, train_loader, val_loader=None, epochs=50, lr=1e-3, device=None, 
                 save_path="best_model.pth", patience=5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.save_path = save_path
        self.patience = patience
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_model_state = None

    def train(self):
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch()
            val_loss = self._evaluate_epoch() if self.val_loader else None

            self._early_stopping(val_loss, epoch)
            self._log_epoch(epoch, train_loss, val_loss)

            if self.epochs_no_improve >= self.patience:
                print(f"⏹️ Early stopping triggered at epoch {epoch}.")
                break

        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        return self.model

    def _train_epoch(self):
        self.model.train()
        running_loss = 0.0

        for X_batch, y_batch in tqdm(self.train_loader, desc="Training Batches", leave=False):
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(X_batch)
            loss = self.criterion(output, y_batch)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        return running_loss / len(self.train_loader.dataset)

    def _evaluate_epoch(self):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in tqdm(self.val_loader, desc="Validation Batches", leave=False):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)

                running_loss += loss.item() * X_batch.size(0)

        return running_loss / len(self.val_loader.dataset)

    def _early_stopping(self, val_loss, epoch):
        if val_loss and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model_state = self.model.state_dict()
            self.epochs_no_improve = 0
            torch.save(self.best_model_state, self.save_path)
            status = "✅ Saved"
        else:
            self.epochs_no_improve += 1
            status = "⚠️ No Improvement"

    def _log_epoch(self, epoch, train_loss, val_loss):
        if val_loss:
            print(f"[Epoch {epoch:02d}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        else:
            print(f"[Epoch {epoch:02d}] Train Loss: {train_loss:.6f}")
