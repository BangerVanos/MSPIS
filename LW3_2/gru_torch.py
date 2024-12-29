import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.0, learning_rate=0.001, num_epochs=20):
        super(GRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # GRU слой
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Полносвязный слой для предсказания выхода
        self.fc = nn.Linear(hidden_size, output_size)

        # Функция потерь и оптимизатор
        self.criterion = nn.MSELoss()
        self.optimizer = None

    def forward(self, x):
        # Ожидаем входной размер (batch_size, input_size)
        x = x.unsqueeze(1)  # Добавляем фиктивный размер для sequence_length        

        # Инициализация скрытых состояний (h0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Прямой проход через GRU        
        out, _ = self.gru(x, h0)
        
        # Используем только последнее скрытое состояние
        out = out[:, -1, :]
        
        # Прогон через полносвязный слой
        out = self.fc(out)        
        
        return out

    def train_model(self, dataloader, device, verbosity: int = 1,
                    use_adam: bool = True):
        self.to(device)

        # Инициализация оптимизатора
        if use_adam:
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            self.train()

            for x_batch, y_batch in dataloader:                
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                # Прямой проход
                outputs = self(x_batch)
                loss = self.criterion(outputs, y_batch)

                # Обратное распространение и обновление весов
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Печать информации об обучении
            if (epoch + 1) % verbosity == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')

        print("Обучение завершено!")

    def predict(self, dataloader, device):
        self.eval()
        predictions = []
        with torch.inference_mode():
            for x_batch, _ in dataloader:
                x_batch = x_batch.to(device)
                outputs = self(x_batch)
                predictions.append(outputs.cpu())
        return torch.cat(predictions, dim=0)


class SequenceDataset(Dataset):

    def __init__(self, x, y):
        super(SequenceDataset, self).__init__()

        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

        self.n_samples = self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples


def create_sliding_window(sequence, window_size, output_size):
    X, y = [], []
    
    # Формирование данных скользящего окна
    for i in range(len(sequence) - window_size - output_size + 1):
        X.append(sequence[i:i + window_size])
        y.append(sequence[i + window_size:i + window_size + output_size])
    
    # Преобразование в массивы numpy
    X, y = np.array(X), np.array(y)

    return X, y


# Arithmetic progression
def arithmetic_progression(n, a0, d):
    for i in range(n):
        yield a0 + i * d


def mape(y_true, y_pred, ignore_zero: bool = True,
         return_percents: bool = False) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)        
    if ignore_zero:
        # Avoiding devision by zero       
        mask = y_true != 0        
        y_true = y_true[mask]
        y_pred = y_pred[mask]      
    return (np.mean(np.absolute((y_true - y_pred) / y_true)) * 
            (100 if return_percents else 1))
