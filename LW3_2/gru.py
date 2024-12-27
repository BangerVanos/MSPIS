# Лабораторная работа 3 по дисциплине МРЗвИС
# Выполнена студентом группы 121702
# БГУИР Заломов Роман Андреевич
#
# Вариант 15: Реализовать модель рекуррентной сети с цепью нейросетевых моделей управляемых рекуррентных блоков 
# с логарифмической функцией активации (гиперболический арксинус) выходного сигнала на скрытом слое
#
# 24.12.24
# Данный файл содержит реализацию сети GRU
# с функцией активации arcsinh на скрытом слое
# Часть исходного кода была взята с: https://pastebin.com/UniAESGy


import numpy as np


def create_sliding_window_batches(sequence, window_size, batch_size, output_size):
    X, y = [], []
    
    # Формирование данных скользящего окна
    for i in range(len(sequence) - window_size - output_size + 1):
        X.append(sequence[i:i + window_size])
        y.append(sequence[i + window_size:i + window_size + output_size])
    
    # Преобразование в массивы numpy
    X, y = np.array(X), np.array(y)

    # Определение количества батчей
    total_samples = len(X)
    batch_amount = int(np.ceil(total_samples / batch_size))

    # Дополнение последнего батча, если данных не хватает
    if total_samples % batch_size != 0:
        pad_size = batch_size - (total_samples % batch_size)
        
        X_pad = np.repeat(X[-1][np.newaxis, :], pad_size, axis=0)
        y_pad = np.repeat(y[-1][np.newaxis, :], pad_size, axis=0)
        
        X = np.vstack((X, X_pad))
        y = np.vstack((y, y_pad))

    # Формирование батчей
    X_batches = X.reshape(batch_amount, batch_size, window_size)
    y_batches = y.reshape(batch_amount, batch_size, output_size)

    return X_batches, y_batches


def arcsinh(x):
    return np.arcsinh(x)


def darcsinh(x):
    # Производная arcsinh(x) = 1 / sqrt(x^2 + 1)
    return 1.0 / np.sqrt(x**2 + 1.0)


def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)


def mse_grad(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_pred.size


def mape(y_true, y_pred) -> float:    
    return np.mean(np.absolute((y_true - y_pred) / y_true))


class GRUCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        limit = np.sqrt(1.0 / hidden_size)
        
        # Инициализация параметров
        self.W_z = np.random.uniform(-limit, limit, (input_size, hidden_size))
        self.U_z = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_z = np.zeros(hidden_size)
        
        self.W_r = np.random.uniform(-limit, limit, (input_size, hidden_size))
        self.U_r = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_r = np.zeros(hidden_size)
        
        self.W_h = np.random.uniform(-limit, limit, (input_size, hidden_size))
        self.U_h = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_h = np.zeros(hidden_size)
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, x_t, h_prev):
        """
        Возвращает h_t и словарь cache для обратного прохода.
        """
        # Рассчёт гейтов
        z_t_in = x_t @ self.W_z + h_prev @ self.U_z + self.b_z
        z_t = self.sigmoid(z_t_in)
        
        r_t_in = x_t @ self.W_r + h_prev @ self.U_r + self.b_r
        r_t = self.sigmoid(r_t_in)
        
        h_hat_t_in = x_t @ self.W_h + (r_t * h_prev) @ self.U_h + self.b_h
        h_hat_t = arcsinh(h_hat_t_in)
        
        h_t = (1 - z_t)*h_prev + z_t*h_hat_t
        
        cache = {
            'x_t': x_t, 'h_prev': h_prev,
            'z_t': z_t, 'z_t_in': z_t_in,
            'r_t': r_t, 'r_t_in': r_t_in,
            'h_hat_t_in': h_hat_t_in, 'h_hat_t': h_hat_t,
        }
        return h_t, cache

    def backward(self, dh_t, cache):
        """
        dh_t: градиент по h_t (след. шага или по функции потерь)
        Возвращает градиенты по параметрам и dh_prev, а также dx_t
        """
        x_t = cache['x_t']
        h_prev = cache['h_prev']
        z_t = cache['z_t']
        z_t_in = cache['z_t_in']
        r_t = cache['r_t']
        r_t_in = cache['r_t_in']
        h_hat_t_in = cache['h_hat_t_in']
        h_hat_t = cache['h_hat_t']

        # dh_t по h_t:
        # h_t = (1 - z_t)*h_prev + z_t*h_hat_t
        # dh_prev_contrib = dh_t * (1 - z_t)
        # dz_t = dh_t * (h_hat_t - h_prev)
        # dh_hat_t = dh_t * z_t
        
        dh_hat_t = dh_t * z_t
        dz_t = dh_t * (h_hat_t - h_prev)
        dh_prev = dh_t * (1 - z_t)

        # Производные через arcsinh:
        # h_hat_t = arcsinh(h_hat_t_in)
        # dh_hat_t_in = dh_hat_t * darcsinh(h_hat_t_in)
        dh_hat_t_in = dh_hat_t * darcsinh(h_hat_t_in)
        
        # r_t = sigmoid(r_t_in)
        # для зависимостей внутри h_hat_t_in:
        # h_hat_t_in = x_t W_h + (r_t * h_prev) U_h + b_h
        
        # d(r_t * h_prev) = (r_t * h_prev) @ U_h
        # Но нам нужна производная по r_t и h_prev.
        # dh_hat_t_in wrt r_t: (h_prev @ U_h)
        # dh_hat_t_in wrt h_prev (через r_t): (r_t @ U_h^T)
        
        # Сначала разложим градиенты по параметрам:
        dW_h = x_t.T @ dh_hat_t_in
        dU_h = (r_t * h_prev).T @ dh_hat_t_in
        db_h = np.sum(dh_hat_t_in, axis=0)
        
        # Производим обратный проход по r_t:
        # h_hat_t_in зависит от r_t: dh_hat_t_in/dr_t = (h_prev @ U_h)
        # dr_t_in = d(r_t)/dr_t_in * ...
        dr_t = (dh_hat_t_in @ self.U_h.T) * h_prev
        # r_t = sigmoid(r_t_in) => dr_t_in = dr_t * r_t*(1-r_t)
        dr_t_in = dr_t * r_t * (1 - r_t)
        
        # Производим обратный проход по h_prev из h_hat_t_in:
        dh_prev += (dh_hat_t_in @ self.U_h.T) * r_t

        # z_t = sigmoid(z_t_in)
        # dz_t_in = dz_t * z_t*(1-z_t)
        dz_t_in = dz_t * z_t * (1 - z_t)

        # Теперь разберем x_t и h_prev воздействия для z_t и r_t:
        dW_z = x_t.T @ dz_t_in
        dU_z = h_prev.T @ dz_t_in
        db_z = np.sum(dz_t_in, axis=0)
        
        dW_r = x_t.T @ dr_t_in
        dU_r = h_prev.T @ dr_t_in
        db_r = np.sum(dr_t_in, axis=0)

        # Теперь учесть влияние z_t и r_t на h_prev, x_t:
        # Часть dh_prev уже учтена:
        # h_t зависит от h_prev через (1 - z_t)*h_prev => dh_prev += dh_t*(1-z_t)
        # У нас уже это учтено выше.

        # h_prev также влияет через z_t_in и r_t_in:
        dh_prev += (dz_t_in @ self.U_z.T)
        dh_prev += (dr_t_in @ self.U_r.T)

        # h_prev влияет также через h_hat_t_in (уже учтено выше)
        
        # Для x_t:
        dx_t = (dz_t_in @ self.W_z.T) + (dr_t_in @ self.W_r.T) + (dh_hat_t_in @ self.W_h.T)

        return dx_t, dh_prev, (dW_z, dU_z, db_z, dW_r, dU_r, db_r, dW_h, dU_h, db_h)


class GRUModel:
    def __init__(self, input_size, hidden_size, output_size):
        self.cell = GRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Выходной слой: h_T -> y
        limit = np.sqrt(1.0 / hidden_size)
        self.W_out = np.random.uniform(-limit, limit, (hidden_size, output_size))
        self.b_out = np.zeros(output_size)
        
    def forward(self, X):
        """
        X: (seq_length, batch_size, input_size)
        Возвращает предсказание y_pred для каждого временного шага и кэш для обратного прохода.
        y_pred теперь рассчитывается для всех временных шагов.
        """
        seq_length, batch_size, _ = X.shape
        h = np.zeros((seq_length, batch_size, self.hidden_size))  # Все скрытые состояния
        y_pred = np.zeros((seq_length, batch_size, self.output_size))  # Предсказания для каждого шага
        h_prev = np.zeros((batch_size, self.hidden_size))  # Инициализация предыдущего состояния
        caches = []
        
        for t in range(seq_length):
            h_t, cache_t = self.cell.forward(X[t], h_prev)  # Вычисляем скрытое состояние
            h[t] = h_t
            h_prev = h_t
            caches.append(cache_t)
            
            # Вычисляем предсказание на текущем временном шаге:
            y_pred[t] = h_t @ self.W_out + self.b_out

        return y_pred, h, caches

    
    def backward(self, dy_pred, h, caches, X):
        """
        Выполняем обратный проход по всей последовательности (BPTT).
        dy_pred: (seq_length, batch_size, output_size) - градиент по выходу на каждом шаге.
        """
        seq_length, batch_size, _ = X.shape

        # Инициализация градиентов
        dW_z = np.zeros_like(self.cell.W_z)
        dU_z = np.zeros_like(self.cell.U_z)
        db_z = np.zeros_like(self.cell.b_z)
        dW_r = np.zeros_like(self.cell.W_r)
        dU_r = np.zeros_like(self.cell.U_r)
        db_r = np.zeros_like(self.cell.b_r)
        dW_h = np.zeros_like(self.cell.W_h)
        dU_h = np.zeros_like(self.cell.U_h)
        db_h = np.zeros_like(self.cell.b_h)

        dW_out = np.zeros_like(self.W_out)
        db_out = np.zeros_like(self.b_out)

        dh_next = np.zeros((batch_size, self.hidden_size))  # Градиент скрытого состояния для следующего шага

        # Градиенты по выходу (y_pred = h @ W_out + b_out)
        for t in reversed(range(seq_length)):
            dW_out += h[t].T @ dy_pred[t]
            db_out += np.sum(dy_pred[t], axis=0)

            # Градиент скрытого состояния через выход
            dh = dy_pred[t] @ self.W_out.T + dh_next

            # Обратный проход через GRU для текущего временного шага
            dx_t, dh_next, grads_cell = self.cell.backward(dh, caches[t])
            gW_z, gU_z, gb_z, gW_r, gU_r, gb_r, gW_h, gU_h, gb_h = grads_cell

            # Суммируем градиенты
            dW_z += gW_z
            dU_z += gU_z
            db_z += gb_z
            dW_r += gW_r
            dU_r += gU_r
            db_r += gb_r
            dW_h += gW_h
            dU_h += gU_h
            db_h += gb_h

        # Сборка всех градиентов в словарь
        grads = {
            'W_z': dW_z, 'U_z': dU_z, 'b_z': db_z,
            'W_r': dW_r, 'U_r': dU_r, 'b_r': db_r,
            'W_h': dW_h, 'U_h': dU_h, 'b_h': db_h,
            'W_out': dW_out, 'b_out': db_out
        }

        return grads


    def update_parameters(self, grads, lr=0.01):
        # Обновление параметров с помощью простого SGD
        self.cell.W_z -= lr * grads['W_z']
        self.cell.U_z -= lr * grads['U_z']
        self.cell.b_z -= lr * grads['b_z']
        
        self.cell.W_r -= lr * grads['W_r']
        self.cell.U_r -= lr * grads['U_r']
        self.cell.b_r -= lr * grads['b_r']
        
        self.cell.W_h -= lr * grads['W_h']
        self.cell.U_h -= lr * grads['U_h']
        self.cell.b_h -= lr * grads['b_h']
        
        self.W_out -= lr * grads['W_out']
        self.b_out -= lr * grads['b_out']
    
    def train(self, x, y, lr: float = 0.01, max_epochs: int = 10000,
          learn_by_loss: bool = False, max_loss: float = 0.01,
          verbosity: int = 1000):
        """
        Обучение модели по заданным данным.
        x: (seq_length, batch_size, input_size) - входные данные
        y: (seq_length, batch_size, output_size) - истинные значения
        """
        training_loss, training_mape = 0, 0

        for epoch in range(max_epochs):
            # Прямой проход
            y_pred, h, caches = self.forward(x)
            
            # Вычисление функции потерь по всем временным шагам
            loss = mse_loss(y_pred, y)  # y_pred и y имеют размерности (seq_length, batch_size, output_size)
            
            # Обратный проход
            dy_pred = mse_grad(y_pred, y)  # Градиент потерь
            grads = self.backward(dy_pred, h, caches, x)
            
            # Обновление параметров
            self.update_parameters(grads, lr)
            
            # Расчет метрики MAPE (по всем временным шагам)
            epoch_mape = mape(y, y_pred)
            
            if (epoch + 1) % verbosity == 0:
                print(f"Epoch {epoch+1}/{max_epochs}, Loss: {loss:.6f}\nMAPE: {epoch_mape:.6f}")
            
            # Условие остановки
            if learn_by_loss and loss <= max_loss:
                break

            training_loss = loss
            training_mape = epoch_mape

        # Итоговые результаты обучения
        print('TRAINING FINISHED')
        print(f"Epoch {epoch+1}/{max_epochs}, Loss: {training_loss:.6f}\nMAPE: {training_mape:.6f}")

        # Возврат результатов обучения
        return training_loss, training_mape


# Fibonacci sequence generator
def fibonacci_generator(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b


# Squared num sequence generator
def squared_generator(n):        
    for i in range(1, n + 1):
        yield i**2


# Arithmetic progression
def arithmetic_progression(n, a0, d):
    for i in range(n):
        yield a0 + i * d


# x = x0 / 2**i sequence generator
def half_generator(n, fst: float):
    num = fst
    for _ in range(n):
        yield num
        num /= 2


# 1/n sequence generator
def one_by_n_generator(n):    
    for i in range(n):
        yield 1 / (i + 1)


# 1, -1, 1, -1, 1,... sequence generator
def plus_one_minus_one_generator(n):    
    for i in range(n):        
        yield 1 if i % 2 == 0 else -1


# 1, 0, 1, 0, 0, 1, 0, 0, 0, 1,... sequence generator
def one_half_generator(n):
    count = 1
    generated = 0
    while generated < n:
        generated += 1
        yield 1
        for _ in range(count):
            if generated >= n:
                break
            generated += 1
            yield 0.5            
        count += 1
