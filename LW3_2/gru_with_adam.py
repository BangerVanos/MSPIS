import numpy as np


def arcsinh(x):
    return np.arcsinh(x)

def darcsinh(x):
    # Производная arcsinh(x) = 1 / sqrt(x^2 + 1)
    return 1.0 / np.sqrt(x**2 + 1.0)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    # Производная arcsinh(x) = 1 / sqrt(x^2 + 1)
    return 1.0 - tanh(x) ** 2

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

def mse_grad(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_pred.size

def mape(y_true, y_pred, ignore_zero: bool = True) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)        
    if ignore_zero:
        # Avoiding devision by zero       
        mask = y_true != 0        
        y_true = y_true[mask]
        y_pred = y_pred[mask]      
    return np.mean(np.absolute((y_true - y_pred) / y_true))


class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}  # Моменты первого порядка (средние градиентов)
        self.v = {}  # Моменты второго порядка (средние квадратов градиентов)

    def update(self, param_name, param, grad):
        """
        Обновление параметра с использованием Adam.
        param_name: имя параметра (строка), чтобы различать моменты.
        param: текущий параметр (матрица/вектор).
        grad: градиент параметра (той же размерности).
        """
        if param_name not in self.m:
            # Инициализация моментов для нового параметра
            self.m[param_name] = np.zeros_like(param)
            self.v[param_name] = np.zeros_like(param)

        self.t += 1  # Увеличение счётчика итераций

        # Обновление моментов
        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)

        # Коррекция смещения
        m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
        v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)

        # Обновление параметра
        param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return param
    

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


class GRUAdam:
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


    def update_parameters(self, grads, optimizer):
        """
        Обновление параметров с использованием AdamOptimizer.
        grads: словарь с градиентами.
        optimizer: объект AdamOptimizer.
        """
        # Обновление параметров
        self.cell.W_z = optimizer.update("W_z", self.cell.W_z, grads['W_z'])
        self.cell.U_z = optimizer.update("U_z", self.cell.U_z, grads['U_z'])
        self.cell.b_z = optimizer.update("b_z", self.cell.b_z, grads['b_z'])

        self.cell.W_r = optimizer.update("W_r", self.cell.W_r, grads['W_r'])
        self.cell.U_r = optimizer.update("U_r", self.cell.U_r, grads['U_r'])
        self.cell.b_r = optimizer.update("b_r", self.cell.b_r, grads['b_r'])

        self.cell.W_h = optimizer.update("W_h", self.cell.W_h, grads['W_h'])
        self.cell.U_h = optimizer.update("U_h", self.cell.U_h, grads['U_h'])
        self.cell.b_h = optimizer.update("b_h", self.cell.b_h, grads['b_h'])

        self.W_out = optimizer.update("W_out", self.W_out, grads['W_out'])
        self.b_out = optimizer.update("b_out", self.b_out, grads['b_out'])

    
    def train(self, x, y, lr: float = 0.01, max_epochs: int = 10000,
          learn_by_loss: bool = False, max_loss: float = 0.01,
          verbosity: int = 1000):
        """
        Обучение модели по заданным данным.
        x: (seq_length, batch_size, input_size) - входные данные
        y: (seq_length, batch_size, output_size) - истинные значения
        """
        training_loss, training_mape = 0, 0
        optimizer = AdamOptimizer(learning_rate=lr)

        for epoch in range(max_epochs):
            # Прямой проход
            y_pred, h, caches = self.forward(x)
            
            # Вычисление функции потерь по всем временным шагам
            loss = mse_loss(y_pred, y)  # y_pred и y имеют размерности (seq_length, batch_size, output_size)
            
            # Обратный проход
            dy_pred = mse_grad(y_pred, y)  # Градиент потерь
            grads = self.backward(dy_pred, h, caches, x)
            
            # Обновление параметров            
            self.update_parameters(grads, optimizer)
            
            # Расчет метрики MAPE (по всем временным шагам)            
            epoch_mape = mape(y, y_pred)
            
            if (epoch + 1) % verbosity == 0:
                print(f"Epoch {epoch+1}/{max_epochs}, Loss: {loss:.6f}\nMAPE: {epoch_mape * 100:.6f}%")
            
            # Условие остановки
            if learn_by_loss and loss <= max_loss:
                break
        
        y_pred, _, _ = self.forward(x)        
        training_loss = mse_loss(y_pred, y)
        training_mape = mape(y, y_pred)

        # Итоговые результаты обучения
        print('TRAINING FINISHED')
        print(f"Epoch {epoch+1}/{max_epochs}, Loss: {training_loss:.6f}\nMAPE: {training_mape * 100:.6f}%")

        # Возврат результатов обучения
        return training_loss, training_mape
