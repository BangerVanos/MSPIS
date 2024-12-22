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
    return np.mean(np.abs((y_true - y_pred) / y_true))


class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        limit = np.sqrt(1.0 / hidden_size)
        
        # Параметры для входного гейта
        self.W_i = np.random.uniform(-limit, limit, (input_size, hidden_size))
        self.U_i = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_i = np.zeros(hidden_size)

        # Параметры для гейта забывания
        self.W_f = np.random.uniform(-limit, limit, (input_size, hidden_size))
        self.U_f = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_f = np.zeros(hidden_size)

        # Параметры для выходного гейта
        self.W_o = np.random.uniform(-limit, limit, (input_size, hidden_size))
        self.U_o = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_o = np.zeros(hidden_size)

        # Параметры для кандидата состояния
        self.W_g = np.random.uniform(-limit, limit, (input_size, hidden_size))
        self.U_g = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.b_g = np.zeros(hidden_size)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, x_t, h_prev, c_prev):
        """
        Выполняет прямой проход LSTM ячейки:
        i_t = sigma(W_i*x_t + U_i*h_{t-1} + b_i)
        f_t = sigma(W_f*x_t + U_f*h_{t-1} + b_f)
        o_t = sigma(W_o*x_t + U_o*h_{t-1} + b_o)
        g_t_in = W_g*x_t + U_g*h_{t-1} + b_g
        g_t = arcsinh(g_t_in)

        c_t = f_t*c_{t-1} + i_t*g_t
        h_t = o_t*arcsinh(c_t)
        """
        i_t_in = x_t @ self.W_i + h_prev @ self.U_i + self.b_i
        i_t = self.sigmoid(i_t_in)

        f_t_in = x_t @ self.W_f + h_prev @ self.U_f + self.b_f
        f_t = self.sigmoid(f_t_in)

        o_t_in = x_t @ self.W_o + h_prev @ self.U_o + self.b_o
        o_t = self.sigmoid(o_t_in)

        g_t_in = x_t @ self.W_g + h_prev @ self.U_g + self.b_g
        g_t = arcsinh(g_t_in)

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * arcsinh(c_t)

        cache = {
            'x_t': x_t, 'h_prev': h_prev, 'c_prev': c_prev,
            'i_t': i_t, 'i_t_in': i_t_in,
            'f_t': f_t, 'f_t_in': f_t_in,
            'o_t': o_t, 'o_t_in': o_t_in,
            'g_t': g_t, 'g_t_in': g_t_in,
            'c_t': c_t,
        }

        return h_t, c_t, cache

    def backward(self, dh_t, dc_t, cache):
        """
        Обратный проход для одного временного шага LSTM.

        dh_t: градиент по h_t
        dc_t: градиент по c_t
        """
        x_t = cache['x_t']
        h_prev = cache['h_prev']
        c_prev = cache['c_prev']
        i_t = cache['i_t']
        f_t = cache['f_t']
        o_t = cache['o_t']
        g_t = cache['g_t']
        g_t_in = cache['g_t_in']
        c_t = cache['c_t']

        # h_t = o_t * arcsinh(c_t)
        # dh_t/dc_t = o_t * d(arcsinh(c_t))/dc_t = o_t * 1/sqrt(1+c_t^2)
        # Для удобства:
        dc_t_total = dc_t + dh_t * o_t * darcsinh(c_t)

        # Разберем градиенты по гейтам:
        # c_t = f_t*c_prev + i_t*g_t
        # dc_t_total/di_t = g_t
        # dc_t_total/df_t = c_prev
        # dc_t_total/dg_t = i_t
        di_t = dc_t_total * g_t
        df_t = dc_t_total * c_prev
        dg_t = dc_t_total * i_t

        # g_t = arcsinh(g_t_in)
        # dg_t_in = dg_t * darcsinh(g_t_in)
        dg_t_in = dg_t * darcsinh(g_t_in)

        # i_t = sigmoid(i_t_in), f_t = sigmoid(f_t_in), o_t = sigmoid(o_t_in)
        # di_t_in = di_t * i_t*(1-i_t)
        di_t_in = di_t * i_t * (1 - i_t)
        # df_t_in = df_t * f_t*(1-f_t)
        df_t_in = df_t * f_t * (1 - f_t)

        # dh_t влияет на o_t: h_t = o_t*arcsinh(c_t)
        # do_t = dh_t * arcsinh(c_t)
        do_t = dh_t * arcsinh(c_t)
        do_t_in = do_t * o_t * (1 - o_t)

        # dc_t_total по c_prev:
        dc_prev = dc_t_total * f_t

        # Соберем градиенты по параметрам:
        # i-тензор:
        dW_i = x_t.T @ di_t_in
        dU_i = h_prev.T @ di_t_in
        db_i = np.sum(di_t_in, axis=0)

        # f-тензор:
        dW_f = x_t.T @ df_t_in
        dU_f = h_prev.T @ df_t_in
        db_f = np.sum(df_t_in, axis=0)

        # o-тензор:
        dW_o = x_t.T @ do_t_in
        dU_o = h_prev.T @ do_t_in
        db_o = np.sum(do_t_in, axis=0)

        # g-тензор:
        dW_g = x_t.T @ dg_t_in
        dU_g = h_prev.T @ dg_t_in
        db_g = np.sum(dg_t_in, axis=0)

        # dx_t:
        dx_t = (di_t_in @ self.W_i.T) + (df_t_in @ self.W_f.T) + (do_t_in @ self.W_o.T) + (dg_t_in @ self.W_g.T)

        # dh_prev:
        dh_prev = (di_t_in @ self.U_i.T) + (df_t_in @ self.U_f.T) + (do_t_in @ self.U_o.T) + (dg_t_in @ self.U_g.T)

        return dx_t, dh_prev, dc_prev, (dW_i, dU_i, db_i,
                                        dW_f, dU_f, db_f,
                                        dW_o, dU_o, db_o,
                                        dW_g, dU_g, db_g)


class LSTMModel:
    def __init__(self, input_size=10, hidden_size=20, output_size=3):
        self.cell = LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        limit = np.sqrt(1.0 / hidden_size)
        self.W_out = np.random.uniform(-limit, limit, (hidden_size, output_size))
        self.b_out = np.zeros(output_size)
        
    def forward(self, X):
        """
        X: (seq_length, batch_size, input_size)
        Возвращает:
        y_pred: (seq_length, batch_size, output_size)
        h: (seq_length, batch_size, hidden_size)
        c: (seq_length, batch_size, hidden_size)
        caches: список кэшей для каждого шага
        """
        seq_length, batch_size, _ = X.shape
        h = np.zeros((seq_length, batch_size, self.hidden_size))
        c = np.zeros((seq_length, batch_size, self.hidden_size))
        y_pred = np.zeros((seq_length, batch_size, self.output_size))

        h_prev = np.zeros((batch_size, self.hidden_size))
        c_prev = np.zeros((batch_size, self.hidden_size))
        caches = []

        for t in range(seq_length):
            h_t, c_t, cache_t = self.cell.forward(X[t], h_prev, c_prev)
            h[t] = h_t
            c[t] = c_t
            h_prev = h_t
            c_prev = c_t
            caches.append(cache_t)
            
            y_pred[t] = h_t @ self.W_out + self.b_out

        return y_pred, h, c, caches

    def backward(self, dy_pred, h, c, caches, X):
        """
        Обратный проход по всей последовательности (BPTT).
        dy_pred: (seq_length, batch_size, output_size) - градиент по y_pred
        """
        seq_length, batch_size, _ = X.shape

        dW_i = np.zeros_like(self.cell.W_i)
        dU_i = np.zeros_like(self.cell.U_i)
        db_i = np.zeros_like(self.cell.b_i)

        dW_f = np.zeros_like(self.cell.W_f)
        dU_f = np.zeros_like(self.cell.U_f)
        db_f = np.zeros_like(self.cell.b_f)

        dW_o = np.zeros_like(self.cell.W_o)
        dU_o = np.zeros_like(self.cell.U_o)
        db_o = np.zeros_like(self.cell.b_o)

        dW_g = np.zeros_like(self.cell.W_g)
        dU_g = np.zeros_like(self.cell.U_g)
        db_g = np.zeros_like(self.cell.b_g)

        dW_out = np.zeros_like(self.W_out)
        db_out = np.zeros_like(self.b_out)

        dh_next = np.zeros((batch_size, self.hidden_size))
        dc_next = np.zeros((batch_size, self.hidden_size))

        for t in reversed(range(seq_length)):
            # Градиент по выходному слою
            dW_out += h[t].T @ dy_pred[t]
            db_out += np.sum(dy_pred[t], axis=0)

            # dh от выхода:
            dh = dy_pred[t] @ self.W_out.T + dh_next
            dc = dc_next

            dx_t, dh_next, dc_next, grads_cell = self.cell.backward(dh, dc, caches[t])
            (gW_i, gU_i, gb_i,
             gW_f, gU_f, gb_f,
             gW_o, gU_o, gb_o,
             gW_g, gU_g, gb_g) = grads_cell

            dW_i += gW_i
            dU_i += gU_i
            db_i += gb_i

            dW_f += gW_f
            dU_f += gU_f
            db_f += gb_f

            dW_o += gW_o
            dU_o += gU_o
            db_o += gb_o

            dW_g += gW_g
            dU_g += gU_g
            db_g += gb_g

        grads = {
            'W_i': dW_i, 'U_i': dU_i, 'b_i': db_i,
            'W_f': dW_f, 'U_f': dU_f, 'b_f': db_f,
            'W_o': dW_o, 'U_o': dU_o, 'b_o': db_o,
            'W_g': dW_g, 'U_g': dU_g, 'b_g': db_g,
            'W_out': dW_out, 'b_out': db_out
        }

        return grads

    def update_parameters(self, grads, lr=0.01):
        self.cell.W_i -= lr * grads['W_i']
        self.cell.U_i -= lr * grads['U_i']
        self.cell.b_i -= lr * grads['b_i']

        self.cell.W_f -= lr * grads['W_f']
        self.cell.U_f -= lr * grads['U_f']
        self.cell.b_f -= lr * grads['b_f']

        self.cell.W_o -= lr * grads['W_o']
        self.cell.U_o -= lr * grads['U_o']
        self.cell.b_o -= lr * grads['b_o']

        self.cell.W_g -= lr * grads['W_g']
        self.cell.U_g -= lr * grads['U_g']
        self.cell.b_g -= lr * grads['b_g']

        self.W_out -= lr * grads['W_out']
        self.b_out -= lr * grads['b_out']

    def train(self, x, y, lr: float = 0.01, max_epochs: int = 10000,
              learn_by_loss: bool = False, max_loss: float = 0.01,
              verbosity: int = 1000):
        training_loss, training_mape_val = 0, 0

        for epoch in range(max_epochs):
            # Прямой проход
            y_pred, h, c, caches = self.forward(x)
            # Потери
            loss = mse_loss(y_pred, y)
            # Обратный проход
            dy_pred = mse_grad(y_pred, y)
            grads = self.backward(dy_pred, h, c, caches, x)
            # Обновление параметров
            self.update_parameters(grads, lr)

            epoch_mape = mape(y, y_pred)

            if (epoch + 1) % verbosity == 0:
                print(f"Epoch {epoch+1}/{max_epochs}, Loss: {loss:.6f}\nMAPE: {epoch_mape:.6f}")

            if learn_by_loss and loss <= max_loss:
                break

            training_loss = loss
            training_mape_val = epoch_mape

        print('TRAINING FINISHED')
        print(f"Epoch {epoch+1}/{max_epochs}, Loss: {training_loss:.6f}\nMAPE: {training_mape_val:.6f}")

        return training_loss, training_mape_val


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
