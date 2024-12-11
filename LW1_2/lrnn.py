import numpy as np
from PIL import Image


MAX_RGB_VALUE = 255
COLOR_CHANNELS_AMOUNT = 3


def image_to_blocks(image, b_h, b_w, overlap = 0):
    i_h, i_w = image.shape[:2]

    step_h = int(b_h * (1 - overlap))
    step_w = int(b_w * (1 - overlap))

    blocks = []

    for i in range(0, i_h - b_h + 1, step_h):
        for j in range(0, i_w - b_w + 1, step_w):
            block = image[i:i+b_h, j:j+b_w]                                  
            blocks.append(block)    
    
    if i_h % b_h != 0:
        for j in range(0, i_w - b_w + 1, step_w):
            block = image[i_h-b_h:i_h, j:j+b_w]
            blocks.append(block)    
    
    if i_w % b_w != 0:
        for i in range(0, i_h - b_h + 1, step_h):
            block = image[i:i+b_h, i_w-b_w:i_w]
            blocks.append(block)    
    
    if i_h % b_h != 0 and i_w % b_w != 0:
        block = image[i_h-b_h:i_h, i_w-b_w:i_w]
        blocks.append(block)
    
    return np.asarray(blocks)


def blocks_to_image(image_blocks, image_shape, b_h, b_w, overlap = 0):
    i_h, i_w = image_shape[:2]
    c = image_shape[2] if len(image_shape) == 3 else 1

    restored_image = np.zeros((i_h, i_w, c), dtype=np.float64)
    count_matrix = np.zeros((i_h, i_w), dtype=np.float64)
    
    step_h = int(b_h * (1 - overlap))
    step_w = int(b_w * (1 - overlap))
    
    block_index = 0
    
    for i in range(0, i_h - b_h + 1, step_h):
        for j in range(0, i_w - b_w + 1, step_w):
            block = image_blocks[block_index]            
            restored_image[i:i+b_h, j:j+b_w] += block
            count_matrix[i:i+b_h, j:j+b_w] += 1
            block_index += 1    
    
    if i_h % b_h != 0:
        for j in range(0, i_w - b_w + 1, step_w):
            block = image_blocks[block_index]
            restored_image[i_h-b_h:i_h, j:j+b_w] += block
            count_matrix[i_h-b_h:i_h, j:j+b_w] += 1
            block_index += 1    
    
    if i_w % b_w != 0:
        for i in range(0, i_h - b_h + 1, step_h):
            block = image_blocks[block_index]
            restored_image[i:i+b_h, i_w-b_w:i_w] += block
            count_matrix[i:i+b_h, i_w-b_w:i_w] += 1
            block_index += 1    
    
    if i_h % b_h != 0 and i_w % b_w != 0:
        block = image_blocks[block_index]
        restored_image[i_h-b_h:i_h, i_w-b_w:i_w] += block
        count_matrix[i_h-b_h:i_h, i_w-b_w:i_w] += 1    
    
    count_matrix[count_matrix == 0] = 1    
    restored_image = restored_image / count_matrix[..., np.newaxis]
    restored_image[restored_image > 255] = 255
    restored_image[restored_image < 0] = 0    
    
    return restored_image.astype(np.uint8)


def normalize_weights(weights):
    norms = np.linalg.norm(weights, axis=0)
    return weights / norms

# Функция активации
def linear_activation(x):
    return x


class LRNN:
    def __init__(self, input_dim, latent_dim, learning_rate=0.001):
        # np.random.seed(1)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate        
        
        self.W_enc = (normalize_weights(np.random.rand(self.input_dim, self.latent_dim))
                      .astype(np.float32))
        self.W_dec = (normalize_weights(np.random.rand(self.latent_dim, self.input_dim))
                      .astype(np.float32))        

        self.epoch: int = 0
    
    def forward(self, x):
        z = linear_activation(x @ self.W_enc)
        x_reconstructed = linear_activation(z @ self.W_dec)
        return z, x_reconstructed
    
    def backward(self, x, x_reconstructed):
        error = x_reconstructed - x        
        
        dW_dec = (x @ self.W_enc).T @ error
        dW_enc = x.T @ (error @ self.W_dec.T)                       
        
        self.W_dec -= self.learning_rate * dW_dec
        self.W_enc -= self.learning_rate * dW_enc        
        
        self.W_dec = normalize_weights(self.W_dec)
        self.W_enc = normalize_weights(self.W_enc)
    
    def squared_error(self, y_true, y_predicted) -> float:
        error = 0
        y_true, y_predicted = np.array(y_true)[0], np.array(y_predicted)[0]
        if len(y_true) != len(y_predicted):
            raise ValueError('True and predicted vectors must be same size!')
        for i in range(len(y_true)):
            error += (y_true[i] - y_predicted[i]) * (y_true[i] - y_predicted[i])
        return error
    
    def train(self, data, epochs=1000, max_loss: float = 100, learn_by_loss: bool = False,
              verbosity: int = 1):
        for epoch in range(epochs):
            self.epoch += 1
            total_loss = 0
            for x in data:                
                x = np.matrix(x)
                _, x_reconstructed = self.forward(x)
                self.backward(x, x_reconstructed)
            for x in data:
                _, x_reconstructed = self.forward(x)
                total_loss += self.squared_error(x, x_reconstructed)
            mse = total_loss / len(data)
            if self.epoch % max(verbosity, 1) == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss}')
                print(f'MSE: {mse:.6f}')            
            if learn_by_loss and total_loss <= max_loss:
                print('TRAINING FINISHED ON:')
                print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss}')
                print(f'MSE: {mse:.6f}')
                break                    


# Image compression/decompression pipeline
def compress_image(compression_weights, img_array, channels_amount: int,
                   block_height: int, block_width: int, overlap: float = 0):    
    normalized = (2.0 * img_array.astype(np.float32) / MAX_RGB_VALUE) - 1.0
    blocks = image_to_blocks(normalized, block_height, block_width, overlap)
    blocks = blocks.reshape((len(blocks), block_height * block_width, channels_amount))
    if channels_amount == 3:
        blocks = blocks.transpose(0, 2, 1)    
    blocks = np.einsum('ijk,kl->ijl', blocks, compression_weights)     
    return blocks
    

def decompress_image(decompression_weights, compressed_img, img_shape, channels_amount: int,
                     block_height: int, block_width: int, overlap: float = 0) -> Image.Image:
    compressed_img = np.einsum('ijk,kl->ijl', compressed_img, decompression_weights)
    compressed_img = MAX_RGB_VALUE * (compressed_img + 1.0) / 2.0
    if channels_amount == 3:
        compressed_img = compressed_img.transpose(0, 2, 1)
    compressed_img = compressed_img.reshape((len(compressed_img), block_height, block_width, channels_amount))    
    img_array = blocks_to_image(compressed_img, img_shape, block_height, block_width, overlap)    
    return Image.fromarray(img_array).convert('RGB' if channels_amount == 3 else 'L')

