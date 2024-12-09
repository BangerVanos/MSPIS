import numpy as np
from numbers import Number


class Hopfield:

    def __init__(self, images, nu: float = 1) -> None:
        self.size = images.shape[1]
        self.w = np.matrix(np.zeros((self.size, self.size)))
        self.images = images
        self.neg_images = self._get_neg_images(self.images)
        self.nu = nu
        self.epochs = 0

    def _get_neg_images(self, images):
        return images * (-1)      

    def msign(self, s: Number, y_t: Number) -> Number:
        if s > 0:            
            return 1
        elif s < 0:            
            return -1
        else:            
            return y_t 

    def train(self, e=1e-6, max_iters=10000):
        '''Hopfield network train using D-projections method.
        Training process continues unless difference between old and new weights is
        small'''

        self.epochs = 0
        for _ in range(max_iters):
            self.epochs += 1
            old_w = self.w.copy()       
            for image in self.images:
                # x = np.matrix(self.images.copy())                        
                # x_plus = np.linalg.pinv(x) # pseudo-inverted matrix                      
                # w = x @ x_plus            
                # # print(w)

                x_t = np.matrix(image.copy()).T            
                self.w = self.w + (self.nu / self.size) * (x_t - (self.w @ x_t)) @ x_t.T
                np.fill_diagonal(self.w, 0)                             

                # x = np.matrix(image)            
                # self.w += (x.T @ x) / len(self.images)
            
            # Difference between old and new weights
            w_diff = np.max(np.absolute(old_w - self.w))           
            print(f'Epoch {self.epochs}/{max_iters}: ' 
                  f'max(w{self.epochs} - w{self.epochs - 1}) = {w_diff}')
            if w_diff < e:
                break
                                                                         
        np.fill_diagonal(self.w, 0)                                
    
    def _update_neuron(self, x, neuron_idx: int) -> Number:
        return self.msign((self.w[neuron_idx] @ x).item(), x[neuron_idx])
    
    def _find_image_num(self, x, images) -> int | None:
        mask = (images == x).all(axis=1)
        search_result = np.where(mask)[0]               
        if len(search_result) > 0:
            return search_result.item()
        return None

    def predict(self, x, max_iters: int = 100):
        state = x.copy()
        relaxation_iters = 0
        for _ in range(max_iters):            
            relaxation_iters += 1
            prev_state = state.copy()
            for i in range(self.size):                
                state[i] = self._update_neuron(state, i)
            if np.array_equal(prev_state, state):
                is_negative = False
                image_num = self._find_image_num(state, self.images)                
                neg_image_num = self._find_image_num(state, self.neg_images)
                is_negative = True if neg_image_num is not None else False

                return (relaxation_iters, state,
                        (image_num if image_num is not None else neg_image_num),
                        is_negative)            
        return max_iters, state, None, None