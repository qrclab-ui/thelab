import numpy as np

class ParityTask:
    """
    Parity Check (XOR) Task.
    Goal: output = XOR(u_t, u_{t-delay})
    This requires non-linear memory.
    """
    def __init__(self, delay=1, length=1000, test_split=0.2):
        self.delay = delay
        self.length = length
        self.test_split = test_split
        
    def generate(self):
        # Binary inputs
        X = np.random.randint(0, 2, self.length)
        
        y = np.zeros_like(X)
        if self.delay < self.length:
            # y[t] = x[t] ^ x[t-delay]
            # using bitwise XOR
            y[self.delay:] = X[self.delay:] ^ X[:-self.delay]
            
        X = X.reshape(-1, 1)
        
        split_idx = int(self.length * (1 - self.test_split))
        
        return (X[:split_idx], y[:split_idx]), (X[split_idx:], y[split_idx:])
