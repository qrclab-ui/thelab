import numpy as np

class MemoryTask:
    """
    Short-term memory task.
    Goal: output = input_{t-delay}
    """
    def __init__(self, delay=1, length=1000, test_split=0.2):
        self.delay = delay
        self.length = length
        self.test_split = test_split
        
    def generate(self):
        # Random inputs in [0, 1] (or [-1, 1] but angle encoding likes scaled)
        # Angle encoding often maps [0, 1] to [0, pi] or similar.
        X = np.random.uniform(0, 1, self.length)
        
        # Target is delayed input
        # y[t] = X[t - delay]
        # First 'delay' outputs are undefined or 0
        y = np.zeros_like(X)
        if self.delay > 0:
            y[self.delay:] = X[:-self.delay]
        else:
            y = X.copy()
            
        # Reshape X for simulator: [samples, features]
        # If single feature per step
        X = X.reshape(-1, 1) # [T, 1]
        
        split_idx = int(self.length * (1 - self.test_split))
        
        return (X[:split_idx], y[:split_idx]), (X[split_idx:], y[split_idx:])
