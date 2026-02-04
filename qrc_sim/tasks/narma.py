import numpy as np

class NARMADataset:
    """
    NARMA-n task (Nonlinear AutoRegressive Moving Average).
    Classic Reservoir Computing benchmark.
    y(t+1) = 0.3y(t) + 0.05y(t) sum(y(t-i)) + 1.5u(t-n+1)u(t) + 0.1
    """
    def __init__(self, order=10, length=2000, test_split=0.2, seed=42):
        self.order = order
        self.length = length
        self.test_split = test_split
        self.rng = np.random.RandomState(seed)
        
    def generate(self):
        # Inputs uniform [0, 0.5] as per standard NARMA papers
        u = self.rng.uniform(0, 0.5, self.length)
        y = np.zeros(self.length)
        
        for t in range(self.order, self.length):
            # Sum term: sum_{i=0}^{n-1} y(t-i)
            # Actually standard formula is slightly different variations exist.
            # Using standard: y(t+1) = 0.3y(t) + 0.05y(t)*sum(y(t-i) for i in 0..n-1) + 1.5u(t-n+1)u(t) + 0.1
            
            # Let's say we calculate y[t] based on history
            
            term1 = 0.3 * y[t-1]
            
            sum_y = np.sum(y[t-self.order:t])
            term2 = 0.05 * y[t-1] * sum_y
            
            term3 = 1.5 * u[t-self.order] * u[t-1]
            
            term4 = 0.1
            
            y[t] = term1 + term2 + term3 + term4
            
        # Reshape u
        X = u.reshape(-1, 1)
        # Shift y to align?
        # Typically we want to predict y[t] given input u[t] (and history)
        # The formula calculates y[t] from past. So at step t we have u[t] and want to predict y[t+1] usually?
        # Or just predict y[t] (which depends on past inputs).
        # In RC, we feed u[t] and train to output y[t+1] or y[t].
        # Let's target y[t] directly.
        
        target = y.reshape(-1)
        
        split_idx = int(self.length * (1 - self.test_split))
        
        return (X[:split_idx], target[:split_idx]), (X[split_idx:], target[split_idx:])
