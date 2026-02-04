from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler

class ReadoutModel:
    """
    Classical readout layer with integrated feature scaling.
    """
    def __init__(self, model_type='ridge', alpha=1.0, scale_features=False):
        self.model_type = model_type
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        
        if model_type == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
    def fit(self, X, y):
        if self.scale_features:
            X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        
    def predict(self, X):
        if self.scale_features:
            X = self.scaler.transform(X)
        return self.model.predict(X)
        
    def score(self, X, y):
        if self.scale_features:
            X = self.scaler.transform(X)
        return self.model.score(X, y)
