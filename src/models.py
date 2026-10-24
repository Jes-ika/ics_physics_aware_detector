import numpy as np
import torch
from torch import nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class ForecastNet(nn.Module):
    """Small self-supervised sequence model: predict the next sensor/actuator vector."""
    def __init__(self, n_features: int, hidden: int = 48):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, n_features))

    def forward(self, x):
        z, _ = self.lstm(x)
        return self.head(z[:, -1, :])

def make_windows(values: np.ndarray, window: int = 20):
    xs, ys, end_indices = [], [], []
    for end in range(window, len(values)):
        xs.append(values[end-window:end])
        ys.append(values[end])
        end_indices.append(end)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(end_indices)

def train_forecaster(train_values, window=20, epochs=12, batch_size=128, seed=17):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, y, _ = make_windows(train_values, window)
    if len(X) == 0:
        raise ValueError("Not enough rows to create training windows.")
    model = ForecastNet(train_values.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        order = np.random.permutation(len(X))
        total = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start:start+batch_size]
            xb = torch.from_numpy(X[idx])
            yb = torch.from_numpy(y[idx])
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"epoch {epoch+1:02d}/{epochs} loss={total/len(X):.6f}")
    return model

def forecast_scores(model, values, window=20):
    X, y, idx = make_windows(values, window)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X)).numpy()
    per_feature = np.abs(pred - y)
    scores = np.mean(np.square(pred - y), axis=1)
    return idx, scores, per_feature

def fit_isolation_forest(train_features, seed=17):
    scaler = StandardScaler()
    x = scaler.fit_transform(train_features)
    model = IsolationForest(n_estimators=200, contamination="auto", random_state=seed, n_jobs=-1)
    model.fit(x)
    return scaler, model

def isolation_scores(scaler, model, features):
    x = scaler.transform(features)
    # Higher score = more anomalous
    return -model.score_samples(x)
