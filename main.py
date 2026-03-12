import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, ConfusionMatrixDisplay, confusion_matrix, classification_report



def load_data(filepath="Model_Data.xlsx"):
    """
    Load the dataset from the first worksheet in the Excel file.
    """
    xlsx_path = filepath
    xls_file = pd.ExcelFile(xlsx_path)
    sheet_name = xls_file.sheet_names[0]
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    return df

def preprocess(df):
    """
    Create train/test split for regression on ΔPurity
    Note: update the dropped columns if the dataset structure changes
    """
    y = df['ΔPurity']
    # Use remaining columns as predictor features
    X = df.drop(columns=['Debutanizer - Stage_Temperature_1 (C)', 'Debutanizer - Stage_Temperature_10 (C)',
                         'Debutanizer - Stage_Temperature_20 (C)', 'Distillate - N-Butane Molar Fraction',
                         'Purity Specification', 'ΔPurity'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def preprocess_noise(X_train,X_test):
    """
    Standardize features and add training noise for robustness.
    Note: the number and order of feature columns must match with noise_factors.
    """
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Create Gaussian Noise per feature (Feed Molar Rate, Reboiler Duty, Condenser Duty, ΔT rectifying, ΔT stripping)
    noise_factors = np.array([0.02, 0.05, 0.05, 0.08, 0.08])

    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, noise_factors, size=X_train_std.shape)

    # Add noise only to training data
    X_train_noisy = X_train_std + noise
    return X_train_noisy, X_train_std, X_test_std

def preprocess_nn(df):
    """
    Standardize features and add training noise for neural network training.
    Note: the number and order of feature columns must match with noise_factors.
    """
    y = df['ΔPurity']
    # Use remaining columns as predictor features
    X = df.drop(columns=['Debutanizer - Stage_Temperature_1 (C)', 'Debutanizer - Stage_Temperature_10 (C)',
                         'Debutanizer - Stage_Temperature_20 (C)', 'Distillate - N-Butane Molar Fraction',
                         'Purity Specification', 'ΔPurity'])
    X_np = X.to_numpy().astype(np.float32)
    y_np = y.to_numpy().astype(np.float32).reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

    # Standardize Features
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train_std = (X_train - mu) / sigma
    X_test_std = (X_test - mu) / sigma

    # Create Gaussian Noise per feature (Feed Molar Rate, Reboiler Duty, Condenser Duty, ΔT rectifying, ΔT stripping)
    noise_factors = np.array([0.02, 0.05, 0.05, 0.08, 0.08])

    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, noise_factors, size=X_train_std.shape).astype(np.float32)

    # Add Noise to only the Training Features
    X_train_noisy = X_train_std + noise.astype(np.float32)

    # DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train_noisy).float(), torch.from_numpy(y_train).float())
    test_ds = TensorDataset(torch.from_numpy(X_test_std).float(), torch.from_numpy(y_test).float())
    return train_ds , test_ds

def get_activation(name: str):
    """
    Return requested activation function.
    """
    name = name.lower()
    if name == "relu":  return nn.ReLU()
    if name == "tanh":  return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")

class MLP(nn.Module):
    """
    Configurable Multi-Layer Perceptron.
    Hidden can be:
      (32,)   -> 1 layer
      (64,64) -> 2 layers
    """
    # Build the network layers
    def __init__(self, d_in, hidden=(32,), activation="relu"):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(get_activation(activation))
            prev = h
        layers.append(nn.Linear(prev, 1))  # Single output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def preprocess_binary(df):
    """
    Create train/test split for purity prediction.
    """
    # Require reflux increase given distillate purity below 99.0 mol%
    df["y_increase"] = (df["Distillate - N-Butane Molar Fraction"] < 0.99).astype(int)
    y = df["y_increase"]
    # Use remaining columns as predictor features
    X = df.drop(columns=['Debutanizer - Stage_Temperature_1 (C)', 'Debutanizer - Stage_Temperature_10 (C)',
                         'Debutanizer - Stage_Temperature_20 (C)', 'Distillate - N-Butane Molar Fraction',
                         'Purity Specification', 'ΔPurity',"y_increase"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test


### Start of Main Workflow


# Load data and prepare regression datasets
df = load_data("Model_Data.xlsx")
X_train, X_test, y_train, y_test = preprocess(df)
X_train_noisy, X_train_std, X_test_std = preprocess_noise(X_train,X_test)

#Linear Regression Training and Evaluation
model_linear = linear_model.LinearRegression()
model_linear.fit(X_train_noisy, y_train)
pred_linear = model_linear.predict(X_test_std)
mse = mean_squared_error(y_test, pred_linear)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred_linear)
print(f"Linear Regression Root Mean Squared Error: {rmse}")
print(f"Linear Regression R-squared Score: {r2}")

#Plot of Linear Regression: Actual vs. Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, pred_linear, color='blue', alpha=0.5, label='Predicted vs Actual')
max_val = max(y_test.max(), pred_linear.max())
min_val = min(y_test.min(), pred_linear.min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', lw=2, label='Perfect Prediction')
plt.xlabel('Actual ΔPurity')
plt.ylabel('Predicted ΔPurity')
plt.title('Linear Regression: Actual vs. Predicted')
plt.legend()
plt.grid(True)
plt.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3g}', transform=plt.gca().transAxes,
         va='top', ha='left', fontsize=12, fontweight='bold')
plt.show()

# Plot of Linear Regression: Feature Coefficients
coefficients = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': model_linear.coef_})
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
plt.figure(figsize=(10, 6))
plt.barh(coefficients['Feature'], coefficients['Coefficient'], color='skyblue')
plt.xlabel('Coefficient Value', fontsize=18, weight='bold')
plt.ylabel('Feature', fontsize=18, weight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('Impact of Each Feature on ΔPurity', fontsize=18, weight='bold')
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
plt.tight_layout()
plt.show()

# Plot of Linear Regression: Residuals
residuals = y_test - pred_linear
plt.figure(figsize=(8, 6))
plt.scatter(pred_linear, residuals, color='purple', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted ΔPurity (mole fraction)', fontsize=18, weight='bold')
plt.ylabel('Residuals (mole fraction)', fontsize=18, weight='bold')
plt.xlim(-0.20, 0.10)
plt.ylim(-0.16, 0.10)
plt.xticks(np.linspace(-0.15, 0.05, 5), fontsize=16)
plt.yticks(np.linspace(-0.12, 0.06, 5), fontsize=16)
plt.title('OLS: Residual Plot', fontsize=18, weight='bold')
plt.grid(True)
plt.show()

# Lasso Regularization
# Train on noisy, test on clean
model_lasso = linear_model.Lasso(alpha=0.01)
model_lasso.fit(X_train_noisy, y_train)
pred_lasso = model_lasso.predict(X_test_std)
mse = mean_squared_error(y_test, pred_lasso)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred_lasso)
print(f"Lasso Root Mean Squared Error: {rmse}")
print(f"Lasso R-squared Score: {r2}")

# Ridge Regularization
# Train on noisy, test on clean
model_ridge = linear_model.Ridge(alpha=0.01)
model_ridge.fit(X_train_noisy, y_train)
pred_ridge = model_ridge.predict(X_test_std)
mse_test = mean_squared_error(y_test, pred_ridge)
rmse_test = np.sqrt(mse)
r2_test = r2_score(y_test, pred_ridge)
print(f"Ridge: Root Mean Squared Error Test: {rmse_test}")
print(f"Ridge: R-squared Score Test: {r2_test}")


# Neural Network
# Hyperparameters and Data
train_ds , test_ds = preprocess_nn(df)
EPOCHS = 50
BATCH_SIZE = 64      
LR = 1e-3           
WEIGHT_DECAY = 0.0
ACTIVATION = "relu" 
HIDDEN = (32, 32) 
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

# Build the neural network model
model = MLP(d_in=X_train_noisy.shape[1], hidden=HIDDEN, activation=ACTIVATION)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.MSELoss()

# Train and evaluation neural network
for EPOCH in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        # forward pass
        pred = model(xb)
        # compute loss
        loss = loss_fn(pred, yb)
        # backward + step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
    train_mse = total_loss / len(train_ds)
    # Begin Evaluation
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb)
            preds.append(pred.detach().cpu().numpy())
            trues.append(yb.detach().cpu().numpy())

    preds = np.vstack(preds).ravel()
    trues = np.vstack(trues).ravel()

    test_rmse = np.sqrt(np.mean((preds - trues) ** 2))
    test_mae = np.mean(np.abs(preds - trues))
    test_r2 = r2_score(trues, preds)

    if EPOCH % 10 == 0:
        print(
            f"Epoch {EPOCH:03d} | Train MSE: {train_mse:.6f} "
            f"| Test RMSE: {test_rmse:.6f} | Test MAE: {test_mae:.6f} | Test R2: {test_r2:.4f}"
        )

print("\nFINAL SETTINGS USED:")
print(f"HIDDEN={HIDDEN}, ACTIVATION={ACTIVATION}, LR={LR}, BATCH_SIZE={BATCH_SIZE}, WEIGHT_DECAY={WEIGHT_DECAY}")

# Plot neural network: residuals
residuals = trues - preds
plt.figure(figsize=(8, 6))
plt.scatter(preds, residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle="--")
plt.xlabel('Predicted ΔPurity (mole fraction)', fontsize=18, weight='bold')
plt.ylabel('Residuals (mole fraction)', fontsize=18, weight='bold')
plt.xlim(-0.25, 0.05)
plt.ylim(-0.015, 0.015)
plt.xticks(np.linspace(-0.20, 0.00, 5), fontsize=16)
plt.yticks(np.linspace(-0.01, 0.01, 5), fontsize=16)
plt.title('Neural Network: Residual Plot', fontsize=18, weight='bold')
plt.grid(True)
plt.show()


#Binary Classification 
X_train, X_test, y_train, y_test = preprocess_binary(df)
X_train_noisy, X_train_std, X_test_std = preprocess_noise(X_train,X_test)
classify = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
classify.fit(X_train_noisy, y_train.to_numpy())
y_prob = classify.predict_proba(X_test_std)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# Evaluation Metrics
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Plot Classifer: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(3, 3))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Increase", "Increase"])
disp.plot(ax=ax, cmap="Blues", colorbar=False)

ax.set_xlabel("Predicted label", fontsize=14, fontweight="bold")
ax.set_ylabel("True label", fontsize=14, fontweight="bold")
ax.tick_params(axis="x", labelsize=13)
for tick in ax.get_yticklabels():
    tick.set_rotation(90)
    tick.set_fontsize(14)
    tick.set_va("center")
    tick.set_ha("center")
for text in ax.texts:
    text.set_fontsize(16)
    text.set_fontweight("bold")
plt.tight_layout()
plt.show()