import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Reproductibilité stricte ---
SEED = 42
def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

# --- Architectures Neuronales Corrigées ---
class SensorLSTM(nn.Module):
    """
    Modèle de référence : LSTM par capteur univarié indépendant.
    Chaque capteur est modélisé sans communication spatiale.
    Prédiction résiduelle : prédit la variation par rapport à la dernière valeur observée.
    """
    def __init__(self, hidden_dim=48, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x : (Batch, Time, Nodes)
        B, T, N = x.shape
        x_reshaped = x.permute(0, 2, 1).reshape(B * N, T, 1)
        out, _ = self.lstm(x_reshaped)
        last_hidden = out[:, -1, :]  # (B*N, hidden_dim)
        delta = self.fc(last_hidden).view(B, N)
        # Prédiction résiduelle : dernière observation + variation
        return x[:, -1, :] + delta


class GCNLSTM(nn.Module):
    """
    Architecture avancée : GCN-LSTM Spatio-Temporel.
    Reçoit 2 canaux par capteur à chaque instant t :
      - Canal 0 : le signal propre du capteur x_t^i
      - Canal 1 : le signal agrégé de ses 4 plus proches voisins (A @ x_t)^i
    Couche spatiale : nn.Linear(2, gcn_dim)
    Puis LSTM temporel et prédiction résiduelle (+ x[:, -1, :]).
    En mode ablation (A = 0), le canal 1 est nul, neutralisant la connectivité spatiale.
    """
    def __init__(self, adj_matrix, hidden_dim=48, gcn_dim=16, num_layers=1, dropout=0.0):
        super().__init__()
        self.register_buffer('A', torch.tensor(adj_matrix, dtype=torch.float32))
        self.gcn_linear = nn.Linear(2, gcn_dim)  # 2 canaux : propre signal + agrégat voisins
        self.lstm = nn.LSTM(input_size=gcn_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x : (Batch, Time, Nodes)
        B, T, N = x.shape
        # 1. Agrégation des voisins par capteur via A (normalisée par ligne)
        x_neighbors = torch.einsum('nm, btm -> btn', self.A, x)
        # 2. Concaténation des 2 canaux (propre signal et agrégat) : (B, T, N, 2)
        x_combined = torch.stack([x, x_neighbors], dim=-1)
        # 3. Projection spatiale
        h_spatial = torch.relu(self.gcn_linear(x_combined))  # (B, T, N, gcn_dim)
        # 4. LSTM séquentiel
        z = h_spatial.permute(0, 2, 1, 3).reshape(B * N, T, -1)  # (B*N, T, gcn_dim)
        out, _ = self.lstm(z)
        last_hidden = out[:, -1, :]  # (B*N, hidden_dim)
        delta = self.fc(last_hidden).view(B, N)
        # Prédiction résiduelle : dernière observation + variation
        return x[:, -1, :] + delta


def build_knn_adj(D_mat, k=4, use_edges=True):
    """
    Graphe routier : k=4 plus proches voisins par capteur, normalisation par ligne.
    Chaque ligne i a une somme égale à 1.0 sur les k voisins.
    Si use_edges=False (ablation) : matrice de zéros.
    """
    n = D_mat.shape[0]
    if not use_edges:
        return np.zeros((n, n), dtype=np.float32)

    A = np.zeros((n, n), dtype=np.float32)
    dist_valid = D_mat[D_mat > 0]
    sigma = np.std(dist_valid) if len(dist_valid) > 0 else 1.0
    if sigma == 0 or np.isnan(sigma):
        sigma = 1.0

    for i in range(n):
        dists = D_mat[i].copy()
        dists[i] = np.inf  # exclure soi-même des voisins
        knn_idx = np.argsort(dists)[:k]
        weights = np.exp(-((dists[knn_idx] / sigma) ** 2))
        row_sum = float(weights.sum())
        if row_sum > 0:
            A[i, knn_idx] = weights / row_sum
        else:
            A[i, knn_idx] = 1.0 / k
    return A.astype(np.float32)


def main():
    set_seed()
    torch.set_num_threads(min(4, os.cpu_count() or 2))
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device d'exécution : {DEVICE} | PyTorch : {torch.__version__}", flush=True)

    FIG_DIR = "figures"
    MODELS_DIR = "models_saved"
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- 1. Chargement des données PeMSD7 ---
    print("\n================ 1. CHARGEMENT DU DATASET PeMSD7 ================", flush=True)
    speeds_raw = pd.read_csv('PeMSD7_V_228.csv', header=None).values.astype(np.float32)
    dist_raw = pd.read_csv('PeMSD7_W_228.csv', header=None).values.astype(np.float32)
    print(f"Données brutes chargées : {speeds_raw.shape[0]} pas de 5 min sur {speeds_raw.shape[1]} capteurs.", flush=True)
    print(f"Matrice de distances : {dist_raw.shape}", flush=True)

    # --- 2. Sous-ensemble & Nettoyage (30 capteurs) ---
    print("\n================ 2. SOUS-ENSEMBLE & NETTOYAGE ================", flush=True)
    N_SENSORS = 30
    rng = np.random.RandomState(SEED)
    sensor_indices = np.sort(rng.choice(speeds_raw.shape[1], N_SENSORS, replace=False))

    X = speeds_raw[:, sensor_indices].copy()
    D = dist_raw[np.ix_(sensor_indices, sensor_indices)].copy()
    print(f"Sous-graphe fixé : {N_SENSORS} capteurs sélectionnés avec SEED={SEED}.", flush=True)

    invalid_mask = (X <= 0) | (X > 100) | np.isnan(X)
    n_invalids = int(invalid_mask.sum())
    print(f"Valeurs invalides / aberrantes détectées : {n_invalids} ({n_invalids/X.size*100:.3f}%)", flush=True)
    if n_invalids > 0:
        X[invalid_mask] = np.nan
        X = pd.DataFrame(X).interpolate(method='linear', limit_direction='both').values.astype(np.float32)

    print(f"Statistiques globales ({N_SENSORS} capteurs) : min = {X.min():.2f} mph, max = {X.max():.2f} mph, moyenne = {X.mean():.2f} mph", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    axes[0].plot(X[:288 * 3, :3], alpha=0.85)
    axes[0].set_title(f"Séries temporelles sur 3 jours consécutifs (3 capteurs sur {N_SENSORS})", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Pas de temps de 5 min")
    axes[0].set_ylabel("Vitesse (mph)")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].hist(X.ravel(), bins=60, color='royalblue', edgecolor='black', alpha=0.75)
    axes[1].set_title(f"Distribution des vitesses ({N_SENSORS} capteurs, {len(X)} pas)", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Vitesse (mph)")
    axes[1].set_ylabel("Fréquence")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/eda_traffic.png", dpi=150)
    plt.close()
    print(f"Figure EDA enregistrée sous '{FIG_DIR}/eda_traffic.png'", flush=True)

    # --- 3. Construction du graphe (4 plus proches voisins, normalisation par ligne) ---
    print("\n================ 3. CONSTRUCTION DU GRAPHE ROUTIER ================", flush=True)
    A_norm = build_knn_adj(D, k=4, use_edges=True)
    A_identity = build_knn_adj(D, k=4, use_edges=False)

    nb_edges = int((A_norm > 0).sum())
    density = nb_edges / (N_SENSORS * (N_SENSORS - 1))
    print(f"Topologie 4-NN : {nb_edges} arêtes actives ({nb_edges//N_SENSORS} voisins/nœud, densité = {density:.3f})", flush=True)
    print(f"Vérification normalisation par ligne : min sum = {A_norm.sum(axis=1).min():.4f}, max sum = {A_norm.sum(axis=1).max():.4f}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    im1 = axes[0].imshow(A_norm, cmap='magma')
    axes[0].set_title(f"Adjacence 4-NN normalisée par ligne $\\hat{{A}}$ ({N_SENSORS} capteurs)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Capteur")
    axes[0].set_ylabel("Capteur")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(A_identity, cmap='magma')
    axes[1].set_title(f"Ablation : Adjacence sans arêtes $\\hat{{A}}_{{no-edge}} = 0$ ({N_SENSORS} capteurs)", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Capteur")
    axes[1].set_ylabel("Capteur")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/graph_adjacency.png", dpi=150)
    plt.close()
    print(f"Figure Graphe enregistrée sous '{FIG_DIR}/graph_adjacency.png'", flush=True)

    # --- 4. Découpage chronologique & Fenêtrage (70% Train, 15% Val, 15% Test) ---
    print("\n================ 4. DÉCOUPAGE CHRONOLOGIQUE SANS FUITE ================", flush=True)
    T = len(X)
    idx_train = int(0.70 * T)
    idx_val = int(0.85 * T)

    mu_train = float(X[:idx_train].mean())
    std_train = float(X[:idx_train].std())
    print(f"Train z-score (calculé uniquement sur Train) : mu = {mu_train:.2f} mph, sigma = {std_train:.2f} mph", flush=True)

    X_scaled = (X - mu_train) / std_train

    IN_LEN = 12   # 12 pas = 60 minutes d'historique
    HORIZON = 3   # 3 pas = prédiction à t+15 minutes

    def make_windows(series):
        xs, ys = [], []
        for t in range(IN_LEN, len(series) - HORIZON + 1):
            xs.append(series[t - IN_LEN:t])
            ys.append(series[t + HORIZON - 1])
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    X_tr, y_tr = make_windows(X_scaled[:idx_train])
    X_va, y_va = make_windows(X_scaled[idx_train:idx_val])
    X_te, y_te = make_windows(X_scaled[idx_val:])

    print(f"Partition Train : {X_tr.shape[0]} fenêtres (forme: {X_tr.shape})", flush=True)
    print(f"Partition Val   : {X_va.shape[0]} fenêtres (forme: {X_va.shape})", flush=True)
    print(f"Partition Test  : {X_te.shape[0]} fenêtres (forme: {X_te.shape})", flush=True)

    def eval_metrics(y_true_n, y_pred_n):
        yt = y_true_n * std_train + mu_train
        yp = y_pred_n * std_train + mu_train
        return {
            "MAE": float(mean_absolute_error(yt.ravel(), yp.ravel())),
            "RMSE": float(np.sqrt(mean_squared_error(yt.ravel(), yp.ravel()))),
            "R2": float(r2_score(yt.ravel(), yp.ravel()))
        }

    RESULTS = {}
    PREDICTIONS = {}
    TRAIN_LOSSES = {}
    VAL_LOSSES = {}

    # --- 5. Algorithme 1 : Baselines simples ---
    print("\n================ 5. BASELINES SIMPLES ================", flush=True)
    # 5.1 Persistance temporelle v(t+h) = v(t)
    t0 = time.time()
    pred_persist = X_te[:, -1, :]
    inf_persist = time.time() - t0
    RESULTS["Baseline Persistance"] = {
        **eval_metrics(y_te, pred_persist),
        "Train_Time_s": 0.0,
        "Infer_Time_s": inf_persist,
        "Params": 0
    }
    PREDICTIONS["Baseline Persistance"] = pred_persist
    print(f"Persistance : MAE = {RESULTS['Baseline Persistance']['MAE']:.3f} mph, RMSE = {RESULTS['Baseline Persistance']['RMSE']:.3f} mph, R² = {RESULTS['Baseline Persistance']['R2']:.3f}", flush=True)

    # 5.2 Moyenne historique
    steps_per_day = 288
    profile_day = np.zeros((steps_per_day, N_SENSORS), dtype=np.float32)
    for s in range(steps_per_day):
        profile_day[s] = X_scaled[:idx_train][s::steps_per_day].mean(axis=0)

    test_start = idx_val + IN_LEN + HORIZON - 1
    tod_test = (np.arange(len(y_te)) + test_start) % steps_per_day
    t0 = time.time()
    pred_hist = profile_day[tod_test]
    inf_hist = time.time() - t0

    RESULTS["Baseline Moyenne Horaire"] = {
        **eval_metrics(y_te, pred_hist),
        "Train_Time_s": 0.0,
        "Infer_Time_s": inf_hist,
        "Params": 0
    }
    PREDICTIONS["Baseline Moyenne Horaire"] = pred_hist
    print(f"Moyenne Horaire : MAE = {RESULTS['Baseline Moyenne Horaire']['MAE']:.3f} mph, RMSE = {RESULTS['Baseline Moyenne Horaire']['RMSE']:.3f} mph, R² = {RESULTS['Baseline Moyenne Horaire']['R2']:.3f}", flush=True)

    # --- 6. Algorithme 2 : Random Forest ---
    print("\n================ 6. MODÈLE CLASSIQUE : RANDOM FOREST ================", flush=True)
    def extract_features_rf(X_win, offset):
        S = X_win.shape[0]
        steps = (np.arange(S) + offset) % steps_per_day
        sin_t = np.sin(2 * np.pi * steps / steps_per_day)
        cos_t = np.cos(2 * np.pi * steps / steps_per_day)
        feats = []
        for s_idx in range(N_SENSORS):
            lags = X_win[:, :, s_idx]
            s_col = np.full((S, 1), s_idx, dtype=np.float32)
            feats.append(np.column_stack([lags, sin_t, cos_t, s_col]))
        return np.vstack(feats)

    def extract_targets_rf(Y_win):
        return Y_win.T.ravel()

    F_tr = extract_features_rf(X_tr, IN_LEN + HORIZON - 1)
    y_tr_rf = extract_targets_rf(y_tr)
    F_va = extract_features_rf(X_va, idx_train + IN_LEN + HORIZON - 1)
    y_va_rf = extract_targets_rf(y_va)
    F_te = extract_features_rf(X_te, idx_val + IN_LEN + HORIZON - 1)

    sub_rf = np.random.RandomState(SEED).choice(len(F_tr), min(len(F_tr), 25000), replace=False)

    rf_grid = [
        {"n_estimators": 40, "max_depth": 12, "min_samples_leaf": 4},
        {"n_estimators": 60, "max_depth": 16, "min_samples_leaf": 4},
        {"n_estimators": 60, "max_depth": 18, "min_samples_leaf": 2},
    ]

    print("Tuning des hyperparamètres Random Forest sur Validation...", flush=True)
    best_rf_mae = float('inf')
    best_rf_params = None

    for params in rf_grid:
        rf_c = RandomForestRegressor(**params, n_jobs=1, random_state=SEED)
        rf_c.fit(F_tr[sub_rf], y_tr_rf[sub_rf])
        p_val = rf_c.predict(F_va)
        val_mae = mean_absolute_error(y_va_rf * std_train + mu_train, p_val * std_train + mu_train)
        print(f"  Configuration RF {params} -> Val MAE = {val_mae:.3f} mph", flush=True)
        if val_mae < best_rf_mae:
            best_rf_mae = val_mae
            best_rf_params = params

    print(f"Meilleurs hyperparamètres RF retenus : {best_rf_params}", flush=True)

    t0 = time.time()
    rf_model = RandomForestRegressor(**best_rf_params, n_jobs=1, random_state=SEED)
    rf_model.fit(F_tr[sub_rf], y_tr_rf[sub_rf])
    tr_time_rf = time.time() - t0

    t0 = time.time()
    p_te_flat = rf_model.predict(F_te)
    inf_time_rf = time.time() - t0
    pred_rf = p_te_flat.reshape(N_SENSORS, -1).T

    # Sauvegarde du Random Forest en joblib
    rf_path = os.path.join(MODELS_DIR, "random_forest.joblib")
    joblib.dump(rf_model, rf_path)
    rf_size_mb = os.path.getsize(rf_path) / (1024 * 1024)
    print(f"Modèle Random Forest sauvegardé : '{rf_path}' (Taille : {rf_size_mb:.2f} Mo)", flush=True)

    RESULTS["Random Forest"] = {
        **eval_metrics(y_te, pred_rf),
        "Train_Time_s": tr_time_rf,
        "Infer_Time_s": inf_time_rf,
        "Params": sum(t.tree_.node_count for t in rf_model.estimators_),
        "FileSize_MB": rf_size_mb
    }
    PREDICTIONS["Random Forest"] = pred_rf
    print(f"Random Forest : MAE = {RESULTS['Random Forest']['MAE']:.3f} mph, RMSE = {RESULTS['Random Forest']['RMSE']:.3f} mph, R² = {RESULTS['Random Forest']['R2']:.3f}", flush=True)

    # --- 7. Modèles Neuronaux PyTorch ---
    print("\n================ 7. MODÈLES NEURONAUX PYTORCH ================", flush=True)
    BATCH_SIZE = 256
    train_ds = torch.utils.data.TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    val_ds = torch.utils.data.TensorDataset(torch.tensor(X_va), torch.tensor(y_va))
    test_ds = torch.utils.data.TensorDataset(torch.tensor(X_te), torch.tensor(y_te))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    def train_nn(model, lr, max_epochs=20, patience=4, verbose=False):
        model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        best_val = float('inf')
        best_state = None
        patience_k = 0
        tr_hist, va_hist = [], []
        
        for ep in range(max_epochs):
            model.train()
            tot_tr = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                preds = model(bx)
                l = criterion(preds, by)
                l.backward()
                optimizer.step()
                tot_tr += l.item() * len(bx)
            ep_tr = tot_tr / len(train_loader.dataset)
            
            # Validation
            model.eval()
            tot_va = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    vl = criterion(model(bx), by)
                    tot_va += vl.item() * len(bx)
            ep_va = tot_va / len(val_loader.dataset)
            
            tr_hist.append(ep_tr)
            va_hist.append(ep_va)
            if verbose:
                print(f"    Époque {ep+1:02d}/{max_epochs:02d} - Train MSE: {ep_tr:.4f} | Val MSE: {ep_va:.4f}", flush=True)
                
            if ep_va < best_val - 1e-5:
                best_val = ep_va
                patience_k = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_k += 1
                if patience_k >= patience:
                    if verbose:
                        print(f"    Early stopping déclenché à l'époque {ep+1}.", flush=True)
                    break
                    
        model.load_state_dict(best_state)
        return best_val, tr_hist, va_hist

    def predict_nn(model, loader):
        model.eval()
        model.to(DEVICE)
        res = []
        with torch.no_grad():
            for bx, _ in loader:
                res.append(model(bx.to(DEVICE)).cpu().numpy())
        return np.vstack(res)

    # 7.1 Tuning hyperparamètres LSTM par capteur
    print("\nTuning des hyperparamètres LSTM sur Validation...", flush=True)
    lstm_grid = [
        {"hidden_dim": 32, "lr": 0.005},
        {"hidden_dim": 48, "lr": 0.003},
        {"hidden_dim": 48, "lr": 0.001},
    ]
    best_lstm_val = float('inf')
    best_lstm_hp = None
    for hp in lstm_grid:
        set_seed(SEED)
        m_t = SensorLSTM(hidden_dim=hp["hidden_dim"])
        v_loss, _, _ = train_nn(m_t, lr=hp["lr"], max_epochs=6, patience=2)
        v_mae = mean_absolute_error(y_va * std_train + mu_train, predict_nn(m_t, val_loader) * std_train + mu_train)
        print(f"  LSTM Config {hp} -> Val MAE = {v_mae:.3f} mph (Loss: {v_loss:.4f})", flush=True)
        if v_loss < best_lstm_val:
            best_lstm_val = v_loss
            best_lstm_hp = hp
    print(f"Meilleurs hyperparamètres LSTM : {best_lstm_hp}", flush=True)

    # 7.2 Tuning hyperparamètres GCN-LSTM
    print("\nTuning des hyperparamètres GCN-LSTM sur Validation...", flush=True)
    gcn_grid = [
        {"hidden_dim": 32, "gcn_dim": 8, "lr": 0.005},
        {"hidden_dim": 48, "gcn_dim": 16, "lr": 0.003},
        {"hidden_dim": 48, "gcn_dim": 16, "lr": 0.001},
    ]
    best_gcn_val = float('inf')
    best_gcn_hp = None
    for hp in gcn_grid:
        set_seed(SEED)
        m_t = GCNLSTM(A_norm, hidden_dim=hp["hidden_dim"], gcn_dim=hp["gcn_dim"])
        v_loss, _, _ = train_nn(m_t, lr=hp["lr"], max_epochs=6, patience=2)
        v_mae = mean_absolute_error(y_va * std_train + mu_train, predict_nn(m_t, val_loader) * std_train + mu_train)
        print(f"  GCN-LSTM Config {hp} -> Val MAE = {v_mae:.3f} mph (Loss: {v_loss:.4f})", flush=True)
        if v_loss < best_gcn_val:
            best_gcn_val = v_loss
            best_gcn_hp = hp
    print(f"Meilleurs hyperparamètres GCN-LSTM : {best_gcn_hp}", flush=True)

    # --- 8. Entraînements complets finaux et Test (Graine 42) ---
    print("\n================ 8. ENTRAÎNEMENTS COMPLETS ET TEST (Graine 42) ================", flush=True)

    # 8.1 LSTM par capteur
    print("\n[Modèle Référence] Entraînement complet du LSTM par capteur...", flush=True)
    set_seed(SEED)
    lstm_final = SensorLSTM(hidden_dim=best_lstm_hp["hidden_dim"])
    t0 = time.time()
    _, tr_h_lstm, va_h_lstm = train_nn(lstm_final, lr=best_lstm_hp["lr"], max_epochs=20, patience=4, verbose=True)
    tr_time_lstm = time.time() - t0

    t0 = time.time()
    pred_lstm = predict_nn(lstm_final, test_loader)
    inf_time_lstm = time.time() - t0

    TRAIN_LOSSES["LSTM par capteur"] = tr_h_lstm
    VAL_LOSSES["LSTM par capteur"] = va_h_lstm
    RESULTS["LSTM par capteur"] = {
        **eval_metrics(y_te, pred_lstm),
        "Train_Time_s": tr_time_lstm,
        "Infer_Time_s": inf_time_lstm,
        "Params": sum(p.numel() for p in lstm_final.parameters()),
        "FileSize_MB": 0.0  # sera mesuré après sauvegarde
    }
    PREDICTIONS["LSTM par capteur"] = pred_lstm
    print(f"-> LSTM : MAE = {RESULTS['LSTM par capteur']['MAE']:.3f} mph, RMSE = {RESULTS['LSTM par capteur']['RMSE']:.3f} mph, R² = {RESULTS['LSTM par capteur']['R2']:.3f}", flush=True)

    # 8.2 GCN-LSTM (avec graphe 4-NN normalisé)
    print("\n[Modèle Avancé Imposé] Entraînement complet de GCN-LSTM...", flush=True)
    set_seed(SEED)
    gcn_final = GCNLSTM(A_norm, hidden_dim=best_gcn_hp["hidden_dim"], gcn_dim=best_gcn_hp["gcn_dim"])
    t0 = time.time()
    _, tr_h_gcn, va_h_gcn = train_nn(gcn_final, lr=best_gcn_hp["lr"], max_epochs=20, patience=4, verbose=True)
    tr_time_gcn = time.time() - t0

    t0 = time.time()
    pred_gcn = predict_nn(gcn_final, test_loader)
    inf_time_gcn = time.time() - t0

    TRAIN_LOSSES["GCN-LSTM"] = tr_h_gcn
    VAL_LOSSES["GCN-LSTM"] = va_h_gcn
    RESULTS["GCN-LSTM"] = {
        **eval_metrics(y_te, pred_gcn),
        "Train_Time_s": tr_time_gcn,
        "Infer_Time_s": inf_time_gcn,
        "Params": sum(p.numel() for p in gcn_final.parameters()),
        "FileSize_MB": 0.0
    }
    PREDICTIONS["GCN-LSTM"] = pred_gcn
    print(f"-> GCN-LSTM : MAE = {RESULTS['GCN-LSTM']['MAE']:.3f} mph, RMSE = {RESULTS['GCN-LSTM']['RMSE']:.3f} mph, R² = {RESULTS['GCN-LSTM']['R2']:.3f}", flush=True)

    # 8.3 Ablation study : GCN-LSTM sans arêtes (A = 0)
    print("\n[Étude d'Ablation] Entraînement de GCN-LSTM sans arêtes (A = 0)...", flush=True)
    set_seed(SEED)
    gcn_no_edges = GCNLSTM(A_identity, hidden_dim=best_gcn_hp["hidden_dim"], gcn_dim=best_gcn_hp["gcn_dim"])
    t0 = time.time()
    _, tr_h_gcn0, va_h_gcn0 = train_nn(gcn_no_edges, lr=best_gcn_hp["lr"], max_epochs=20, patience=4, verbose=True)
    tr_time_gcn0 = time.time() - t0

    t0 = time.time()
    pred_gcn0 = predict_nn(gcn_no_edges, test_loader)
    inf_time_gcn0 = time.time() - t0

    TRAIN_LOSSES["GCN-LSTM (sans arêtes)"] = tr_h_gcn0
    VAL_LOSSES["GCN-LSTM (sans arêtes)"] = va_h_gcn0
    RESULTS["GCN-LSTM (sans arêtes)"] = {
        **eval_metrics(y_te, pred_gcn0),
        "Train_Time_s": tr_time_gcn0,
        "Infer_Time_s": inf_time_gcn0,
        "Params": sum(p.numel() for p in gcn_no_edges.parameters()),
        "FileSize_MB": 0.0
    }
    PREDICTIONS["GCN-LSTM (sans arêtes)"] = pred_gcn0
    print(f"-> GCN-LSTM (sans arêtes) : MAE = {RESULTS['GCN-LSTM (sans arêtes)']['MAE']:.3f} mph, RMSE = {RESULTS['GCN-LSTM (sans arêtes)']['RMSE']:.3f} mph, R² = {RESULTS['GCN-LSTM (sans arêtes)']['R2']:.3f}", flush=True)

    # --- Sauvegarde des modèles PyTorch ---
    lstm_pt_path = os.path.join(MODELS_DIR, "sensor_lstm.pt")
    gcn_pt_path = os.path.join(MODELS_DIR, "gcn_lstm.pt")
    gcn0_pt_path = os.path.join(MODELS_DIR, "gcn_lstm_no_edges.pt")
    torch.save(lstm_final.state_dict(), lstm_pt_path)
    torch.save(gcn_final.state_dict(), gcn_pt_path)
    torch.save(gcn_no_edges.state_dict(), gcn0_pt_path)
    np.save(os.path.join(MODELS_DIR, "adjacency_matrix.npy"), A_norm)
    np.save(os.path.join(MODELS_DIR, "norm_stats.npy"), np.array([mu_train, std_train], dtype=np.float32))
    np.save(os.path.join(MODELS_DIR, "selected_sensor_indices.npy"), sensor_indices)

    RESULTS["LSTM par capteur"]["FileSize_MB"] = os.path.getsize(lstm_pt_path) / (1024 * 1024)
    RESULTS["GCN-LSTM"]["FileSize_MB"] = os.path.getsize(gcn_pt_path) / (1024 * 1024)
    RESULTS["GCN-LSTM (sans arêtes)"]["FileSize_MB"] = os.path.getsize(gcn0_pt_path) / (1024 * 1024)

    # --- 9. Tableau comparatif final ---
    print("\n================== TABLEAU COMPARATIF DES RÉSULTATS (TEST) ==================", flush=True)
    df_res = pd.DataFrame(RESULTS).T[["MAE", "RMSE", "R2", "Train_Time_s", "Infer_Time_s", "Params", "FileSize_MB"]]
    df_res["Latence_ms_par_echantillon"] = (df_res["Infer_Time_s"] / len(y_te)) * 1000.0
    df_res = df_res.sort_values("RMSE")
    print(df_res.round(3).to_string(), flush=True)
    df_res.round(3).to_csv("resultats_comparatifs.csv")
    print("Tableau comparatif enregistré sous 'resultats_comparatifs.csv'", flush=True)

    # --- 10. Courbes d'apprentissage ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    model_keys = ["LSTM par capteur", "GCN-LSTM", "GCN-LSTM (sans arêtes)"]
    colors = ["darkorange", "crimson", "purple"]
    for ax, key, col in zip(axes, model_keys, colors):
        ax.plot(TRAIN_LOSSES[key], label="Train MSE", color=col, lw=2)
        ax.plot(VAL_LOSSES[key], label="Val MSE", color="navy", lw=2, linestyle="--")
        ax.set_title(key, fontsize=11, fontweight='bold')
        ax.set_xlabel("Époque")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
    axes[0].set_ylabel("MSE (normalisée)")
    plt.suptitle(f"Courbes d'apprentissage et de convergence — {N_SENSORS} capteurs (Train vs Validation)", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/learning_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Courbes d'apprentissage enregistrées sous '{FIG_DIR}/learning_curves.png'", flush=True)

    # --- 11. Analyse détaillée des erreurs ---
    print("\n================ 11. ANALYSE DÉTAILLÉE DES ERREURS ================", flush=True)
    y_true_real = y_te * std_train + mu_train

    # 11.1 MAE par capteur
    sensor_maes = {}
    for m_name, p_norm in PREDICTIONS.items():
        p_real = p_norm * std_train + mu_train
        sensor_maes[m_name] = np.abs(p_real - y_true_real).mean(axis=0)
    df_sensor_mae = pd.DataFrame(sensor_maes)

    plt.figure(figsize=(14, 4.8))
    plt.plot(df_sensor_mae["Baseline Persistance"], label="Baseline Persistance", color="gray", linestyle=":", lw=1.5)
    plt.plot(df_sensor_mae["Random Forest"], label="Random Forest", color="forestgreen", lw=1.5)
    plt.plot(df_sensor_mae["LSTM par capteur"], label="LSTM par capteur", color="darkorange", lw=1.8)
    plt.plot(df_sensor_mae["GCN-LSTM (sans arêtes)"], label="GCN-LSTM (sans arêtes)", color="purple", linestyle="--", lw=1.8)
    plt.plot(df_sensor_mae["GCN-LSTM"], label="GCN-LSTM (Complet)", color="crimson", lw=2.2)
    plt.title(f"Distribution spatiale des erreurs : MAE par capteur ({N_SENSORS} capteurs)", fontsize=12, fontweight='bold')
    plt.xlabel("Indice du capteur")
    plt.ylabel("MAE (mph)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/mae_per_sensor.png", dpi=150)
    plt.close()
    print(f"MAE par capteur enregistré sous '{FIG_DIR}/mae_per_sensor.png'", flush=True)

    better = int((df_sensor_mae["GCN-LSTM"] < df_sensor_mae["LSTM par capteur"]).sum())
    print(f"Capteurs où GCN-LSTM surpasse le LSTM par capteur : {better}/{N_SENSORS} ({better/N_SENSORS*100:.1f}%)", flush=True)

    # 11.2 Profil temporel Observé vs Prédit (sur le capteur médian d'erreur)
    med_s = int(np.argsort(df_sensor_mae["GCN-LSTM"].values)[N_SENSORS // 2])
    slice_2d = slice(120, 120 + 288 * 2)  # 2 jours continus de test (576 pas de 5 min)

    plt.figure(figsize=(15, 4.5))
    plt.plot(y_true_real[slice_2d, med_s], label="Observé (Vérité terrain)", color="black", lw=2)
    plt.plot((PREDICTIONS["Random Forest"] * std_train + mu_train)[slice_2d, med_s],
             label="Random Forest", color="forestgreen", alpha=0.7, lw=1.5)
    plt.plot((PREDICTIONS["LSTM par capteur"] * std_train + mu_train)[slice_2d, med_s],
             label="LSTM par capteur", color="darkorange", alpha=0.8, lw=1.8)
    plt.plot((PREDICTIONS["GCN-LSTM"] * std_train + mu_train)[slice_2d, med_s],
             label="GCN-LSTM", color="crimson", alpha=0.9, lw=2.0)
    plt.title(f"Suivi temporel Réel vs Prédit — Capteur n°{med_s} sur {N_SENSORS} (2 jours continus, test)", fontsize=12, fontweight='bold')
    plt.xlabel("Pas de temps (5 min)")
    plt.ylabel("Vitesse (mph)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/observed_vs_predicted.png", dpi=150)
    plt.close()
    print(f"Profil Observé vs Prédit enregistré sous '{FIG_DIR}/observed_vs_predicted.png'", flush=True)

    # 11.3 Erreurs par régime de circulation
    bins = [-1, 40, 55, 150]
    labels = ["Congestionné (< 40 mph)", "Dense (40 - 55 mph)", "Fluide (> 55 mph)"]
    traffic_regimes = pd.cut(pd.Series(y_true_real.ravel()), bins=bins, labels=labels)

    regime_records = {}
    for m_name, p_norm in PREDICTIONS.items():
        p_real = p_norm * std_train + mu_train
        abs_err = np.abs(p_real - y_true_real).ravel()
        regime_records[m_name] = pd.Series(abs_err).groupby(traffic_regimes, observed=False).mean()

    df_reg = pd.DataFrame(regime_records).T[["Congestionné (< 40 mph)", "Dense (40 - 55 mph)", "Fluide (> 55 mph)"]]
    print("\n--- MAE par régime de circulation (mph) ---", flush=True)
    print(df_reg.round(2).to_string(), flush=True)
    df_reg.round(2).to_csv("erreurs_par_regime.csv")

    ax = df_reg.plot(kind="bar", figsize=(12, 5), colormap="viridis", width=0.75, edgecolor='black')
    plt.title(f"Erreur MAE (mph) selon le régime de congestion routière ({N_SENSORS} capteurs)", fontsize=12, fontweight='bold')
    plt.ylabel("MAE (mph)")
    plt.xlabel("Algorithme")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.legend(title="Régime de trafic")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/regimes_error.png", dpi=150)
    plt.close()
    print(f"Graphique régimes de trafic enregistré sous '{FIG_DIR}/regimes_error.png'", flush=True)

    # --- 12. Évaluation multi-graines optionnelle (Graines 42, 43, 44) ---
    print("\n================ 12. ÉVALUATION MULTI-GRAINES (42, 43, 44) ================", flush=True)
    SEEDS_LIST = [42, 43, 44]
    multi_records = []

    for s_val in SEEDS_LIST:
        print(f"\n--- Exécution avec Graine {s_val} ---", flush=True)
        # Modèle 1 : LSTM
        set_seed(s_val)
        m_lstm_s = SensorLSTM(hidden_dim=best_lstm_hp["hidden_dim"])
        train_nn(m_lstm_s, lr=best_lstm_hp["lr"], max_epochs=15, patience=3)
        p_lstm_s = predict_nn(m_lstm_s, test_loader)
        met_lstm = eval_metrics(y_te, p_lstm_s)
        multi_records.append({"Modèle": "LSTM par capteur", "Seed": s_val, **met_lstm})

        # Modèle 2 : GCN-LSTM
        set_seed(s_val)
        m_gcn_s = GCNLSTM(A_norm, hidden_dim=best_gcn_hp["hidden_dim"], gcn_dim=best_gcn_hp["gcn_dim"])
        train_nn(m_gcn_s, lr=best_gcn_hp["lr"], max_epochs=15, patience=3)
        p_gcn_s = predict_nn(m_gcn_s, test_loader)
        met_gcn = eval_metrics(y_te, p_gcn_s)
        multi_records.append({"Modèle": "GCN-LSTM", "Seed": s_val, **met_gcn})

        # Modèle 3 : GCN-LSTM sans arêtes
        set_seed(s_val)
        m_gcn0_s = GCNLSTM(A_identity, hidden_dim=best_gcn_hp["hidden_dim"], gcn_dim=best_gcn_hp["gcn_dim"])
        train_nn(m_gcn0_s, lr=best_gcn_hp["lr"], max_epochs=15, patience=3)
        p_gcn0_s = predict_nn(m_gcn0_s, test_loader)
        met_gcn0 = eval_metrics(y_te, p_gcn0_s)
        multi_records.append({"Modèle": "GCN-LSTM (sans arêtes)", "Seed": s_val, **met_gcn0})

    df_multi_raw = pd.DataFrame(multi_records)
    df_multi_agg = df_multi_raw.groupby("Modèle")[["MAE", "RMSE", "R2"]].agg(["mean", "std"])
    print("\n--- Résultats Multi-Graines (Moyenne ± Écart-type) ---", flush=True)
    print(df_multi_agg.round(4).to_string(), flush=True)

    # Format plat pour CSV et JSON
    df_multi_flat = df_multi_agg.copy()
    df_multi_flat.columns = [f"{col[0]}_{col[1]}" for col in df_multi_agg.columns]
    df_multi_flat.round(4).to_csv("resultats_multi_graines.csv")
    print("Tableau multi-graines enregistré sous 'resultats_multi_graines.csv'", flush=True)

    # --- 13. Métadonnées & Résumé d'exécution ---
    print("\n================ 13. SAUVEGARDE DES MÉTADONNÉES ================", flush=True)
    summary_meta = {
        "N_SENSORS": N_SENSORS,
        "IN_LEN": IN_LEN,
        "HORIZON": HORIZON,
        "mu_train": mu_train,
        "std_train": std_train,
        "best_lstm_params": best_lstm_hp,
        "best_gcn_params": best_gcn_hp,
        "best_rf_params": best_rf_params,
        "test_metrics": df_res.round(4).to_dict(orient="index"),
        "multi_seed_metrics": df_multi_flat.round(4).to_dict(orient="index"),
        "health_checks": {
            "gcn_lstm_train_mse_final": float(TRAIN_LOSSES["GCN-LSTM"][-1]),
            "gcn_lstm_val_mse_final": float(VAL_LOSSES["GCN-LSTM"][-1]),
            "gcn_lstm_test_mae": float(RESULTS["GCN-LSTM"]["MAE"]),
            "persistence_test_mae": float(RESULTS["Baseline Persistance"]["MAE"]),
            "n_sensors_verified": int(N_SENSORS)
        }
    }
    with open(os.path.join(MODELS_DIR, "pipeline_summary.json"), "w") as f:
        json.dump(summary_meta, f, indent=4)

    print("Tous les modèles et métadonnées ont été sauvegardés avec succès dans 'models_saved/'.", flush=True)
    print("\n>>> Pipeline expérimental complet terminé avec succès ! <<<", flush=True)


if __name__ == '__main__':
    main()
