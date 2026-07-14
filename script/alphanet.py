import warnings
from pathlib import Path
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')
import os
import pandas as pd
import torch
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader, random_split
import argparse
from utils import ROOT_DIR, get_trade_calender
from numpy.lib.stride_tricks import sliding_window_view
import clickhouse_connect
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import akshare as ak
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def _build_client():
    load_dotenv()

    client = clickhouse_connect.get_client(
        host=os.getenv('CH_HOST'),
        port=os.getenv('CH_PORT'),
        username=os.getenv('CH_USERNAME'),
        password=os.getenv('CH_PASSWORD'),
        database=os.getenv('CH_DB'),
        compress=False,
        connect_timeout=60,
        send_receive_timeout=6000
    )

    return client


def _get_raw_data(start_date, end_date):
    client = _build_client()
    _df = client.query_df(f"""
                SELECT * FROM quant.v_stock_daily_online 
                WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY trade_date, stock_code
            """)

    return _df


def process_raw_data(data_df: pd.DataFrame, calender_df, window_size=30, future_size=10, keep_time_series=False, normalize=False):
    codes = data_df['stock_code'].unique()
    grid = pd.MultiIndex.from_product(
        [codes, calender_df['trade_date']],
        names=['stock_code', 'trade_date']
    ).to_frame(index=False)

    full_df = grid.merge(data_df, how='left', on=['trade_date', 'stock_code'])

    X_lst = []
    y_lst = []

    feature_cols = ['open', 'close', 'high', 'low', 'volume', 'amount', 'pct_chg', 'turnover_rate']

    for stock_code, single_stock_df in full_df.groupby('stock_code'):
        total = window_size + future_size
        windows = sliding_window_view(single_stock_df[feature_cols].values.astype(float), window_shape=total, axis=0)
        X_lst.append(windows[..., :window_size])
        close_idx = feature_cols.index('close')
        y_lst.append(windows[:, close_idx, -1] / windows[:, close_idx, -future_size] - 1)

    X_lst = np.array(X_lst).swapaxes(0, 1)  # time, stock, col, feat_time
    y_lst = np.array(y_lst).swapaxes(0, 1)  # time, stock

    y_zscore = (y_lst - np.nanmean(y_lst, axis=1, keepdims=True)) / np.nanstd(y_lst, axis=1, keepdims=True)

    X_flat = X_lst.reshape(-1, len(feature_cols), window_size)
    y_flat = y_zscore.reshape(-1)

    x_valid = ~np.any(np.isnan(X_flat), axis=(1, 2))
    y_valid = ~np.isnan(y_flat)
    mask = x_valid & y_valid

    X_clean = X_flat[mask]
    y_clean = y_flat[mask]

    if keep_time_series:
        if normalize:
            return X_lst, y_lst
        else:
            return X_lst, y_zscore
    else:
        return X_clean, y_clean


class OHLCVDataset(Dataset):
    def __init__(self, start_date, end_date):
        self.raw_df = _get_raw_data(start_date, end_date)
        self.trade_calender = get_trade_calender(start_date, end_date)
        self.X, self.y = process_raw_data(self.raw_df, self.trade_calender)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TSOperators:
    @staticmethod
    def _get_rolling_window(x, d, stride):
        return x.unfold(-1, d, stride)

    @staticmethod
    def ts_sum(x, d, stride=10):
        return TSOperators._get_rolling_window(x, d, stride).sum(-1)

    @staticmethod
    def ts_mean(x, d, stride=10):
        return TSOperators._get_rolling_window(x, d, stride).mean(-1)

    @staticmethod
    def ts_stddev(x, d, stride=10):
        return TSOperators._get_rolling_window(x, d, stride).std(unbiased=False, dim=-1)

    @staticmethod
    def ts_max(x, d, stride=10):
        return TSOperators._get_rolling_window(x, d, stride).max(-1)[0]

    @staticmethod
    def ts_min(x, d, stride=10):
        return TSOperators._get_rolling_window(x, d, stride).min(-1)[0]

    @staticmethod
    def ts_return(x, d, stride=10):
        windows = TSOperators._get_rolling_window(x, d, stride)
        first = windows[..., 0]
        last  = windows[..., -1]
        return (last - first) / (first + 1e-6)

    @staticmethod
    def ts_decaylinear(x, d, stride=10):
        weights = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
        weights = weights / weights.sum()
        window = TSOperators._get_rolling_window(x, d, stride)
        return (window * weights).sum(-1)

    @staticmethod
    def ts_zscore(x, d, stride=10):
        window = TSOperators._get_rolling_window(x, d, stride)
        mean = window.mean(dim=-1)
        std  = window.std(unbiased=False, dim=-1) + 1e-6
        return mean / std

    @staticmethod
    def ts_cov(x, d, stride=10):
        v = TSOperators._get_rolling_window(x, d, stride)
        v = v - v.mean(dim=-1, keepdim=True)
        B, C, T_out, D = v.shape

        v = v.permute(0, 2, 1, 3)
        v_flat = v.reshape(B * T_out, C, D)

        cov_mat = torch.bmm(v_flat, v_flat.transpose(1, 2)) / D

        row_idx, col_idx = torch.triu_indices(C, C, offset=1, device=x.device)
        pairwise = cov_mat[:, row_idx, col_idx]

        pairwise = pairwise.reshape(B, T_out, -1).permute(0, 2, 1)
        return pairwise

    @staticmethod
    def ts_corr(x, d, stride=10):
        v = TSOperators._get_rolling_window(x, d, stride)
        v = v - v.mean(dim=-1, keepdim=True)
        B, C, T_out, D = v.shape

        v = v.permute(0, 2, 1, 3)
        v_flat = v.reshape(B * T_out, C, D)

        conv = torch.bmm(v_flat, v_flat.transpose(1, 2)) / D
        std_flat = torch.sqrt((v_flat ** 2).sum(dim=-1) / D + 1e-8)
        std_mat = std_flat.unsqueeze(2) * std_flat.unsqueeze(1)
        corr_mat = conv / std_mat

        row_idx, col_idx = torch.triu_indices(C, C, offset=1, device=x.device)
        pairwise = corr_mat[:, row_idx, col_idx]

        pairwise = pairwise.reshape(B, T_out, -1).permute(0, 2, 1)
        return pairwise


class AlphaLayer(nn.Module):
    def __init__(self, op_name, d, num_channels, stride=10):
        super().__init__()
        self.op_name = op_name
        self.d = d
        self.stride = stride
        self.bn = nn.BatchNorm1d(num_channels)

    def forward(self, x):
        op_func = getattr(TSOperators, self.op_name)
        x = op_func(x, self.d, self.stride)
        x = self.bn(x)
        return x


class PoolLayer(nn.Module):
    def __init__(self, pool_type, d, num_channels, stride=3):
        super().__init__()
        self.pool_type = pool_type
        self.d = d
        self.stride = stride
        self.bn = nn.BatchNorm1d(num_channels)

    def forward(self, x):
        if self.pool_type == 'mean':
            x = nn.functional.avg_pool1d(x, self.d, stride=self.stride)
        elif self.pool_type == 'max':
            x = nn.functional.max_pool1d(x, self.d, stride=self.stride)
        elif self.pool_type == 'min':
            x = -nn.functional.max_pool1d(-x, self.d, stride=self.stride)
        x = self.bn(x)
        return x


class AlphaNet(nn.Module):
    def __init__(self, input_channels=8, op_d=10, hidden_dim=30,
                 window_size=30, op_stride=10, pool_d=3, pool_stride=3):
        super().__init__()
        pair_channels = input_channels * (input_channels - 1) // 2

        self.uni_ops = nn.ModuleDict({
            'ts_stddev': AlphaLayer('ts_stddev', op_d, input_channels, op_stride),
            'ts_zscore': AlphaLayer('ts_zscore', op_d, input_channels, op_stride),
            'ts_return': AlphaLayer('ts_return', op_d, input_channels, op_stride),
            'ts_decaylinear': AlphaLayer('ts_decaylinear', op_d, input_channels, op_stride),
            'ts_mean': AlphaLayer('ts_mean', op_d, input_channels, op_stride),
        })

        self.bi_ops = nn.ModuleDict({
            'ts_corr': AlphaLayer('ts_corr', op_d, pair_channels, op_stride),
            'ts_cov': AlphaLayer('ts_cov', op_d, pair_channels, op_stride),
        })

        self.pools_uni = nn.ModuleDict({
            name: nn.ModuleList([
                PoolLayer('mean', pool_d, input_channels, pool_stride),
                PoolLayer('max', pool_d, input_channels, pool_stride),
                PoolLayer('min', pool_d, input_channels, pool_stride),
            ]) for name in self.uni_ops.keys()
        })

        self.pools_bi = nn.ModuleDict({
            name: nn.ModuleList([
                PoolLayer('mean', pool_d, pair_channels, pool_stride),
                PoolLayer('max', pool_d, pair_channels, pool_stride),
                PoolLayer('min', pool_d, pair_channels, pool_stride),
            ]) for name in self.bi_ops.keys()
        })

        T_extract = (window_size - op_d) // op_stride + 1
        T_pool = (T_extract - pool_d) // pool_stride + 1
        uni_extract_dim = T_extract * input_channels
        bi_extract_dim = T_extract * pair_channels
        uni_pool_dim = T_pool * input_channels
        bi_pool_dim = T_pool * pair_channels
        total_dim = (5 * (uni_extract_dim + 3 * uni_pool_dim) +
                     2 * (bi_extract_dim + 3 * bi_pool_dim))

        self.flatter = nn.Flatten(start_dim=-2, end_dim=-1)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=total_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=hidden_dim, out_features=1),
        )

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                if layer == self.mlp[0]:
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        feat_list = []
        for name, op in self.uni_ops.items():
            out = op(x)
            feat_list.append(self.flatter(out))
            for pool in self.pools_uni[name]:
                feat_list.append(self.flatter(pool(out)))
        for name, op in self.bi_ops.items():
            out = op(x)
            feat_list.append(self.flatter(out))
            for pool in self.pools_bi[name]:
                feat_list.append(self.flatter(pool(out)))
        combined = torch.cat(feat_list, dim=-1)
        return self.mlp(combined)


def train_on_period(epochs, batch_size, lr, start_date, end_date, device):
    train_size = 0.8
    val_size = 0.2

    full_dataset = OHLCVDataset(start_date, end_date)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model = AlphaNet(input_channels=8, op_d=10, hidden_dim=30, window_size=30).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    no_improve = 0
    patience = 5

    model_path =  ROOT_DIR / '1-weights'/ 'best_alpha_net.pth'

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_train = 0

        for X, y in tqdm(train_loader, desc=f"Epoch {epoch + 1} Trained", leave=False):
            X = X.to(device).float()
            y = y.to(device).float()
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y.unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            n_train += X.size(0)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for X, y in tqdm(val_loader, desc="Validating", leave=False):
                X = X.to(device).float()
                y = y.to(device).float()
                pred = model(X)
                loss = criterion(pred, y.unsqueeze(1))
                val_loss += loss.item() * X.size(0)
                n_val += X.size(0)
        val_loss /= n_val

        scheduler.step(val_loss)
        tqdm.write(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break


def backtest_on_period(start_date, end_date, device):
    dataset = OHLCVDataset(start_date, end_date)
    X, y = process_raw_data(dataset.raw_df, dataset.trade_calender, keep_time_series=True)

    X = X[::10, ...]
    y = y[::10, ...]

    model = AlphaNet(input_channels=8, op_d=10, hidden_dim=30, window_size=30).to(device)

    model_path = ROOT_DIR / '1-weights' / 'best_alpha_net.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    ic_daily = []
    for X_cs, y_cs in zip(X, y):
        cs_valid = ~np.any(np.isnan(X_cs), axis=(1, 2))
        y_valid = ~np.isnan(y_cs)

        valid = cs_valid & y_valid

        X_tensor = torch.tensor(X_cs[valid], dtype=torch.float32, device=device)

        pred = model(X_tensor)
        pred_np = pred.detach().cpu().numpy().flatten()
        label = y_cs[valid]

        ic, _ = spearmanr(pred_np, label)
        ic_daily.append(ic)

    ic_daily = np.array(ic_daily)
    ic_mean = np.mean(ic_daily)
    ic_std = np.std(ic_daily)
    icir = ic_mean / ic_std if ic_std != 0 else 0.0

    return ic_daily, icir


def plot_ic_result(ic_daily, icir):
    plt.figure(figsize=(12, 5))
    plt.plot(ic_daily, marker='o', linestyle='-', markersize=3, linewidth=0.8, label='Daily Rank IC')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.axhline(y=np.mean(ic_daily), color='red', linestyle='--', linewidth=1.2,
                label=f'Mean IC = {np.mean(ic_daily):.4f}')
    plt.title(f'Daily Rank IC Sequence (ICIR = {icir:.3f})')
    plt.xlabel('Trading Day Index')
    plt.ylabel('Rank IC')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT_DIR/ '2-result' / 'ic_daily.png', dpi=150, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--start_date', '-start', type=str, default='2010-01-01')
    parser.add_argument('--end_date', '-end', type=str, default='2015-01-01')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.mode == 'train':
        train_on_period(args.epoch, args.batch_size, args.learning_rate,
                        start_date=args.start_date, end_date=args.end_date, device=device)
    elif args.mode == 'val':
        ic_daily, icir= backtest_on_period(start_date=args.start_date, end_date=args.end_date,
                                           device=device)
        plot_ic_result(ic_daily, icir)