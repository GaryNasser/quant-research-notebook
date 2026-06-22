import os
import argparse
import clickhouse_connect
from datetime import datetime
from dotenv import load_dotenv
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from utils import ROOT_DIR
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict


def build_client():
    load_dotenv()
    return clickhouse_connect.get_client(
        host=os.getenv('CH_HOST'),
        port=os.getenv('CH_PORT'),
        username=os.getenv('CH_USERNAME'),
        password=os.getenv('CH_PASSWORD'),
        database=os.getenv('CH_DB'),
        compress=True,
        connect_timeout=60,
        send_receive_timeout=300
    )


def valid_date(date_string):
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: '{date_string}'. Expected YYYY-MM-DD"
        )
    return date_string


def parse_args():
    parser = argparse.ArgumentParser(description='AlphaNet training script')
    parser.add_argument('--train_start_date', type=valid_date, default='2019-01-01')
    parser.add_argument('--train_end_date', type=valid_date, default='2025-01-01')
    parser.add_argument('--test_start_date', type=valid_date, default='2025-01-01')
    parser.add_argument('--test_end_date', type=valid_date, default='2026-01-01')
    parser.add_argument('--window_size', type=int, default=30)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--op_d', type=int, default=10, help='Rolling window size for operators')
    parser.add_argument('--stride', type=int, default=2, help='Stride for sliding window when building samples')
    return parser.parse_args()


def build_dataset(client, start_date, end_date, feature_columns,
                  window_size, horizon, stride, return_dates=False):
    sql = f"""
        SELECT stock_code, trade_date, {', '.join(feature_columns)}
        FROM quant.v_stock_daily_online
        WHERE trade_date >= '{start_date}' 
            AND trade_date < '{end_date}'
        ORDER BY stock_code, trade_date
    """

    result = client.query(sql)
    col_names = result.column_names
    col_idx = {n: i for i, n in enumerate(col_names)}
    stock_idx = col_idx['stock_code']
    date_idx = col_idx['trade_date']
    feat_idx = [col_idx[c] for c in feature_columns]
    close_idx = col_idx['close']

    stock_data = defaultdict(list)
    for row in result.result_set:
        stock = row[stock_idx]
        trade_date = row[date_idx]
        features = [row[i] for i in feat_idx]
        close = row[close_idx]
        stock_data[stock].append({
            'date': trade_date,
            'features': features,
            'close': close
        })

    date_samples = defaultdict(list)
    for code, series in stock_data.items():
        series.sort(key=lambda x: x['date'])
        L = len(series)
        idx = 0
        while idx + window_size + horizon <= L:
            window = series[idx: idx + window_size + horizon]
            feat = [d['features'] for d in window[:window_size]]
            cur_price = window[window_size - 1]['close']
            fut_price = window[-1]['close']
            raw_target = (fut_price - cur_price) / (cur_price + 1e-9)
            sample_date = window[-1]['date']
            date_samples[sample_date].append(
                (torch.tensor(feat, dtype=torch.float32), raw_target)
            )
            idx += stride

    all_samples = []
    all_dates = [] if return_dates else None
    for date in sorted(date_samples.keys()):
        items = date_samples[date]
        raw_targets = [t for _, t in items]
        n = len(raw_targets)
        mean_t = sum(raw_targets) / n
        var_t = sum((v - mean_t) ** 2 for v in raw_targets) / n
        std_t = var_t ** 0.5 + 1e-9

        for feat, raw in items:
            norm_target = (raw - mean_t) / std_t
            all_samples.append((feat, torch.tensor(norm_target, dtype=torch.float32)))
            if return_dates:
                all_dates.append(date)

    class _StockDailyDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    dataset = _StockDailyDataset(all_samples)
    if return_dates:
        return dataset, all_dates
    return dataset


class TSOperators:
    @staticmethod
    def _get_rolling_window(x, d):
        # x: (B, T, C) -> (B, W, d, C)
        return x.unfold(1, d, d).transpose(-1, -2)

    @staticmethod
    def ts_corr(x, d):
        rw = TSOperators._get_rolling_window(x, d)
        B, W, D, C = rw.shape
        rw = rw.transpose(-1, -2)
        rw = rw.reshape(B * W, C, D)
        rw_flat = rw - rw.mean(dim=-1, keepdim=True)
        cov = torch.bmm(rw_flat, rw_flat.transpose(-1, -2)) / D
        std = torch.sqrt(torch.clamp((rw_flat ** 2).sum(dim=-1), min=1e-8))
        std_prod = std.unsqueeze(2) * std.unsqueeze(1)
        corr = cov / (std_prod + 1e-6)
        idx_row, idx_col = torch.triu_indices(C, C, offset=1, device=x.device)
        pairwise = corr[:, idx_row, idx_col]
        return pairwise.reshape(B, W, -1)

    @staticmethod
    def ts_cov(x, d):
        rw = TSOperators._get_rolling_window(x, d)
        B, W, D, C = rw.shape
        rw = rw.transpose(-1, -2)
        rw = rw.reshape(B * W, C, D)
        rw_flat = rw - rw.mean(dim=-1, keepdim=True)
        cov = torch.bmm(rw_flat, rw_flat.transpose(-1, -2)) / D
        idx_row, idx_col = torch.triu_indices(C, C, offset=1, device=x.device)
        pairwise = cov[:, idx_row, idx_col]
        return pairwise.reshape(B, W, -1)

    @staticmethod
    def ts_stddev(x, d):
        return TSOperators._get_rolling_window(x, d).std(dim=-2)

    @staticmethod
    def ts_zscore(x, d):
        rw = TSOperators._get_rolling_window(x, d)
        return rw.mean(dim=-2) / (rw.std(dim=-2) + 1e-8)

    @staticmethod
    def ts_return(x, d):
        return (x[:, d:] - x[:, :-d]) / (x[:, :-d] + 1e-8)

    @staticmethod
    def ts_decay_linear(x, d):
        weights = torch.arange(1, d + 1, device=x.device, dtype=torch.float32)
        weights = weights / weights.sum()
        rw = TSOperators._get_rolling_window(x, d)
        return (weights[..., None] * rw).sum(dim=-2)

    @staticmethod
    def ts_min(x, d):
        return TSOperators._get_rolling_window(x, d).min(dim=-2).values

    @staticmethod
    def ts_max(x, d):
        return TSOperators._get_rolling_window(x, d).max(dim=-2).values

    @staticmethod
    def ts_sum(x, d):
        return TSOperators._get_rolling_window(x, d).sum(dim=-2)


class AlphaLayer(nn.Module):
    def __init__(self, op_name, d, num_channels):
        super().__init__()
        self.op_name = op_name
        self.d = d
        self.bn = nn.BatchNorm1d(num_channels)

    def forward(self, x):
        op_func = getattr(TSOperators, self.op_name)
        x = op_func(x, self.d)
        x = x.transpose(-1, -2)
        x = self.bn(x)
        x = x.transpose(-1, -2)
        return x


class AlphaNet(nn.Module):
    def __init__(self, in_channels=8, op_d=10, hidden_d=30):
        super().__init__()
        pair_channels = in_channels * (in_channels - 1) // 2
        self.op_d = op_d
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)

        self.mlp = nn.Sequential(
            nn.Linear(424, hidden_d),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_d, 1)
        )
        self.ops = nn.ModuleDict({
            'ts_corr':         AlphaLayer('ts_corr', op_d, pair_channels),
            'ts_cov':          AlphaLayer('ts_cov', op_d, pair_channels),
            'ts_stddev':       AlphaLayer('ts_stddev', op_d, in_channels),
            'ts_zscore':       AlphaLayer('ts_zscore', op_d, in_channels),
            'ts_return':       AlphaLayer('ts_return', op_d, in_channels),
            'ts_decay_linear': AlphaLayer('ts_decay_linear', op_d, in_channels),
            'ts_max':          AlphaLayer('ts_max', op_d, in_channels),
        })

    def forward(self, x):
        extracted = []
        for layer in self.ops.values():
            extracted.append(self.flatten(layer(x)))
        return self.mlp(torch.cat(extracted, dim=-1))


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    count = 0
    for batch_x, batch_y in tqdm(loader, desc='Training', leave=True, dynamic_ncols=True):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).view(-1, 1)
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
        count += batch_x.size(0)
    return total_loss / count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    count = 0
    for batch_x, batch_y in tqdm(loader, desc='Evaluating', leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).view(-1, 1)
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        total_loss += loss.item() * batch_x.size(0)
        count += batch_x.size(0)
    return total_loss / count


def compute_daily_ic(predictions, targets, dates, min_stocks=20):
    daily_ics = defaultdict(list)
    for p, t, d in zip(predictions, targets, dates):
        daily_ics[d].append((p, t))

    ic_values = []
    for date in sorted(daily_ics.keys()):
        items = daily_ics[date]
        if len(items) < min_stocks:
            continue
        day_preds = np.array([it[0] for it in items])
        day_trues = np.array([it[1] for it in items])
        ic, _ = spearmanr(day_preds, day_trues)
        if not np.isnan(ic):
            ic_values.append(ic)

    if not ic_values:
        return {}

    ic_values = np.array(ic_values)
    mean_ic = np.mean(ic_values)
    std_ic = np.std(ic_values)
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    pos_ratio = np.mean(ic_values > 0)

    return {
        'mean_IC': mean_ic,
        'std_IC': std_ic,
        'ICIR': icir,
        'IC>0_ratio': pos_ratio,
        'n_days': len(ic_values)
    }


if __name__ == '__main__':
    args = parse_args()

    feature_columns = ['open', 'high', 'low', 'close', 'volume',
                       'amount', 'pct_chg', 'turnover_rate']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    client = build_client()

    train_dataset = build_dataset(
        client, args.train_start_date, args.train_end_date,
        feature_columns,
        window_size=args.window_size,
        horizon=args.horizon,
        stride=args.stride,
        return_dates=False
    )

    test_dataset, test_dates = build_dataset(
        client, args.test_start_date, args.test_end_date,
        feature_columns,
        window_size=args.window_size,
        horizon=args.horizon,
        stride=args.stride,
        return_dates=True
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = AlphaNet(in_channels=len(feature_columns), op_d=args.op_d, hidden_d=30).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    best_val_loss = float('inf')
    model_save_path = ROOT_DIR / "1-权重" / f"alpha_best_model_trained_on_{args.train_start_date}_{args.train_end_date}.pth"
    model_save_path.parent.mkdir(exist_ok=True)

    for epoch in range(args.epoch):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_loss)

        tqdm.write(f"Epoch {epoch+1:3d}/{args.epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            tqdm.write(f"  -> Best model saved (val_loss={val_loss:.6f})")

    print("\nLoading best model for IC evaluation...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc='IC Prediction', leave=False):
            batch_x = batch_x.to(device)
            pred = model(batch_x).cpu().numpy().flatten()
            all_preds.append(pred)
            all_targets.append(batch_y.numpy().flatten())
    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    ic_stats = compute_daily_ic(predictions, targets, test_dates)
    print("\n=== IC Statistics (Test Set) ===")
    print(f"Number of valid trading days (>=20 stocks): {ic_stats.get('n_days', 0)}")
    print(f"Mean RankIC:     {ic_stats.get('mean_IC', 0):.4f}")
    print(f"Std RankIC:      {ic_stats.get('std_IC', 0):.4f}")
    print(f"ICIR:            {ic_stats.get('ICIR', 0):.4f}")
    print(f"IC > 0 ratio:    {ic_stats.get('IC>0_ratio', 0):.2%}")