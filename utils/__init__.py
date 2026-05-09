import yfinance as yf
import pandas as pd
from pathlib import Path
import baostock as bs
from contextlib import contextmanager
import sys
from io import StringIO

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "0-数据"


@contextmanager
def baostock_connection():
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    lg = bs.login()
    sys.stdout = old_stdout

    try:
        yield lg
    finally:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        bs.logout()
        sys.stdout = old_stdout

def _get_data_save_path(source: str, symbol: str, start_date, end_date, interval):
    return f"{source}-{symbol}-{start_date}-{end_date}-{interval}.csv"

def get_yahoo_data(symbols: list | str, start_date: str, end_date: str, interval: str):
    """
    获取雅虎财经数据

    Args:
        symbols: 单个股票代码(str) 或 股票代码列表(list)
        start_date: 开始日期，格式 YYYY-MM-DD
        end_date: 结束日期，格式 YYYY-MM-DD
        interval: 数据周期 (1m/5m/15m/30m/1h/1d/1wk/1mo)

    Returns:
        - str 输入: 返回 DataFrame
        - list 输入: 返回 {symbol: DataFrame}
    """
    _data_dict = {}

    is_single_symbol = True if isinstance(symbols, str) else False
    symbols = symbols if isinstance(symbols, list) else [symbols]
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        file_path = DATA_DIR / _get_data_save_path('yf', symbol, start_date, end_date, interval)

        if file_path.exists():
            _data_dict[symbol] = pd.read_csv(file_path, index_col=0)
        else:
            _df = yf.download(symbol, start_date, end_date, interval, auto_adjust=False)
            if isinstance(_df.columns, pd.MultiIndex):
                _df.columns = _df.columns.get_level_values(0)

            _data_dict[symbol] = _df
            _df.to_csv(file_path)

    return _data_dict[symbols[0]] if is_single_symbol and len(symbols) else _data_dict

def _convert_baostock_code(symbol: str) -> str:
    """
    转换股票代码为 baostock 格式

    Args:
        symbol: 股票代码，如 "000001", "600000"

    Returns:
        baostock 格式代码，如 "sh.000001", "sh.600000"
    """
    symbol = symbol.strip()
    if symbol.startswith('6'):
        return f"sh.{symbol}"
    elif symbol.startswith('0') or symbol.startswith('3'):
        return f"sz.{symbol}"
    else:
        return symbol


def _process_baostock_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'pctChg' in df.columns:
        df['pctChg'] = df['pctChg'] / 100.0
    else:
        df['pctChg'] = df['Close'].pct_change()

    df.dropna(subset=['pctChg'], inplace=True)

    column_map = {
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume', 'amount': 'Amount'
    }
    df.rename(columns=column_map, inplace=True)
    return df

def fetch_baostock_data(symbols: list | str, start_date: str, end_date: str, interval: str = "d"):
    """
    获取 BaoStock 财经数据

    Args:
        symbols: 单个股票代码(str) 或 股票代码列表(list)
                支持格式：纯数字代码，如 "000001", "600000"
        start_date: 开始日期，格式 YYYY-MM-DD
        end_date: 结束日期，格式 YYYY-MM-DD
        interval: 数据周期 (d/w/m/5/15/30/60)

    Returns:
        - 输入为 str 时，返回单个 DataFrame
        - 输入为 list 时，返回字典 {symbol: DataFrame}
    """

    _data_dict = {}

    is_single_symbol = isinstance(symbols, str)
    symbols = symbols if isinstance(symbols, list) else [symbols]

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with baostock_connection() as lg:
        if lg.error_code != '0':
            return None if is_single_symbol else {}

        fields = "date,open,high,low,close,volume,amount,pctChg"

        for symbol in symbols:
            file_path = DATA_DIR / _get_data_save_path('bs', symbol, start_date, end_date, interval)

            if file_path.exists():
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                _data_dict[symbol] = df
                continue

            bs_code = _convert_baostock_code(symbol)

            rs = bs.query_history_k_data_plus(
                bs_code,
                fields,
                start_date=start_date,
                end_date=end_date,
                frequency=interval,
                adjustflag="2"
            )

            if rs.error_code != '0':
                continue

            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                continue

            df = pd.DataFrame(data_list, columns=rs.fields)
            df = _process_baostock_data(df)

            if df.empty:
                continue

            df.to_csv(file_path)
            _data_dict[symbol] = df

    if is_single_symbol and symbols:
        return _data_dict.get(symbols[0], pd.DataFrame())
    return _data_dict