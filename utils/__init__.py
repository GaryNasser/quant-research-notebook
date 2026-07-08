import yfinance as yf
import pandas as pd
from pathlib import Path
import baostock as bs
from contextlib import contextmanager
import sys
from io import StringIO
import akshare as ak

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "0-data"


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


def get_stock_code_list(base_date='2024-01-02', force_refresh=False, filter_exchange=False):
    """
    获取股票列表
    """
    file_path = DATA_DIR / 'stock_list.csv'

    if file_path.exists() and not force_refresh:
        stock_df = pd.read_csv(file_path)
        if filter_exchange:
            stock_df['code'] = stock_df['code'].str.extract(r'\.(\d+)')
        return stock_df['code'].tolist()
    else:
        with baostock_connection() as lg:
            if lg.error_code != '0':
                return []
            rs = bs.query_all_stock(day=base_date)
            stock_df = rs.get_data()
            stock_df = stock_df[(stock_df['code'] > 'sh.600000') & (stock_df['code'] < 'sz.399000')]
            stock_df.to_csv(file_path)
            if filter_exchange:
                stock_df['code'] = stock_df['code'].str.extract(r'\.(\d+)')
            stock_code = stock_df['code'].tolist()
            return stock_code

def get_yahoo_data(symbols: list | str, start_date: str, end_date: str, interval: str, persistent=False):
    """
    获取雅虎财经数据

    Args:
        symbols: 单个股票代码(str) 或 股票代码列表(list)
        start_date: 开始日期，格式 YYYY-MM-DD
        end_date: 结束日期，格式 YYYY-MM-DD
        interval: 数据周期 (1m/5m/15m/30m/1h/1d/1wk/1mo)
        persistent: 是否持久化数据文件

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
            if persistent:
                _df.to_csv(file_path)

    return _data_dict[symbols[0]] if is_single_symbol and len(symbols) else _data_dict


def _convert_baostock_code(symbol: str) -> str:
    """
    转换股票代码为 baostock 格式
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

    numeric_cols = [
        'open', 'high', 'low', 'close', 'volume', 'amount',
        'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM', 'turn'
    ]
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


def fetch_baostock_data(symbols: list | str, start_date: str, end_date: str, interval: str = "d", persistent=False):
    """
    Args:
        symbols: 单个股票代码(str) 或 股票代码列表(list)
                支持格式：纯数字代码，如 "000001", "600000"
        start_date: 开始日期，格式 YYYY-MM-DD
        end_date: 结束日期，格式 YYYY-MM-DD
        interval: 数据周期 (d/w/m/5/15/30/60)
        persistent: 是否持久化数据文件

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

        fields = "date,open,high,low,close,volume,amount,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,turn"

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
                adjustflag="1"
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

            if persistent:
                df.to_csv(file_path)
            _data_dict[symbol] = df

    if is_single_symbol and symbols:
        return _data_dict.get(symbols[0], pd.DataFrame())
    return _data_dict


def query_zz500_stocks():
    with baostock_connection() as lg:
        if lg.error_code != '0':
            return None

        rs = bs.query_zz500_stocks()
        if rs.error_code != '0':
            return pd.DataFrame()
        else:
            data = rs.get_data()[['code', 'code_name']]
            return data



def get_china_10_year_treasury_yield(start_year: str | int, end_year: str | int=None):
    end_year = start_year if end_year is None else end_year
    start, end = int(start_year), int(end_year)
    year_lst = [str(year) for year in range(start, end + 1)]
    yield_lst = []

    for year in year_lst:
        start_date, end_date = f"{year}0101", f"{year}1231"

        bond_china_yield_df = ak.bond_china_yield(start_date, end_date)
        rf_df = bond_china_yield_df[bond_china_yield_df['曲线名称'] == '中债国债收益率曲线'][['日期', '10年']]
        rf_df['日期'] = pd.to_datetime(rf_df['日期'])
        yield_lst.append(rf_df)

    if yield_lst is None:
        return pd.DataFrame()

    result_df = pd.concat(yield_lst, ignore_index=True)
    return result_df