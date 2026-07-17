"""
Import Level2 CSV data (from 7z archives or loose files) into ClickHouse.
"""

import os
import sys
import argparse
import logging
import clickhouse_connect
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import subprocess
import shutil
import warnings
import threading

_db_lock = threading.Lock()

logging.getLogger("clickhouse_connect").setLevel(logging.CRITICAL + 1)
warnings.filterwarnings('ignore')

_client = None
_table_prefix = "level2"
_max_workers = 8

stats_lock = Lock()


def parse_args():
    """Parse command line arguments, falling back to environment variables."""
    parser = argparse.ArgumentParser(
        description="Import Level2 CSV data into ClickHouse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --password mypassword
  %(prog)s --data-dir /data/level2 --host 10.0.0.5 --user reader
  %(prog)s --max-workers 16
        """
    )
    parser.add_argument('--data-dir', type=str,
                        default=os.getenv("DATA_DIR", "/data"),
                        help='Directory containing CSV files or 7z archives')
    parser.add_argument('--host', type=str,
                        default=os.getenv("CLICKHOUSE_HOST", "127.0.0.1"),
                        help='ClickHouse host')
    parser.add_argument('--port', type=int,
                        default=int(os.getenv("CLICKHOUSE_PORT", "8123")),
                        help='ClickHouse HTTP port')
    parser.add_argument('--user', type=str,
                        default=os.getenv("CLICKHOUSE_USER", "default"),
                        help='ClickHouse user')
    parser.add_argument('--password', type=str,
                        default=os.getenv("CLICKHOUSE_PASSWORD"),
                        help='ClickHouse password')
    parser.add_argument('--database', type=str,
                        default=os.getenv("CLICKHOUSE_DB", "default"),
                        help='ClickHouse database')
    parser.add_argument('--table-prefix', type=str,
                        default=os.getenv("CLICKHOUSE_TABLE_PREFIX", "level2"),
                        help='Table name prefix')
    parser.add_argument('--max-workers', type=int,
                        default=int(os.getenv("MAX_WORKERS", "8")),
                        help='Maximum parallel workers')
    parser.add_argument('--log-file', type=str, default='import_level2.log',
                        help='Path to log file')
    args = parser.parse_args()
    if not args.password:
        parser.error("--password is required (or set CLICKHOUSE_PASSWORD)")
    return args


def setup_logging(log_file: str):
    """Configure logging to console and file (INFO level)."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not create log file {log_file}: {e}")


def clean_dataframe(df: pd.DataFrame, table_type: str) -> pd.DataFrame:
    """
    Clean and transform raw CSV dataframe to match ClickHouse table schema.
    Supports 'snapshot', 'entrust', and 'trade' table types.
    """
    # Remove unnamed columns and completely empty rows
    unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    df = df.dropna(how='all')

    # Strip spaces from column names
    df.columns = df.columns.str.strip()

    if table_type == 'snapshot':
        # Identify the 10 ask/bid price/volume columns
        ask_price_cols = [f'申卖价{i}' for i in range(1, 11) if f'申卖价{i}' in df.columns]
        ask_volume_cols = [f'申卖量{i}' for i in range(1, 11) if f'申卖量{i}' in df.columns]
        bid_price_cols = [f'申买价{i}' for i in range(1, 11) if f'申买价{i}' in df.columns]
        bid_volume_cols = [f'申买量{i}' for i in range(1, 11) if f'申买量{i}' in df.columns]

        all_array_cols = ask_price_cols + ask_volume_cols + bid_price_cols + bid_volume_cols

        if ask_price_cols and ask_volume_cols:
            # Convert to numeric
            for col in all_array_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Build list columns
            df['ask_price'] = df[ask_price_cols].values.tolist()
            df['ask_volume'] = df[ask_volume_cols].values.tolist()
            df['bid_price'] = df[bid_price_cols].values.tolist()
            df['bid_volume'] = df[bid_volume_cols].values.tolist()

            # Drop original columns
            df = df.drop(columns=all_array_cols)

            # Replace NaN inside lists with None
            def replace_nan_with_none(lst):
                return [None if (isinstance(x, float) and np.isnan(x)) else x for x in lst]

            for col in ['ask_price', 'ask_volume', 'bid_price', 'bid_volume']:
                if col in df.columns:
                    df[col] = df[col].apply(replace_nan_with_none)

        column_mapping = {
            '万得代码': 'wind_code',
            '交易所代码': 'exchange_code',
            '自然日': 'trade_date',
            '时间': 'trade_time',
            '成交价': 'price',
            '成交量': 'volume',
            '成交额': 'turnover',
            '成交笔数': 'trade_count',
            'IOPV': 'iopv',
            '成交标志': 'trade_flag',
            'BS标志': 'bs_flag',
            '当日累计成交量': 'cumulative_volume',
            '当日成交额': 'cumulative_turnover',
            '最高价': 'high_price',
            '最低价': 'low_price',
            '开盘价': 'open_price',
            '前收盘': 'pre_close',
            '加权平均叫卖价': 'weighted_avg_ask',
            '加权平均叫买价': 'weighted_avg_bid',
            '叫卖总量': 'total_ask_volume',
            '叫买总量': 'total_bid_volume',
            '不加权指数': 'unweighted_index',
            '品种总数': 'total_securities',
            '上涨品种数': 'advancing_securities',
            '下跌品种数': 'declining_securities',
            '持平品种数': 'unchanged_securities'
        }

    elif table_type == 'entrust':
        column_mapping = {
            '万得代码': 'wind_code',
            '交易所代码': 'exchange_code',
            '自然日': 'trade_date',
            '时间': 'trade_time',
            '委托编号': 'entrust_no',
            '交易所委托号': 'exchange_entrust_no',
            '委托类型': 'entrust_type',
            '委托代码': 'entrust_code',
            '委托价格': 'entrust_price',
            '委托数量': 'entrust_volume'
        }

    elif table_type == 'trade':
        column_mapping = {
            '万得代码': 'wind_code',
            '交易所代码': 'exchange_code',
            '自然日': 'trade_date',
            '时间': 'trade_time',
            '成交编号': 'trade_no',
            '成交代码': 'trade_code',
            '委托代码': 'entrust_code',
            'BS标志': 'bs_flag',
            '成交价格': 'trade_price',
            '成交数量': 'trade_volume',
            '叫卖序号': 'ask_order_no',
            '叫买序号': 'bid_order_no'
        }
    else:
        return None

    # Rename existing columns
    existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_mapping)

    # Keep only needed columns
    keep_cols = list(existing_mapping.values())
    if table_type == 'snapshot':
        for col in ['ask_price', 'ask_volume', 'bid_price', 'bid_volume']:
            if col in df.columns:
                keep_cols.append(col)
    df = df[keep_cols]

    # Type conversion
    int_cols = {
        'trade_date', 'trade_time', 'trade_count', 'trade_flag', 'bs_flag',
        'entrust_no', 'exchange_entrust_no', 'trade_no', 'ask_order_no', 'bid_order_no',
        'total_securities', 'advancing_securities', 'declining_securities', 'unchanged_securities'
    }
    float_cols = {
        'price', 'volume', 'turnover', 'iopv',
        'cumulative_volume', 'cumulative_turnover', 'high_price', 'low_price',
        'open_price', 'pre_close', 'weighted_avg_ask', 'weighted_avg_bid',
        'total_ask_volume', 'total_bid_volume', 'unweighted_index',
        'entrust_price', 'entrust_volume', 'trade_price', 'trade_volume'
    }
    str_cols = {
        'wind_code', 'exchange_code', 'entrust_type', 'entrust_code', 'trade_code'
    }
    array_cols = {'ask_price', 'ask_volume', 'bid_price', 'bid_volume'}

    for col in df.columns:
        if col in array_cols:
            continue
        elif col in int_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        elif col in float_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Float64')
        elif col in str_cols:
            df[col] = df[col].astype(str).replace(['', 'nan', 'NaN', 'None', 'NaT'], None)
            df[col] = df[col].astype('string')
        else:
            df[col] = df[col].astype(str).replace(['', 'nan', 'NaN', 'None', 'NaT'], None)
            df[col] = df[col].astype('string')

    return df


def import_to_clickhouse(df: pd.DataFrame, table_name: str) -> int:
    """Insert a cleaned DataFrame into the corresponding ClickHouse table."""
    if _client is None or df is None or len(df) == 0:
        return 0

    with _db_lock:
        try:
            full_table = f"{_table_prefix}.{table_name}"
            _client.insert_df(full_table, df)
            return len(df)
        except Exception as e:
            logging.error(f"Insert error ({table_name}): {e}")
            return 0


def get_file_type(file_path: Path) -> str:
    """Determine data type based on filename ('snapshot', 'entrust', 'trade')."""
    file_name = file_path.stem
    if '行情' in file_name or 'snapshot' in file_name.lower():
        return 'snapshot'
    elif '委托' in file_name or 'entrust' in file_name.lower():
        return 'entrust'
    elif '成交' in file_name or 'trade' in file_name.lower():
        return 'trade'
    return None


def count_csv_files_by_type(csv_files: list) -> dict:
    """Count CSV files by table type."""
    stats = {'snapshot': 0, 'entrust': 0, 'trade': 0, 'unknown': 0}
    for file_path in csv_files:
        file_type = get_file_type(file_path)
        stats[file_type or 'unknown'] += 1
    return stats


def print_file_statistics(csv_files: list, source_desc: str = ""):
    """Print summary of CSV files by type."""
    stats = count_csv_files_by_type(csv_files)
    total = len(csv_files)
    logger = logging.getLogger(__name__)
    if source_desc:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"[Statistics] {source_desc}")
    else:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"[Statistics] File overview")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total CSV files: {total}")
    logger.info(f"  - Snapshot:  {stats['snapshot']} files")
    logger.info(f"  - Entrust:   {stats['entrust']} files")
    logger.info(f"  - Trade:     {stats['trade']} files")
    if stats['unknown'] > 0:
        logger.info(f"  - Unknown:   {stats['unknown']} files")
    logger.info(f"{'=' * 60}\n")


def process_single_file(file_path: Path) -> dict:
    """Read, clean, and import a single CSV file."""
    logger = logging.getLogger(__name__)
    file_type = get_file_type(file_path)
    if not file_type:
        return None

    try:
        df = pd.read_csv(file_path, encoding='gbk')

        # Remove duplicate header rows
        if df.shape[0] > 0:
            first_col = df.columns[0]
            df = df[df[first_col] != first_col]

        df_clean = clean_dataframe(df, file_type)

        if df_clean is not None and len(df_clean) > 0:
            rows_inserted = import_to_clickhouse(df_clean, file_type)
            return {'type': file_type, 'rows': rows_inserted, 'success': rows_inserted > 0}
        else:
            return {'type': file_type, 'rows': 0, 'success': False}
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None


def process_csv_files(csv_files: list, desc: str = "Import progress") -> dict:
    """Process a list of CSV files in parallel and aggregate statistics."""
    stats = {
        'snapshot': {'files': 0, 'rows': 0},
        'entrust': {'files': 0, 'rows': 0},
        'trade': {'files': 0, 'rows': 0}
    }

    with ThreadPoolExecutor(max_workers=_max_workers) as executor:
        futures = {executor.submit(process_single_file, fp): fp for fp in csv_files}

        with tqdm(total=len(csv_files), desc=desc, unit="file",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result and result['success']:
                    with stats_lock:
                        stats[result['type']]['files'] += 1
                        stats[result['type']]['rows'] += result['rows']
                pbar.update(1)

    return stats


def merge_stats(stats1: dict, stats2: dict) -> dict:
    """Merge two statistics dicts."""
    for key in stats1.keys():
        stats1[key]['files'] += stats2[key]['files']
        stats1[key]['rows'] += stats2[key]['rows']
    return stats1


def process_archive_files(data_dir: Path) -> dict:
    """Discover 7z archives, extract them, import CSVs, then clean up."""
    logger = logging.getLogger(__name__)
    archive_files = sorted(list(data_dir.glob('*.7z')))

    if not archive_files:
        return {'snapshot': {'files': 0, 'rows': 0},
                'entrust': {'files': 0, 'rows': 0},
                'trade': {'files': 0, 'rows': 0}}

    global_stats = {
        'snapshot': {'files': 0, 'rows': 0},
        'entrust': {'files': 0, 'rows': 0},
        'trade': {'files': 0, 'rows': 0}
    }

    logger.info(f"\nFound {len(archive_files)} archive(s)")
    logger.info(f"{'=' * 60}")

    logger.info("Scanning archive contents...")
    total_csv_count = 0
    type_counts = {'snapshot': 0, 'entrust': 0, 'trade': 0, 'unknown': 0}

    for arch in archive_files:
        try:
            cmd = ["7z", "l", str(arch)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            for line in result.stdout.split('\n'):
                if '.csv' in line.lower():
                    parts = line.split()
                    if parts and parts[-1].endswith('.csv'):
                        filename = parts[-1]
                        total_csv_count += 1
                        file_type = get_file_type(Path(filename))
                        type_counts[file_type or 'unknown'] += 1
        except Exception as e:
            logger.warning(f"  Warning: Failed to scan {arch.name}: {e}")

    logger.info(f"\nScan result - all archives combined:")
    logger.info(f"  Total CSV files: {total_csv_count}")
    logger.info(f"  - Snapshot: {type_counts['snapshot']}")
    logger.info(f"  - Entrust:  {type_counts['entrust']}")
    logger.info(f"  - Trade:    {type_counts['trade']}")
    if type_counts['unknown'] > 0:
        logger.info(f"  - Unknown:  {type_counts['unknown']}")
    logger.info(f"{'=' * 60}\n")

    logger.info("Starting archive processing...")

    for idx, arch in enumerate(archive_files, 1):
        logger.info(f"\n[{idx}/{len(archive_files)}] Processing: {arch.name}")

        temp_extract_dir = data_dir / f"tmp_{arch.stem}"
        temp_extract_dir.mkdir(exist_ok=True)

        try:
            logger.info("    Extracting...")
            cmd = ["7z", "x", str(arch), f"-o{str(temp_extract_dir)}", "-y"]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

            csv_files = list(temp_extract_dir.rglob('*.csv'))
            if csv_files:
                current_stats = count_csv_files_by_type(csv_files)
                logger.info(f"    Extracted {len(csv_files)} CSV file(s): "
                            f"Snapshot: {current_stats['snapshot']}, "
                            f"Entrust: {current_stats['entrust']}, "
                            f"Trade: {current_stats['trade']}")

                logger.info("    Importing...")
                arch_stats = process_csv_files(csv_files, desc="        Import progress")
                merge_stats(global_stats, arch_stats)
                logger.info("    Import finished")
            else:
                logger.warning("    Warning: No CSV files found in archive")

            shutil.rmtree(temp_extract_dir)
            logger.info("    Cleanup done")
            arch.unlink()
            logger.info("    Archive deleted")

        except Exception as e:
            logger.error(f"    Error: {e}")
            if temp_extract_dir.exists():
                shutil.rmtree(temp_extract_dir)

    return global_stats


def process_direct_csv_files(data_dir: Path) -> dict:
    """Process loose CSV files from a directory."""
    logger = logging.getLogger(__name__)
    csv_files = list(data_dir.rglob('*.csv'))

    if not csv_files:
        return {'snapshot': {'files': 0, 'rows': 0},
                'entrust': {'files': 0, 'rows': 0},
                'trade': {'files': 0, 'rows': 0}}

    print_file_statistics(csv_files, "Direct CSV processing mode")
    logger.info("Starting import...")
    return process_csv_files(csv_files)


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    logger.info("Starting Level2 data import")

    global _client, _table_prefix, _max_workers

    try:
        _client = clickhouse_connect.get_client(
            host=args.host,
            port=args.port,
            username=args.user,
            password=args.password,
            database=args.database,
        )
        _client.query("SELECT 1")
        logger.info("ClickHouse connected successfully")
    except Exception as e:
        logger.error(f"ClickHouse connection failed: {e}")
        sys.exit(1)

    _table_prefix = args.table_prefix
    _max_workers = args.max_workers

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} does not exist")
        sys.exit(1)

    archive_files = list(data_dir.glob('*.7z'))

    if archive_files:
        stats = process_archive_files(data_dir)
    else:
        logger.info("No archives found, falling back to direct CSV processing...")
        stats = process_direct_csv_files(data_dir)

    summary = f"""
{'=' * 60}
Import completed!
{'=' * 60}
Snapshot:  {stats['snapshot']['files']:>6} files, {stats['snapshot']['rows']:>12,} rows
Entrust:   {stats['entrust']['files']:>6} files, {stats['entrust']['rows']:>12,} rows
Trade:     {stats['trade']['files']:>6} files, {stats['trade']['rows']:>12,} rows
{'=' * 60}
Total:     {sum(s['files'] for s in stats.values()):>6} files, {sum(s['rows'] for s in stats.values()):>12,} rows
{'=' * 60}
"""
    logger.info(summary)
    print(summary)

    if _client:
        _client.close()
        logger.info("ClickHouse connection closed")


if __name__ == "__main__":
    main()