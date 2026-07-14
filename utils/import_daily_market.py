"""
import_daily_market.py

Fetch daily stock market data from BaoStock (via utils.fetch_baostock_data) and
insert it into a ClickHouse database in batches. The script supports checkpointing
so that interrupted runs can resume without re-fetching already processed stocks.

Dependencies:
    - clickhouse-connect
    - pandas, numpy
    - tqdm
    - python-dotenv (optional, for loading .env files)
    - A local `utils` module providing:
        * get_stock_code_list(filter_exchange=True) -> list of stock codes
        * fetch_baostock_data(code, start_date, end_date) -> pandas.DataFrame
        * DATA_DIR (pathlib.Path) pointing to the data directory for logs/checkpoints

Usage:
    # Process all eligible stocks (default behavior)
    python import_daily_market.py --start-date 2015-01-01 --end-date 2020-12-31 --batch-size 100

    # Process specific stocks only
    python import_daily_market.py --codes sh.600000,sz.000001 --start-date 2020-01-01 --end-date 2020-12-31
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="py_mini_racer")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", module="baostock")

import os
import json
import atexit
import datetime
import gc
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import clickhouse_connect
from utils import get_stock_code_list, fetch_baostock_data, DATA_DIR


def parse_args():
    """Parse command-line arguments and return the populated namespace."""
    parser = argparse.ArgumentParser(
        description="Fetch daily stock data from BaoStock and insert into ClickHouse"
    )
    # Date range
    parser.add_argument(
        "--start-date", type=str, default="2010-01-01",
        help="Start date in YYYY-MM-DD format (default: 2010-01-01)"
    )
    parser.add_argument(
        "--end-date", type=str, default="2018-01-01",
        help="End date in YYYY-MM-DD format (default: 2018-01-01)"
    )
    # Batch size
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Number of stocks to accumulate before inserting into ClickHouse (default: 50)"
    )
    # ClickHouse connection
    parser.add_argument(
        "--host", type=str, default=os.getenv("CH_HOST", "localhost"),
        help="ClickHouse host (default: localhost or CH_HOST env var)"
    )
    parser.add_argument(
        "--port", type=int, default=8123, help="ClickHouse HTTP port (default: 8123)"
    )
    parser.add_argument(
        "--user", type=str, default="default", help="ClickHouse user (default: default)"
    )
    parser.add_argument(
        "--password", type=str, default="", help="ClickHouse password (default: empty)"
    )
    parser.add_argument(
        "--database", type=str, default="quant",
        help="ClickHouse database name (default: quant)"
    )
    parser.add_argument(
        "--table", type=str, default="stock_daily_market",
        help="Target table name in ClickHouse (default: stock_daily_market)"
    )
    parser.add_argument(
        "--connect-timeout", type=int, default=60,
        help="Connection timeout in seconds (default: 60)"
    )
    parser.add_argument(
        "--send-receive-timeout", type=int, default=6000,
        help="Send/receive timeout in seconds (default: 6000)"
    )
    # Stock list filter
    parser.add_argument(
        "--filter-exchange", action="store_true", default=True,
        help="Filter stocks by exchange in get_stock_code_list (default: True)"
    )
    parser.add_argument(
        "--no-filter-exchange", dest="filter_exchange", action="store_false",
        help="Do not filter by exchange"
    )
    # *** New argument: specific stock codes ***
    parser.add_argument(
        "--codes", type=str, default=None,
        help="Comma-separated list of stock codes to process (e.g., sh.600000,sz.000001). "
             "If provided, only these codes are imported and --filter-exchange is ignored."
    )
    # File paths
    parser.add_argument(
        "--checkpoint-file", type=str, default=None,
        help="Path to checkpoint file (auto-generated if not provided)"
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Path to log file (auto-generated if not provided)"
    )
    return parser.parse_args()


def main():
    """Main entry point: parse args, connect to DB, process stocks, and insert data."""
    args = parse_args()

    # Ensure the data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Determine checkpoint and log file paths
    if args.checkpoint_file:
        checkpoint_file = Path(args.checkpoint_file)
    else:
        checkpoint_file = DATA_DIR / f"stock_daily_checkpoint_{args.start_date}_{args.end_date}.json"

    if args.log_file:
        log_file = Path(args.log_file)
    else:
        log_file = DATA_DIR / f"stock_daily_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Internal state for logging and checkpointing
    log_events = []

    def add_log_event(level, message, details=None):
        """Record a log event to be saved later."""
        event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "level": level,
            "message": message,
        }
        if details is not None:
            event["details"] = str(details)
        log_events.append(event)

    def save_log():
        """Persist all log events to the JSON log file."""
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(log_events, f, ensure_ascii=False, indent=2)
        except Exception as e:
            tqdm.write(f"CRITICAL: Failed to save log file: {e}")

    # Register log saving on normal exit
    atexit.register(save_log)

    def load_checkpoint():
        """Load the set of already processed stock codes from the checkpoint file."""
        if checkpoint_file.exists():
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                return set(json.load(f))
        return set()

    def save_checkpoint(processed_codes):
        """Save the current set of processed stock codes to the checkpoint file."""
        try:
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(list(processed_codes), f, ensure_ascii=False, indent=2)
        except Exception as e:
            add_log_event("WARNING", f"Failed to save checkpoint: {e}")

    # Connect to ClickHouse
    client = clickhouse_connect.get_client(
        host=args.host,
        port=args.port,
        username=args.user,
        password=args.password,
        database=args.database,
        compress=False,
        connect_timeout=args.connect_timeout,
        send_receive_timeout=args.send_receive_timeout,
    )
    add_log_event("INFO", f"Connected to ClickHouse at {args.host}:{args.port}/{args.database}")

    if args.codes:
        # Use the explicitly provided codes
        stock_code_lst = [code.strip() for code in args.codes.split(",") if code.strip()]
        add_log_event("INFO", f"Using {len(stock_code_lst)} explicitly provided stock codes")
        # When codes are given, filter_exchange has no effect
        if args.filter_exchange:
            add_log_event("INFO", "filter_exchange is ignored when --codes is provided")
    else:
        # Default behavior: get all eligible codes from utils
        stock_code_lst = get_stock_code_list(filter_exchange=args.filter_exchange)
        add_log_event("INFO", f"Fetched {len(stock_code_lst)} stocks from get_stock_code_list")

    processed_set = load_checkpoint()
    codes_to_process = [code for code in stock_code_lst if code not in processed_set]
    add_log_event(
        "INFO",
        f"Total stocks: {len(stock_code_lst)}, already done: {len(processed_set)}, to process: {len(codes_to_process)}"
    )

    if not codes_to_process:
        add_log_event("INFO", "All stocks have been processed in previous runs. Exiting.")
        save_log()
        return

    # Column mapping from BaoStock fields to our target schema
    column_mapping = {
        "date": "trade_date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Amount": "amount",
        "pctChg": "pct_chg",
        "turn": "turnover_rate",
    }
    target_columns = [
        "trade_date", "stock_code", "open", "close", "high", "low",
        "volume", "amount", "pct_chg", "turnover_rate",
    ]

    batch_dfs = []          # Accumulate DataFrames before insertion
    failed_records = []     # (code, stage, error_message)

    progress = tqdm(codes_to_process, desc="Fetching daily data")
    for code in progress:
        # 1. Fetch and clean data for a single stock
        try:
            raw_df = fetch_baostock_data(code, args.start_date, args.end_date)
            if raw_df is None or raw_df.empty:
                add_log_event("INFO", f"No data for {code}, skip")
                continue

            raw_df.reset_index(inplace=True)
            raw_df.rename(columns=column_mapping, inplace=True)
            raw_df["stock_code"] = code
            # Keep only columns that exist in the raw data
            existing_cols = [col for col in target_columns if col in raw_df.columns]
            raw_df = raw_df[existing_cols]
            batch_dfs.append(raw_df)

        except Exception as e:
            add_log_event("ERROR", f"Error processing {code}: {e}", details=str(e))
            failed_records.append((code, "fetch/clean", str(e)))
            continue

        # 2. If the batch is full, insert into ClickHouse
        if len(batch_dfs) >= args.batch_size:
            _flush_batch(client, args.table, batch_dfs, target_columns, processed_set,
                         save_checkpoint, add_log_event, failed_records)

    # 3. Insert any remaining data
    if batch_dfs:
        _flush_batch(client, args.table, batch_dfs, target_columns, processed_set,
                     save_checkpoint, add_log_event, failed_records)

    # Report outcome
    if failed_records:
        add_log_event("ERROR", f"Total failures: {len(failed_records)} stocks")
        for code, stage, err in failed_records:
            add_log_event("ERROR", f"{code} ({stage}): {err}")
    else:
        add_log_event("SUCCESS", "All stocks processed successfully.")

    save_log()


def _flush_batch(client, table_name, batch_dfs, target_columns, processed_set,
                 save_checkpoint_fn, add_log_event_fn, failed_records):
    """
    Concatenate the accumulated DataFrames, ensure all target columns exist,
    and insert into ClickHouse. Update checkpoint and handle errors.
    """
    try:
        full_df = pd.concat(batch_dfs, ignore_index=True)

        # Fill missing columns with appropriate defaults
        for col in target_columns:
            if col not in full_df.columns:
                if col == "volume":
                    full_df[col] = pd.NA
                elif col in ("amount", "pct_chg", "turnover_rate"):
                    full_df[col] = np.nan
                else:
                    full_df[col] = None

        # Cast to correct types
        full_df["trade_date"] = pd.to_datetime(full_df["trade_date"])
        full_df["volume"] = full_df["volume"].astype("Int64")
        full_df = full_df[target_columns]

        # Insert into ClickHouse
        client.insert_df(table_name, full_df)

        # Mark stocks as processed and save checkpoint
        for df in batch_dfs:
            processed_set.add(df["stock_code"].iloc[0])
        save_checkpoint_fn(processed_set)

    except Exception as e:
        batch_codes = [df["stock_code"].iloc[0] for df in batch_dfs]
        add_log_event_fn(
            "ERROR",
            f"Insert batch failed for {len(batch_codes)} stocks: {batch_codes}",
            details=str(e),
        )
        for code in batch_codes:
            failed_records.append((code, "insert", str(e)))
    finally:
        batch_dfs.clear()
        if "full_df" in locals():
            del full_df
        gc.collect()


if __name__ == "__main__":
    main()