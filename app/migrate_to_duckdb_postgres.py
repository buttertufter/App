import os
import duckdb
import pandas as pd
import psycopg2
from pathlib import Path

DATA_STORE = Path(__file__).resolve().parent / "data_store"

TABLES = {
    "tickers": "symbol,name,sector,country",
    "prices": "symbol,date,close,volume",
    "fundamentals": "symbol,date,revenue,operating_expenses,ocf,cash,debt,undrawn_credit,financing_in",
    "peers": "symbol,peer",
    "gauges": "symbol,asof_month,A,B,C,D,E,wave,details_json"
}

def migrate_duckdb(db_path="bridge.db"):
    con = duckdb.connect(db_path)
    for table, columns in TABLES.items():
        csv_path = DATA_STORE / f"{table}.csv"
        if not csv_path.exists():
            print(f"Skipping {csv_path} (not found)")
            continue
        print(f"Migrating {table} from {csv_path}")
        df = pd.read_csv(csv_path)
        con.execute(f"CREATE TABLE IF NOT EXISTS {table} ({', '.join([c+' VARCHAR' for c in columns.split(',')])})")
        con.register("df", df)
        con.execute(f"INSERT INTO {table} SELECT * FROM df")
    print("DuckDB migration complete.")

def migrate_postgres(pg_conn_str):
    conn = psycopg2.connect(pg_conn_str)
    cur = conn.cursor()
    for table, columns in TABLES.items():
        csv_path = DATA_STORE / f"{table}.csv"
        if not csv_path.exists():
            print(f"Skipping {csv_path} (not found)")
            continue
        print(f"Migrating {table} from {csv_path}")
        df = pd.read_csv(csv_path)
        col_list = columns.split(',')
        cur.execute(f"CREATE TABLE IF NOT EXISTS {table} ({', '.join([c+' TEXT' for c in col_list])})")
        for _, row in df.iterrows():
            vals = tuple(row.get(c, None) for c in col_list)
            placeholders = ','.join(['%s']*len(col_list))
            cur.execute(f"INSERT INTO {table} ({','.join(col_list)}) VALUES ({placeholders})", vals)
    conn.commit()
    cur.close()
    conn.close()
    print("Postgres migration complete.")

if __name__ == "__main__":
    migrate_duckdb()
    # For Postgres, set your connection string and uncomment:
    # migrate_postgres("dbname=bridge user=postgres password=yourpw host=localhost")
