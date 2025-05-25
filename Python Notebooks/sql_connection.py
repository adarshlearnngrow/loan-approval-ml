import pyodbc
from dotenv import load_dotenv
import pandas as pd
import os
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
from urllib.parse import quote_plus

load_dotenv(dotenv_path='.env')

server = os.getenv("SQL_SERVER")
database = os.getenv("SQL_DATABASE")
user = os.getenv("SQL_USER")
password = os.getenv("SQL_PASSWORD")

def sql_connect(query):
    conn = None
    try:
        conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={user};"
        f"PWD={password};"
        "TrustServerCertificate=yes;"
        )
        print("Connection to SQL Server established successfully.")
        data = pd.read_sql(query, conn)
        return data

    except Exception as e:
        print(f"Database connection or query failed: {e}")
        return None

    finally:
        if conn:
            conn.close()
            print("Connection closed.")

def write_sql_conn_create():
    conn = None
    try:
        engine = create_engine(
            f"mssql+pyodbc://{user}:{quote_plus(password)}@{server}/{database}"
            f"?driver={quote_plus('ODBC Driver 17 for SQL Server')}"
        )
        # Optionally test connection
        with engine.connect() as conn:
            print("Connection to SQL Server established successfully.")
        return engine
    except Exception as e:
        print(f"Failed to create SQL engine: {e}")
        return None


            