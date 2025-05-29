import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pyodbc

from dotenv import load_dotenv
from urllib.parse import quote_plus
from sqlalchemy import create_engine

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sql_connection import sql_connect, write_sql_conn_create

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Suppress specific warnings
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")
