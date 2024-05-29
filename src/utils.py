import pandas as pd
import numpy as np
import os, sys
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from static.constants import *


def customer_cumsum(df):
    customer_df = df.groupby(customerid).agg({amount: ['sum'], invoiceno: [lambda x: x.nunique()]})
    customer_df.columns = customer_df.columns.droplevel(1) + '_' + customer_df.columns.droplevel(0)
    customer_df = customer_df.reset_index()
    customer_df.sort_values('amount_sum', ascending=False, inplace=True)
    customer_df = customer_df.rename(columns={'InvoiceNo_<lambda>': 'num_of_unique_purchases'})
    customer_df['amount_sum'] = np.round(customer_df['amount_sum'], 2)
    customer_df = customer_df[customer_df['amount_sum'] > 0]
    customer_df['amount_cumsum'] = (customer_df['amount_sum'] / customer_df['amount_sum'].sum(axis=0)).cumsum()
    customer_df = customer_df.reset_index(drop=True)

    return customer_df


def monthly_revenue_and_transactions(df):
    month_year = 'month_year'
    df['month_year'] = pd.to_datetime(df[invoicedate]).dt.to_period(freq='M')
    a = df.groupby('month_year', as_index=False)[amount].sum()
    a[month_year] = a[month_year].astype(str)

    b = df.groupby('month_year', as_index=False)[invoiceno].nunique()
    b[month_year] = b[month_year].astype(str)

    c = df.groupby(month_year).agg({amount: ['sum'], invoiceno: [lambda x: x.nunique()]})
    c.columns = c.columns.droplevel(1) + '_' + c.columns.droplevel(0)
    c = c.reset_index().rename(columns={'InvoiceNo_<lambda>': 'num_of_unique_purchases'})
    c[month_year] = c[month_year].astype(str)
    c['avg_transaction_amount'] = c['amount_sum'] / c['num_of_unique_purchases']

    return a, b, c


def country_amount(df, N=6):
    country_df = df.groupby([country], as_index=False)[amount].sum().sort_values(amount, ascending=False)
    country_df = country_df.reset_index(drop=True)
    top_countries = country_df[:N][country]
    country_df['amount_percent'] = country_df[amount] / country_df[amount].sum(axis=0)
    country_df = country_df[country_df[country].isin(top_countries)]
    country_df['amount_percent'] = np.round(country_df['amount_percent'], 2)

    return country_df


def top_countries_by_month_year(df, top_countries):
    month_year = 'month_year'
    country_month_year = df.groupby(['month_year', country], as_index=False)[amount].sum()
    country_month_year[month_year] = country_month_year[month_year].astype(str)
    country_month_year_top = country_month_year[country_month_year[country].isin(top_countries)]

    return country_month_year_top


def get_whale_customers(processed_df, q=.95):
    topc = processed_df.groupby([customerid, country], as_index=False)[amount].sum()
    topc = topc.sort_values(amount, ascending=False)
    topc.reset_index(drop=True, inplace=True)
    topc[amount] = np.round(topc[amount], 2)
    top_qunatile = topc[amount].quantile([q]).values[0]
    whales = topc[topc[amount] > top_qunatile]

    return whales


def get_whales_by_column(whales, groupby_column, top_countries_n=5):
    whales_count = whales.groupby(groupby_column, as_index=False)[customerid].nunique().rename(
        columns={customerid: 'customers'})
    # whales_count = whales_count[whales_count[country].isin(top_countries)]
    whales_count.sort_values('customers', ascending=False, inplace=True)

    return whales_count[:top_countries_n]
