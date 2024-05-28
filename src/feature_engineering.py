import pandas as pd
import numpy as np
from typing import List

from static.constants import *
from src.preprocessor import Preprocessor


def create_recency_column(df: pd.DataFrame) -> pd.DataFrame:
    column_name = 'recency'
    rfm_df = df.groupby(customerid, as_index=False)[invoicedate].max()
    CURRENT_TIME = df[invoicedate].max()
    rfm_df[column_name] = (CURRENT_TIME - rfm_df[invoicedate]).dt.days

    return rfm_df[[customerid, column_name]]


def create_monetary_value_column(df: pd.DataFrame) -> pd.DataFrame:
    column_name = 'monetary_value'
    rfm_df = df.groupby(customerid, as_index=False)[amount].sum()
    rfm_df = rfm_df.rename(columns={'amount': column_name})

    return rfm_df


def create_frequency_column(df: pd.DataFrame) -> pd.DataFrame:
    column_name = 'frequency'
    rfm_df = df.groupby(customerid, as_index=False)[invoiceno].nunique()
    rfm_df = rfm_df.rename(columns={invoiceno: column_name})

    return rfm_df


def create_unique_items_count_column(df: pd.DataFrame) -> pd.DataFrame:
    # unique items average per transaction

    column_name = 'unique_items_count'
    unique_items_avg = df.groupby([customerid, invoiceno], as_index=False)[stockcode].size().rename(
        columns={'size': 'unique_items_count'})
    unique_items = unique_items_avg.groupby([customerid], as_index=False)[column_name].mean()

    return unique_items


def create_items_sold_column(df: pd.DataFrame) -> pd.DataFrame:
    column_name = 'quantity_sold_avg'
    unique_items_avg = df.groupby([customerid, invoiceno], as_index=False)[quantity].sum().rename(
        columns={quantity: 'total_items_sold'})
    unique_items = unique_items_avg.groupby([customerid], as_index=False)['total_items_sold'].mean().rename(
        columns={'total_items_sold': column_name})

    return unique_items


def feature_engineering_factory() -> List:
    """Factory Method"""
    functions = {
        'create_recency_column': create_recency_column,
        'create_monetary_value_column': create_monetary_value_column,
        'create_frequency_column': create_frequency_column,
        'create_unique_items_count_column': create_unique_items_count_column,
        'create_items_sold_column': create_items_sold_column,
    }

    return [f for f in functions.values()]


class FeatureEngineering:
    registered_functions = feature_engineering_factory()

    @staticmethod
    def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
        unique_customers_df = pd.DataFrame({customerid: df[customerid].unique()})
        for i, func in enumerate(FeatureEngineering.registered_functions):
            df_new_column = func(df)
            if i == 0:
                customer_df = unique_customers_df.merge(df_new_column, on=customerid)
            else:
                customer_df = customer_df.merge(df_new_column, on=customerid)

        return customer_df


if __name__ == '__main__':
    df = pd.read_parquet('df.parquet.gzip')
    CURRENT_TIME = df[invoicedate].max()
    preprocessor = Preprocessor()
    processed_df = preprocessor.preprocessing_pipeline(df)
    print(f'total customers - {processed_df[customerid].nunique()}')

    fe = FeatureEngineering()
    fe_df = fe.feature_engineering_pipeline(processed_df)
    print(f'total customers - {processed_df[customerid].nunique()}')
