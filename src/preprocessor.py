import pandas as pd

from static.constants import *


def duplicated_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    check for duplicated rows.
    Note: there are some duplicated rows but i don't think they indicate on an error.
    it may mean that the item was registered one at a time, so this won't be used"

    Args:
        df: dataframe to be analyzed
    """
    return df


def remove_customers_with_negative_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    remove customers that have negative amount spent. in some cases could indicate for fraud

    Args:
        df: dataframe to be analyzed

    Returns:
        df: filtered dataframe without those customers
    """
    customer_amount = df.groupby(customerid, as_index=False)[amount].sum()
    negative_amount_customer_list = customer_amount.loc[customer_amount[amount] < 0, customerid]
    print(f'dropped {negative_amount_customer_list.shape[0]} customers with negative balance')

    return df[~df[customerid].isin(negative_amount_customer_list)]


def remove_customers_with_suspicious_return(df: pd.DataFrame, suspicious_price_th: int = -1_000) -> pd.DataFrame:
    """
    remove customers that have high negative transaction, i.e. high return

    Args:
        df: dataframe to be analyzed

        suspicious_price_th: above this price it's a suspicious transaction

    Returns:
        df: filtered dataframe without those customers
    """
    high_return = df.loc[df[amount] < suspicious_price_th, customerid].unique()
    print(f'dropped {high_return.shape[0]} customers with suspicious transaction')

    return df[~df[customerid].isin(high_return)]


def add_new_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    add new columns for analysis and modeling:
    amount = defined as the unit price multiplied by quantity
    month_year = defined as the invoice data month and year

    Args:
        df: dataframe to be analyzed

    Returns:
        df: dataframe with added columns
    """
    df['amount'] = df[unitprice] * df[quantity]
    df['month_year'] = pd.to_datetime(df[invoicedate]).dt.to_period(freq='M')

    return df


def remove_transactions_without_customerid(df: pd.DataFrame) -> pd.DataFrame:
    """
    remove transactions without customerid, i.e. customerid is null

    Args:
        df: dataframe to be filtered

    Returns:
        df_no_null: dataframe without null for customer ids
    """
    df_no_null = df[~pd.isnull(df[customerid])]

    return df_no_null


def preprocessor_factory():
    """Factory Method"""
    functions = {
        'duplicated_rows': duplicated_rows,
        'remove_transactions_without_customerid': remove_transactions_without_customerid,
        'add_new_columns': add_new_columns,
        'remove_customers_with_negative_amount': remove_customers_with_negative_amount,
        'remove_customers_with_suspicious_return': remove_customers_with_suspicious_return,
    }

    return [f for f in functions.values()]


class Preprocessor:
    registered_functions = preprocessor_factory()

    @staticmethod
    def preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
        for func in Preprocessor.registered_functions:
            df = func(df)

        return df


if __name__ == '__main__':
    df = pd.read_parquet('df.parquet.gzip')
    preprocessor = Preprocessor()
    processed_df = preprocessor.preprocessing_pipeline(df)
