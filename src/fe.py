import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display as di
import plotly.express as px

from constants import *
from preprocessor import Preprocessor

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 99999)


def create_recency_column(df):
    column_name = 'recency'
    rfm_df = df.groupby(customerid, as_index=False)[invoicedate].max()
    CURRENT_TIME = df[invoicedate].max()
    rfm_df[column_name] = (CURRENT_TIME - rfm_df[invoicedate]).dt.days

    return rfm_df[[customerid, column_name]]


def create_monetary_value_column(df):
    column_name = 'monetary_value'
    rfm_df = df.groupby(customerid, as_index=False)[amount].sum()
    rfm_df = rfm_df.rename(columns={'amount': column_name})

    return rfm_df


def create_frequency_column(df):
    column_name = 'frequency'
    rfm_df = df.groupby(customerid, as_index=False)[invoiceno].nunique()
    rfm_df = rfm_df.rename(columns={invoiceno: column_name})

    return rfm_df


def create_unique_items_count_column(df):
    # unique items average per transaction

    column_name = 'unique_items_count'
    unique_items_avg = df.groupby([customerid, invoiceno], as_index=False)[stockcode].size().rename(
        columns={'size': 'unique_items_count'})
    unique_items = unique_items_avg.groupby([customerid], as_index=False)[column_name].mean()

    return unique_items


def create_items_sold_column(df):
    column_name = 'quantity_sold_avg'
    unique_items_avg = df.groupby([customerid, invoiceno], as_index=False)[quantity].sum().rename(
        columns={quantity: 'total_items_sold'})
    unique_items = unique_items_avg.groupby([customerid], as_index=False)['total_items_sold'].mean().rename(
        columns={'total_items_sold': column_name})

    return unique_items


def feature_engineering_factory():
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


####

if __name__ == '__main__':
    df = pd.read_parquet('df.parquet.gzip')
    CURRENT_TIME = df[invoicedate].max()
    preprocessor = Preprocessor()
    processed_df = preprocessor.preprocessing_pipeline(df)
    print(f'total customers - {processed_df[customerid].nunique()}')

    fe = FeatureEngineering()
    fe_df = fe.feature_engineering_pipeline(processed_df)
    print(f'total customers - {processed_df[customerid].nunique()}')

#
# ##RFM
# # del df
# CURRENT_TIME = df_no_null[invoicedate].max()
#
# # recency
# max_tran_date = df_no_null.groupby(customerid, as_index=False)[invoicedate].max()
# max_tran_date['recency'] = (CURRENT_TIME - max_tran_date[invoicedate]).dt.days
#
# rfm_df = df_no_null.groupby(customerid).agg({amount: ['sum'], invoiceno: [lambda x: x.nunique()]})
# rfm_df.columns = rfm_df.columns.droplevel(1) + '_' + rfm_df.columns.droplevel(0)
# rfm_df = rfm_df.reset_index()
# rfm_df = rfm_df.rename(columns={'InvoiceNo_<lambda>': 'frequency', 'amount_sum': 'monetary_value'})
# rfm_df = rfm_df.merge(max_tran_date[[customerid, 'recency']], on=customerid)
#
# # unique items average per transaction
# unique_items_avg = df_no_null.groupby([customerid, invoiceno], as_index=False)[stockcode].size().rename(
#     columns={'size': 'unique_items_count'})
# unique_items = unique_items_avg.groupby([customerid], as_index=False)['unique_items_count'].mean()
# rfm_df = rfm_df.merge(unique_items, on=customerid)
#
# # unique items quantity average per transaction
# unique_items_avg = df_no_null.groupby([customerid, invoiceno], as_index=False)[quantity].sum().rename(
#     columns={quantity: 'total_items_sold'})
# unique_items = unique_items_avg.groupby([customerid], as_index=False)['total_items_sold'].mean().rename(
#     columns={'total_items_sold': 'quantity_sold_avg'})
# rfm_df = rfm_df.merge(unique_items, on=customerid)
#
# # elbow
# # from sklearn.cluster import KMeans
# # import matplotlib.pyplot as plt
# #
# # distorsions = []
# # to_range = range(1, 15)
# # for k in to_range:
# #     kmeans = KMeans(n_clusters=k)
# #     kmeans.fit(rfm_df)
# #     distorsions.append(kmeans.inertia_)
# #
# # fig = plt.figure(figsize=(15, 5))
# # plt.plot(to_range, distorsions)
# # plt.grid(True)
# # plt.title('Elbow curve')
# # plt.show()
#
#
# # kmeans
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.cluster import KMeans
#
# ms = MinMaxScaler()
# X = pd.DataFrame(rfm_df, columns=[i for i in rfm_df.columns])
# X = ms.fit_transform(X)
#
# kmeans = KMeans(n_clusters=5, random_state=0)
# kmeans.fit(X)
#
# rfm_df['Cluster'] = kmeans.fit_predict(X)
# #
# # fig = plt.figure(figsize=(10, 7))
# # ax = fig.add_subplot(111, projection='3d')
# #
# #
# # rfm_df = rfm_df[rfm_df['frequency'] < 50]
# # rfm_df = rfm_df[rfm_df['monetary_value'] < 100_000]
# #
# #
# # # Define colors
# # colors = ['r', 'g', 'b', 'c', 'm']
# # for cluster in range(5):
# #     clustered_data = rfm_df[rfm_df['Cluster'] == cluster]
# #     ax.scatter(clustered_data['recency'],
# #                clustered_data['frequency'],
# #                clustered_data['monetary_value'],
# #                c=colors[cluster],
# #                label=f'Cluster {cluster}')
# #
# # # Labels and title
# # ax.set_xlabel('Recency')
# # ax.set_ylabel('Frequency')
# # ax.set_zlabel('Monetary')
# # ax.set_title('3D K-means Clustering of RFM Data')
# # ax.legend()
# #
# # # Show plot
# # plt.show()
#
# # print(kmeans.inertia_)
#
#
# # pca
#
# from sklearn.decomposition import PCA
#
# pca = PCA(n_components=5)
# # pca_components.fit(X)
# pca_components = pca.fit_transform(X)
# # print(pca_components.explained_variance_ratio_)
#
# rfm_df['PCA1'] = pca_components[:, 0]
# rfm_df['PCA2'] = pca_components[:, 1]
#
# # Plot the first two PCA components with different colors for each cluster
# plt.figure(figsize=(10, 7))
# colors = ['r', 'g', 'b', 'c', 'm']
# for cluster in range(5):
#     clustered_data = rfm_df[rfm_df['Cluster'] == cluster]
#     plt.scatter(clustered_data['PCA1'],
#                 clustered_data['PCA2'],
#                 c=colors[cluster],
#                 label=f'Cluster {cluster}')
#
# # Labels and title
# plt.xlabel('PCA1')
# plt.ylabel('PCA2')
# plt.title('PCA of RFM Data Colored by Cluster')
# plt.legend()
#
# # Show plot
# plt.show()
