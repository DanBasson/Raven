import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from static.constants import *
from src.feature_engineering import FeatureEngineering
from src.preprocessor import Preprocessor

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 99999)


class ClusterModel:

    def __init__(self, X, model_hyperparameters=None, elbow_max_clusters=10,
                 model=KMeans):
        if model_hyperparameters is None:
            model_hyperparameters = {}
        self.elbow_distortions = []
        self.X = X
        self.elbow_iter_range = range(1, elbow_max_clusters)
        self.model = model
        self.model_hyperparameters = model_hyperparameters

    def fit_model(self):
        self.fitted_model = self.model(n_clusters=self.model_hyperparameters['n_clusters'],
                                random_state=self.model_hyperparameters['random_state'])
        self.fitted_model.fit(self.X)

    def plot_model_clusters(self, df):
        df['Cluster'] = self.model.fit_predict(self.X)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        colors = ['r', 'g', 'b', 'c', 'm']
        for cluster in range(5):
            clustered_data = df[df['Cluster'] == cluster]
            ax.scatter(clustered_data['recency'],
                       clustered_data['frequency'],
                       clustered_data['monetary_value'],
                       c=colors[cluster],
                       label=f'Cluster {cluster}')

        # Labels and title
        ax.set_xlabel('Recency')
        ax.set_ylabel('Frequency')
        ax.set_zlabel('Monetary')
        ax.set_title('3D K-means Clustering of RFM Data')
        ax.legend()

        # Show plot
        plt.show()

        return

    def elbow_method(self):
        for k in self.elbow_iter_range:
            model_ = self.model(n_clusters=k)
            model_.fit(self.X)
            self.elbow_distortions.append(model_.inertia_)

    def plot_elbow(self):
        plt.figure(figsize=(15, 5))
        plt.plot(self.elbow_iter_range, self.elbow_distortions)
        plt.grid(True)
        plt.xlabel('Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow curve')
        plt.show()


if __name__ == '__main__':
    df = pd.read_parquet('df.parquet.gzip')
    preprocessor = Preprocessor()
    processed_df = preprocessor.preprocessing_pipeline(df)

    fe = FeatureEngineering()
    fe_df = fe.feature_engineering_pipeline(processed_df)

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import KMeans

    mms = MinMaxScaler()
    X = pd.DataFrame(fe_df, columns=[i for i in fe_df.columns])
    X = mms.fit_transform(X)

    hyper_params = {'n_clusters': 5, 'random_state': 1}
    k_means = ClusterModel(X, elbow_max_clusters=10, model=KMeans, model_hyperparameters=hyper_params)
    k_means.fit_model()
    fe_df['Cluster'] = k_means.fitted_model.fit_predict(X=k_means.X)

    fe_df['Cluster'] = fe_df['Cluster'].astype(str)
    # import plotly.express as px
    #
    # fig = px.scatter_3d(fe_df, x='recency', y='frequency', z='monetary_value',
    #                     color='Cluster', title='3D Plot of K-means Clusters on RFM Data'
    #                     , width=800, height=800)  # .update_layout(showlegend=False)
    # fig.update_layout(showlegend=False)
    # # fig.update_traces(showlegend=False)
    #
    # fig.show()

    from sklearn.decomposition import PCA

    pca = PCA(n_components=5)
    # pca_components.fit(X)
    pca_components = pca.fit_transform(X)
    # print(pca_components.explained_variance_ratio_)

    fe_df['PCA1'] = pca_components[:, 0]
    fe_df['PCA2'] = pca_components[:, 1]

    # Plot the first two PCA components with different colors for each cluster
    plt.figure(figsize=(10, 7))
    colors = ['r', 'g', 'b', 'c', 'm']
    for cluster in range(5):
        clustered_data = fe_df[fe_df['Cluster'] == cluster]
        plt.scatter(clustered_data['PCA1'],
                    clustered_data['PCA2'],
                    c=colors[cluster],
                    label=f'Cluster {cluster}')

    # Labels and title
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title('PCA of RFM Data Colored by Cluster')
    plt.legend()

    # Show plot
    plt.show()



# distorsions = []
# to_range = range(1, 15)
# for k in to_range:
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(rfm_df)
#     distorsions.append(kmeans.inertia_)
#
# fig = plt.figure(figsize=(15, 5))
# plt.plot(to_range, distorsions)
# plt.grid(True)
# plt.title('Elbow curve')
# plt.show()


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
