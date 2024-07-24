import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, silhouette_samples
from imblearn.over_sampling import SMOTENC
<<<<<<< HEAD
import warnings

warnings.filterwarnings('ignore')
=======
>>>>>>> 92d31836236700bf0a18a9f7a60616bb1852aabd

class FairBalance:
    def __init__(
        self, data, features, continous_features, 
        drop_features, sensitive_attribute, target, 
        cluster_algo='kmeans', ratio=0.25, knn=5,
    ):
        self.data = data.copy()
        self.continous_features = continous_features
        self.sensitive_attribute = sensitive_attribute
        self.features = features
        self.drop_features = drop_features
        self.target = target
        self.cluster_algo = cluster_algo
        self.ratio = ratio
        self.knn = knn

    def fit(self):
        self.cluster()
        self.filter()

    def cluster(self):
        scaler = MinMaxScaler()
        X = self.data.drop([self.sensitive_attribute, self.target], axis=1)
        X[self.continous_features] = scaler.fit_transform(X[self.continous_features])

        if self.cluster_algo == 'kmeans':
            model = KMeans()
        elif self.cluster_algo == 'agg':
            model = AgglomerativeClustering()
        elif self.cluster_algo == 'spe':
            model = SpectralClustering()
        else:
            model = self.cluster_algo

        max_s = -np.inf
        for k in range(2, 10):
            model = model.set_params(n_clusters=k)
            model.fit(X)
            groups = model.labels_
            s_score = silhouette_score(X, groups)
            score_list = silhouette_samples(X, groups)
            if s_score > max_s:
                best_k = k
                max_s = s_score
                best_clist = groups
                best_slist = score_list
        print(f'cluster the original dataset into {best_k} clusters.')
        self.data['score'] = best_slist
        self.data['group'] = best_clist
        
    def filter(self):
        scores = self.data['score'].tolist()
        s_rank = np.sort(scores)
        idx = int(len(s_rank)*self.ratio)
        threshold = s_rank[idx]
        print(f'removing {idx} samples from the original dataset...')
        self.X_clean = self.data[self.data['score'] > threshold]

    def new_smote(self, dfi):
        label = self.target
        categorical_features = list(set(dfi.keys().tolist()) - set(self.continous_features))
        categorical_loc = [dfi.columns.get_loc(c) for c in categorical_features if c in dfi.keys()]

        # count the class distribution 
        min_y = dfi[label].value_counts().idxmin()
        max_y = dfi[label].value_counts().idxmax()
        min_X = dfi[dfi[label] == min_y]
        max_X = dfi[dfi[label] == max_y]
        ratio = len(max_X) - len(min_X)
        
        nbrs = NearestNeighbors(
            n_neighbors=min(self.knn, len(dfi)), algorithm='auto',
        ).fit(dfi[self.features])
        dfs = []
        for j in range(len(min_X)):
            dfj = min_X.iloc[j]
            nn_list = nbrs.kneighbors(
                np.array(dfj[self.features]).reshape(1, -1), 
                return_distance=False,
            )[0]
            df_nn = dfi.iloc[nn_list]
            dfs.append(df_nn)
        df_nns = pd.concat(list(dfs), ignore_index=True).drop_duplicates()   

        X_temp = pd.concat([df_nns, min_X], ignore_index=True)
        y_temp = list(np.repeat(1, len(df_nns))) + list(np.repeat(0, len(min_X)))
        
        min_k = max(1, min(self.knn, len(df_nns)-1))
        sm = SMOTENC(
            categorical_features=categorical_loc, random_state=42, 
            sampling_strategy={1: len(df_nns)+ratio, 0: len(min_X)}, 
            k_neighbors=min_k,
        )
        Xi_res, yi_res = sm.fit_resample(X_temp, y_temp)
        df_res = pd.DataFrame(Xi_res, columns=dfi.keys().tolist())
        df_add = df_res.iloc[len(X_temp):]
        df_add[label] = min_y
        df_new = pd.concat([dfi, df_add], ignore_index=True)
        return df_new

    def generater(self):
        dfs = []
        groups = list(self.X_clean['group'].unique())
        for i in groups:
            dfi = self.X_clean[self.X_clean['group'] == i].drop(['group', 'score'], axis=1)
            if (len(dfi[self.target].unique()) == 1 or len(dfi) == 0):
                continue
            Xi_res = self.new_smote(dfi)
            dfs.append(Xi_res)
        X_cres = pd.concat(list(dfs), ignore_index=True)
        return X_cres[self.features], X_cres[self.target]
<<<<<<< HEAD

def balancing(x_train, target, knn, sensitive_attribute, features, drop_features, continous_features):
    fcb = FairBalance(
        x_train, features, 
        continous_features, drop_features, 
        sensitive_attribute, target, knn=knn,
    )
    fcb.fit()
    X_balanced, y_balanced = fcb.generater()
    return X_balanced, y_balanced

def main():
    frac = 0.01
    x_train = pd.read_csv('/rdf/db/public-tabular-datasets/adult/x_train.csv', index_col=0)
    x_eval = pd.read_csv('/rdf/db/public-tabular-datasets/adult/x_eval.csv', index_col=0)
    x_test = pd.read_csv('/rdf/db/public-tabular-datasets/adult/x_test.csv', index_col=0)

    y_train = pd.read_csv('/rdf/db/public-tabular-datasets/adult/y_train.csv', index_col=0)
    y_eval = pd.read_csv('/rdf/db/public-tabular-datasets/adult/y_eval.csv', index_col=0)
    y_test = pd.read_csv('/rdf/db/public-tabular-datasets/adult/y_test.csv', index_col=0)
    y_train = y_train[['label']]
    y_eval = y_eval[['label']]
    y_test = y_test[['label']]

    data_train = pd.concat([x_train, y_train], axis=1)
    data_train = data_train.sample(frac=frac, random_state=42)
    
    # validation and test sets which are not used
    data_eval = pd.concat([x_eval, y_eval], axis=1)
    data_test = pd.concat([x_test, y_test], axis=1)

    df = data_train
    knn = 5
    target = 'label'
    drop_features = []
    features = list(set(df.keys().tolist()) - set(drop_features + [target]))
    continous_features = [
        'age',
        'fnlwgt',
        'education-num',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
    ]
    sensitive_attribute = 'sex'
    privileged_group = 1

    X_balanced, y_balanced = balancing(
        data_train, 'label', 
        knn, sensitive_attribute, features, 
        drop_features, continous_features,
    )
    data_balanced = pd.concat([X_balanced, y_balanced], axis=1)

if __name__ == '__main__':
    main()
=======
>>>>>>> 92d31836236700bf0a18a9f7a60616bb1852aabd
