import os
import multiprocessing
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from adapt.instance_based import KMM, KLIEP, NearestNeighborsWeighting
from adapt.metrics import linear_discrepancy, make_uda_scorer

from datasets import open_uci_dataset
from utils import (preprocessor, CustomEarlyStopping, EarlyStoppingByLossVal,
                   remove_columns, get_sample_bias_X, get_sample_bias_y)
from _iwn import IWN


def get_estimator(shape=2, activation=None):
    model = Sequential()
    model.add(Dense(100, activation="relu", input_shape=(shape,)))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(1, activation=activation))
    model.compile(loss="mse", optimizer=Adam(0.001))
    return model


def run_kmm(name, Xs, X, ys, random_state, results_folder):

    kmm = KMM(
        estimator=Ridge(1.), 
        Xt=X,
        kernel="rbf",
        gamma=0.,
        verbose=0
    )

    score = make_uda_scorer(linear_discrepancy, Xs, X)

    gs = GridSearchCV(kmm, {"gamma": [10**(i-4) for i in range(8)]},
                      scoring=score, refit=False, n_jobs=8,
                      return_train_score=True, cv=5, verbose=0)
    gs.fit(Xs, ys);

    best_params = gs.cv_results_["params"][gs.cv_results_["mean_train_score"].argmin()]
    kmm.set_params(**best_params);
    weights = kmm.fit_weights(Xs, X);
    np.save("%s/%s_weights_kmm_%s.npy"%(results_folder, name, str(random_state)), weights.ravel())


def run_kliep(name, Xs, X,  ys, random_state, results_folder):

    kliep = KLIEP(Ridge(1.),
                  gamma=[10**(i-4) for i in range(8)],
                  cv=5,
                  max_iter=1000,
                  lr=0.01)

    weights = kliep.fit_weights(Xs, X);
    
    np.save("%s/%s_weights_kliep_%s.npy"%(results_folder, name, str(random_state)), weights.ravel())
    
    
def run_nnw(name, Xs, X, ys, random_state, results_folder):

    nnw = NearestNeighborsWeighting(Ridge(1.),
                                    Xt=X,
                                    n_neighbors=1,
                                    verbose=0)

    score = make_uda_scorer(linear_discrepancy, Xs, X)

    gs = GridSearchCV(nnw, {"n_neighbors": [1, 5, 10, 20, 50, 100]},
                      scoring=score, refit=False, n_jobs=8,
                      return_train_score=True, cv=5, verbose=0)
    gs.fit(Xs, ys);

    best_params = gs.cv_results_["params"][gs.cv_results_["mean_train_score"].argmin()]
    nnw.set_params(**best_params);
    weights = nnw.fit_weights(Xs, X);
    
    np.save("%s/%s_weights_nnw_%s.npy"%(results_folder, name, str(random_state)), weights.ravel())


def run_iwn(name, Xs, X, ys, random_state, results_folder):
    
    epochs = min(int(np.ceil(5*10**5 / len(X))), 1000)
    patience = min(int(np.ceil(2*10**5 / len(X))), 400)
    first_patience = 1
    fit_params = dict(epochs=epochs, batch_size=256, verbose=0,
                      callbacks =[CustomEarlyStopping(min_delta=1e-6,
                                                      patience=patience,
                                                      first_patience=first_patience)])
    pretrain_params = dict(pretrain__epochs=int(np.ceil(400000 / max(250, len(Xs)))),
                           pretrain__callbacks=[EarlyStoppingByLossVal(value=1e-3)])
    iwn = IWN(get_estimator(), optimizer=Adam(0.001), **pretrain_params)
    weights = iwn.fit_weights(Xs, X, **fit_params)
    
    np.save("%s/%s_weights_iwn_%s.npy"%(results_folder, name, str(random_state)), weights.ravel())


NAMES = [
 'yacht',
 'forest',
 'airfoil',
 'sml',
 'parkinson',
 'power',
 'superconductivity',
 'protein',
 'blog',
 'ctscan',
 'virus',
 'gpu',
 'airquality',
 'road3d',
 'housepower']
    

if __name__ == "__main__":
    
    results_folder = "results"
    time_file = "time"
    
    os.makedirs(results_folder, exist_ok=True)
    
    df_time = pd.DataFrame(dict(method=[""], name=[""], random_state=[0], time=[500]))
    df_time.to_csv("%s.csv"%time_file)
    
    for random_state in range(10):
        for name in NAMES:
            for method, func in zip(["iwn", "kmm", "kliep", "nnw"], [run_iwn, run_kmm, run_kliep, run_nnw]):
            
                np.random.seed(random_state)
                tf.random.set_seed(random_state)

                X, y = open_uci_dataset(name, online=True)
                print(name, len(X), [y[c].nunique() for c in y.columns])
                X, y = remove_columns(X, y, max_cat_perc=0.5, max_cat=1000)

                X = preprocessor.fit_transform(X)
                y = y.values

                sample_bias, pca1 = get_sample_bias_X(X) # Uncomment next line to make sample bias on y
                # sample_bias, pca1 = get_sample_bias_y(y)
                biased_index = np.random.choice(len(X), len(X), p=sample_bias/sample_bias.sum())
                Xs = X[biased_index]
                ys = y[biased_index]

                print("Random state %i - Name %s - Method %s"%(random_state, name, method))
                
                df_time = pd.read_csv("%s.csv"%time_file, index_col=0)
                max_time = df_time.loc[(df_time.method == method) & (df_time.name == name), "time"].max()
                
                t0 = time.time()

                if not max_time >= 500:
                    process = multiprocessing.Process(target=func, name="Foo", args=(name, Xs, X, ys, random_state, results_folder))
                    process.start()

                    process.join(500)

                    # If thread is active
                    if process.is_alive():
                        print("foo is running... let's kill it...")

                        # Terminate foo
                        process.terminate()
                        process.join()

                t1 = time.time()

                df_time = df_time.append(pd.DataFrame(dict(method=[method], name=[name],
                                                           random_state=[random_state],
                                                           time=[t1-t0])), ignore_index=True)
                df_time.to_csv("%s.csv"%time_file)

                print("Time : %.2f"%(t1-t0))

                try:
                    weights = np.load("%s/%s_weights_%s_%s.npy"%(results_folder, name, method, str(random_state)))

                    alphas = [10**(i-4) for i in range(8)]

                    lr = RidgeCV(alphas=alphas, cv=None)
                    lr.fit(Xs, ys)
                    np.save("%s/%s_scores_unif_%s.npy"%(results_folder, name, str(random_state)),
                            mean_absolute_error(lr.predict(X), y, multioutput="raw_values"))

                    lr2 = RidgeCV(alphas=alphas, cv=None)
                    lr2.fit(Xs, ys, sample_weight=weights/weights.mean())
                    np.save("%s/%s_scores_%s_%s.npy"%(results_folder, name, method, str(random_state)),
                            mean_absolute_error(lr2.predict(X), y, multioutput="raw_values"))

                    print("Time : %.2f"%(t1-t0))
                    print("Unif", mean_absolute_error(lr.predict(X), y, multioutput="raw_values"))
                    print(method, mean_absolute_error(lr2.predict(X), y, multioutput="raw_values"))
                    print("frac", mean_absolute_error(lr2.predict(X), y)/mean_absolute_error(lr.predict(X), y))
                except:
                    raise
                    print("Time : %.2f"%(t1-t0))
                    print(method, "Too long...")