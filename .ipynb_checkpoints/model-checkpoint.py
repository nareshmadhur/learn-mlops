import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

with open('IDEA_NS.pkl', 'rb') as file:
    df = pickle.load(file)

Y = df.close
Y.index = df.date

Y_fr = Y.reindex(pd.date_range(Y.index.min(), Y.index.max()))
Y_fr.fillna(Y_fr.rolling(9, min_periods=1).mean(), inplace=True)


# ss = StandardScaler()
# Y_std = ss.fit_transform(Y_fr.values.reshape(-1,1))

def arima_cross_val(df, order, s_order, errormetric, start=10, steps=10, horizon=10):
    errors = np.array([])

    for i in np.arange(start, df.shape[0] - horizon, steps):
        train = Y_fr.iloc[:int(i)]
        test = Y_fr.iloc[int(i):int(i) + horizon]
        model = SARIMAX(train, order=order, seasonal_order=s_order)
        arima_fit = model.fit(disp=0)
        pred = arima_fit.predict(start=test.index[0], end=test.index[-1])
        score = errormetric(pred.values, test.values)
        errors = np.append(errors, score)
        print("Error for iteration - {} = {}".format(i, score))
    return errors


def mae(pred, act):
    try:
        return abs(pred - act).sum() / pred.shape[0]
    except Exception as e:
        print("Error while calculating MAE - " + str(e))


def getgraph_SARIMAX(order, s_order, model):
    order = order
    s_order = s_order

    # res = arima_cross_val(Y_fr, order=order, s_order=s_order, errormetric=mae)

    tr = Y_fr[:int(Y_fr.shape[0] * 0.8)]
    ts = Y_fr[int(Y_fr.shape[0] * 0.8):]

    new_model = model(tr, order=order, seasonal_order=s_order).fit()
    plt.plot(Y_fr)
    plt.plot(new_model.forecast(steps=ts.shape[0]))
    return


def getgraph_ARIMAX(order, model):
    order = order

    # res = arima_cross_val(Y_fr, order=order, s_order=s_order, errormetric=mae)

    tr = Y_fr[:int(Y_fr.shape[0] * 0.8)]
    ts = Y_fr[int(Y_fr.shape[0] * 0.8):]

    new_model = model(tr, order=order).fit()
    plt.plot(Y_fr)
    plt.plot(new_model.forecast(steps=ts.shape[0]))
    return
