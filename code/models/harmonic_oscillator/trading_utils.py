
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Tuple, Sequence


def value_at_risk(samples, alpha=5):
    '''
    samples: numpy array of size (batch , samples)
    retunrs var: numpy array of size (batch , 1)
    '''
    samples.sort(axis=-1)
    var = np.percentile(samples, alpha, axis=-1)
    return var


def expected_shortfall(samples, alpha=5):
    '''
    expected left tail risk

    samples: numpy array of size (batch , samples)
    '''
    var = value_at_risk(samples, alpha=alpha)
    es = np.zeros(np.shape(var))
    for i in range(len(var)):
        es[i] = np.mean([s for s in samples[i, :] if s < var[i]])
    return -1.*es

def unconditional_expected_shortfall(samples, alpha=5):
    '''
    expected left tail risk

    samples: numpy array of size (batch , samples)
    '''
    var = value_at_risk(samples, alpha=alpha)
    es = np.zeros(np.shape(var))
    for i in range(len(var)):
        es[i] = np.mean(np.divide([s if s < var[i] else 0 for s in samples[i, :]], alpha/100.))
    return -1.*es


def likelihood_ratio(pihat, pi, T1, T0):
    true = pi**T1 *(1-pi)**T0
    model = pihat**T1 *(1-pihat)**T0
    return true/model 

def z1_score(observations, violations, es, T1, alpha):
    true_loss = violations*observations
    part = np.sum(true_loss/es)
    return part/T1 + 1

def z2_score(observations, violations, es_unc, T, alpha):
    true_loss = violations*observations
    expected_loss = T*alpha*es_unc
    ratio = true_loss/expected_loss
    return np.sum(ratio+1.)


def get_next_or_none(iterable):
    try:
        return next(iterable)
    except StopIteration:  # there is no tsell
        return None

def compute_roi(timeseries: Sequence[float], events: Sequence[str], transaction_cost, violations) -> float:
    """
    Given a timeseries and a list of events, compute the return on investment
    Note: only buys once every tick, but sells everything within 1 tick

    :param timeseries: A array of n_steps data points
    :param events: A List of events
        e.g. ['hold', buy', 'sell','buy']
    :return: total return made after excecuting the events on timeseries

    """

    bought = 0
    sold = 0
    roi = []
    profit = 0
    # plt.plot(timeseries, alpha=0.5, c='b')
    # plt.plot(var, alpha=0.5, c='r')
    # plt.plot(es, alpha=0.5, c='m')
    # for i in range(len(violations)):
    #     if violations[i]==1:
    #         plt.scatter(i, timeseries[i], marker='*', s=70, color='r')

    for i in range(len(events)):
        if events[i]=='buy':
            sold = 0
            # plt.scatter(i, timeseries[i], marker='x', s=70 )
            bought = timeseries[i] + timeseries[i]*transaction_cost
            profit -= bought
        elif events[i]=='sell':
            # plt.scatter(i, timeseries[i], marker='o', s=70 )
            sold = timeseries[i] - timeseries[i]*transaction_cost
            roi.append( (sold-bought)/bought )
            profit += sold

    # if time is up before able to sell, do not discount
    if sold == 0 and bought>0:
        profit += bought

    # plt.show()

    return np.sum(roi), profit

