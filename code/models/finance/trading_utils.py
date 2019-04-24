
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
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
    es = np.mean([s for s in samples if s < var])
    return -1.*es

def unconditional_expected_shortfall(samples, alpha=5):
    '''
    expected left tail risk

    samples: numpy array of size (batch , samples)
    '''
    var = value_at_risk(samples, alpha=alpha)
    es = np.mean(np.divide([s if s < var else 0 for s in samples], alpha/100.))
    return -1.*es


def likelihood_ratio(pihat, pi, T1, T0):
    true = pi**T1 *(1-pi)**T0
    model = pihat**T1 *(1-pihat)**T0
    return true/model 

def z1_score(observations, violations, es, T1, alpha):

    print('Z1: --------------------------------------')
    true_loss = violations*observations
    part = np.sum(true_loss/es)
    return part/T1 + 1

def z2_score(observations, violations, es_unc, T, alpha):
    print('Z2: --------------------------------------')
    true_loss = violations*observations
    expected_loss = T*alpha*es_unc
    ratio = true_loss/expected_loss
    # print(np.sum(true_loss, axis=-1))
    # print(np.sum(expected_loss, axis=-1))
    return np.sum(ratio)+1.


def mark_trade(trade_time, trade_price, trade_type, color='k', label_prefix=''):
    plt.plot(trade_time, trade_price, marker='x' if trade_type == 'buy' else 'o', markersize=10, color=(0, 0, 0, 0),
             markeredgecolor=color, markeredgewidth=3, label=label_prefix + trade_type)


def get_next_or_none(iterable):
    try:
        return next(iterable)
    except StopIteration:  # there is no tsell
        return None


def transaction_chooser(futures_stream, transaction_cost, riskiness=0.1, initial_have_state=False):
    have_state = initial_have_state

    for t, futures in enumerate(futures_stream):
        expected_future = np.mean(futures, axis=0)  # (samples, time)
        x0 = expected_future[0]
      
        if not have_state:
            # check when and if there is a profitable sell moment
            t_sell = get_next_or_none(tau for tau, m in enumerate(expected_future) if m > x0 + transaction_cost
                                      and riskiness > -1 * expected_shortfall(x0, futures[:, tau]))

            if t_sell is not None:
                # ES = expected_shortfall(x0, futures[:, t_sell])
                # check if until that sell moment arrives, there is a better buy moment
                if (t_sell == 1 or expected_future[1:t_sell].min() > x0):  # and riskiness > (-1*ES):
                    # mark_trade(trade_time=t, trade_price=x0, trade_type='buy')
                    have_state = True

        else:
            # check if there is a moment when buying a new share is cheaper than keeping this one
            t_buy = get_next_or_none(tau for tau, m in enumerate(expected_future) if m < x0 - transaction_cost)
            # check if until that moment arrives, there is a better sell moment
            if t_buy is not None:
                if t_buy == 1 or not expected_future[1:t_buy].max() > x0:
                    # mark_trade(trade_time=t, trade_price=x0, trade_type='sell')
                    have_state = False


def make_futures_stream(model, n_samples=10, n_steps=100):
    for _ in itertools.count(0):
        futures = []
        context = model.simulate(n_steps=1, rng=np.random.RandomState(0))
        for i in range(1, n_samples+1):
            this_model = model.clone()
            future = this_model.simulate(n_steps=n_steps, rng=np.random.RandomState(i))
            future.insert(0, context[-1])
            futures.append(future)
        yield np.array(futures)

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
    plt.figure()
    plt.plot(timeseries, alpha=0.5, c='b')
    P = []
    R = []
    # for i in range(len(violations)):
    #     if violations[i]==1:
    #         plt.scatter(i, timeseries[i], marker='*', s=70, color='r')

    for i in range(len(events)):
        if events[i]=='buy':
            sold = 0
            plt.scatter(i, timeseries[i], marker='x', s=70 )
            bought = timeseries[i] + timeseries[i]*transaction_cost
            profit -= bought
        elif events[i]=='sell':
            plt.scatter(i, timeseries[i], marker='o', s=70 )
            sold = timeseries[i] - timeseries[i]*transaction_cost
            roi.append( (sold-bought)/bought )
            R.append(roi)
            profit += sold
            P.append(profit)
            print(profit)

    # if time is up before able to sell, do not discount
    if sold == 0 and bought>0:
        profit += bought

    plt.show()
    plt.figure('profit')
    plt.plot(P)
    plt.show()
    plt.figure('ROI')
    plt.plot(P)
    plt.show()

    return np.sum(roi), profit



    # current_price = timeseries[0]
    # returns = [(p/current_price)-1 for p in timeseries]
    # final_returns = []
    # bought = 0 # keep track of how much you have bought 
    # wallet = 0 # keep track of how many "coins" you have bought
    # plt.figure('events')
    # plt.plot(timeseries)
    # for i in range(len(events)):
    #     event = events[i]
    #     if event=='buy':
    #         bought += ((returns[i]+1)*(1+transaction_cost))
    #         wallet += 1
    #         plt.scatter(i, timeseries[i], c='g')
    #     if event=='sell' and wallet > 0:
    #         # final_returns.append(wallet*[timeseries[i]])
    #         plt.scatter(i, timeseries[i], c='r')
    #         roi +=  ((wallet*((returns[i]+1)*(1-transaction_cost))) - bought)
    #         bought = 0 # empty the wallet
    #         wallet = 0

    # sharpe_ratio = roi / np.std(returns)
    # if show:
    #     plt.show()
    # return roi, sharpe_ratio

