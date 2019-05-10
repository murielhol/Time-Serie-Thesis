
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle




def point(df):
    '''
    deterministicly predicted images
    '''
    # denormalize 
    df = 0.5* (1+df)
    print(df.head())

    fig, axarr = plt.subplots(2,10, figsize = (9.5, 2.5))
    for i in range(10):
        ax_real = axarr[0, i]
        ax_fake = axarr[1, i]
        real = np.reshape(df[str(i)+'_real'], (28,28))
        fake = np.reshape(df[str(i)+'_fake'], (28,28))
        ax_real.imshow(real, cmap='gray')
        ax_fake.imshow(fake, cmap='gray')
        ax_fake.axhline(y=15, c='w')
        ax_fake.set_xticks([])
        ax_fake.set_yticks([])
        ax_real.set_yticks([])
        ax_real.set_xticks([])
    axarr[0,4].set_title('Data')
    axarr[1,4].set_title('Model')
    plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0, wspace=0, hspace=0.5)
    plt.savefig('wn/images/point_digits_wn.pdf')
    plt.show()

def sampled(df):
    '''
    stochasticly predicted images
    '''
    sims = [1, 3, 7]
    # denormalize 
    df = 0.5* (1+df)
    fig, axarr = plt.subplots(3,10, figsize = (10, 3))
    for i in range(10):
        for j in range(3):
            ax = axarr[j, i]
            fake = np.reshape(df[str(i)+'_fake_seed'+str(sims[j])], (28,28))
            ax.imshow(fake, cmap='gray')
            ax.axhline(y=15, c='w')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig('swn/images/simulation_digits_swn.pdf')
    plt.show()




# def sampled(df)


# dswn = pd.read_csv('digits_dswn.csv')
# swn = pd.read_csv('digits_swn.csv')
# dwn = pd.read_csv('digits_dwn.csv')

wn = pd.read_csv('wn/images/digits.csv')
point(wn)

# wn = pd.read_csv('swn/images/digits_swn.csv')
# point(wn)

# wn = pd.read_csv('swn/images/simulations_swn.csv')
# sampled(wn)

