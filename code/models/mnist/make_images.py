
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

paths = ['wn', 'swn']

################################  TSNE images ######################

def make_tsne(path):
    fake = pickle.load( open( path+"/images/FakeY.p", "rb" ) )
    real = pickle.load( open( path+"/images/RealY.p", "rb" ) )
    labels = pickle.load( open( path+"/images/labels.p", "rb" ) )
    for i in range(10):
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.scatter(real[i*200:i*200+200, 0], real[i*200:i*200+200, 1], c=labels[i*200:i*200+200], s=20, alpha=0.4)
        ax2.scatter(fake[i*200:i*200+200, 0], fake[i*200:i*200+200, 1], c=labels[i*200:i*200+200], s=20, alpha=0.4)
        plt.show()



make_tsne(paths[1])

