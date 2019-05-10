
import numpy as np

SEED = 1234567890
rng = np.random.RandomState( SEED )

print(rng.randint(10,20))