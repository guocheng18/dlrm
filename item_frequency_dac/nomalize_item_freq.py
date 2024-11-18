# give the item frequency file, normalize the frequency to 0-1
# the item frequency file is a npz file, there are 26 numpy arrays inside, you should idenpendently normalize
# each numpy array.

import numpy as np

itemfreq = np.load("itemfreq.npz")

for i in range(26):
    itemfreq[i] = itemfreq[i] / np.max(itemfreq[i])

np.savez("itemfreq_normalized.npz", *itemfreq)