import numpy as np

quest_1 = [0.9168888888888889, 0.9142222222222223, 0.9235555555555556]

vqbet_1 = [0.8888888888888888, 0.876, 0.8333333333333334]


print(np.median(quest_1), np.std(quest_1) / np.sqrt(3))
print(np.median(vqbet_1), np.std(vqbet_1) / np.sqrt(3))