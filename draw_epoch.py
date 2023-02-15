import pandas as pd
from config import *
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame



data = pd.read_csv(MODEL_SAVE_CLASS_ADD.parent / "class_add" / "trained_opech_add.csv", sep=" ", names=["add", "epoch"])

fig, ax = plt.subplots(figsize=(20,10))
print(data)
data = data[:250]

data.plot(x = 'add', y = 'epoch', ax = ax)
plt.show()