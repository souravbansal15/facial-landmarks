import matplotlib.pyplot as plt
import json
import pandas as pd

data_file = open('./data/ear_data.json','r')
ear_data = json.load(data_file)

plt.plot(ear_data['time'], ear_data['ear'])
plt.show()