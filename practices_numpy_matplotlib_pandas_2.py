import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sinüs Grafiği')
plt.grid(True)
plt.show()

data = {'Name': ['Ahmet', 'Mehmet', 'Rabia', 'Bugra'],
        'Age': [25, 30, 35, 28],
        'City': ['İstanbul', 'Konya', 'Ankara', 'Bursa']}

df = pd.DataFrame(data)
print(df)
