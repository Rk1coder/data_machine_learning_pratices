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

data = {'Name': ['John', 'Emily', 'Ryan', 'Jane'],
        'Age': [25, 30, 35, 28],
        'City': ['New York', 'Paris', 'London', 'Sydney']}

df = pd.DataFrame(data)
print(df)
