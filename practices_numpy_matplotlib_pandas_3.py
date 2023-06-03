import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)
x = np.random.rand(100) * 10
y = 3 * x + np.random.randn(100) * 2


data = {'X': x, 'Y': y}
df = pd.DataFrame(data)


plt.scatter(df['X'], df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Doğrusal İlişki')
plt.show()


correlation = df['X'].corr(df['Y'])
linear_fit = np.polyfit(df['X'], df['Y'], 1)
slope = linear_fit[0]
intercept = linear_fit[1]

print('Korelasyon:', correlation)
print('Eğim:', slope)
print('Düzey:', intercept)
