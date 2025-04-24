import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from matplotlib.animation import FuncAnimation 

data1 = fetch_california_housing() 
df = pd.DataFrame(data1.data, columns=data1.feature_names)
df['MedHouseVal'] = data1.target 
data = df[:100]
x=np.array(data['MedInc'])
y = np.array(data['HouseAge'])

def linear(m,x,b):
    return m*x + b


#inisialisasi 
m= 5
b=2
learning_rate = 0.01
epoch = 100
#training

m_list_prediksi = []
b_list_prediksi = []
for i in range(epoch):
    y_pred = linear(m,x[i],b)
    y_actual = y[i]
    error = y_actual - y_pred

    #update m nya
    delta_m =  learning_rate * error * x[i]
    delta_b=  learning_rate * error
    m = m+ delta_m
    b =  b +delta_b 
    m_list_prediksi.append(m)
    b_list_prediksi.append(b)





fig, ax = plt.subplots()
scatter = ax.scatter(x, y, s=10, label="Data")
line, = ax.plot([], [], color='red', label='Regression Line')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.legend()

# Fungsi animasi
def animate(i):
    y_pred_line = linear(m_list_prediksi[i], x, b_list_prediksi[i])
    line.set_data(x, y_pred_line)
    return line,

# Animasi
ani = FuncAnimation(fig, animate, frames=epoch, interval=1)
plt.show()

pendapatan = 1.9

umur_rumah = linear(m, pendapatan, b)
print(f"Prediksi harga rumah untuk pendapatan {pendapatan} adalah {umur_rumah}")