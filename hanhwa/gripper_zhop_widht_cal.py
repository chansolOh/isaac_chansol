
import numpy as np
import matplotlib.pyplot as plt

import json 

with open("/home/uon/ochansol/isaac_chansol/hanhwa/robotiq_zhop_width_rate.json", "r") as f:
    json_data = json.load(f)

data = np.array(json_data["data"]).T

x = data[0]
y = data[1]
y -= y[0]
print(y)

mat = np.array([x**3, x**2, x, np.ones_like(x)]).T
mat_inv = np.linalg.pinv(mat)
a,b,c,d = mat_inv.dot(y)

# x_g = np.arange(x[0],x[-1], -1)
x_g = np.linspace(x[0], x[-1], len(x)+1)
print(a,b,c,d)
plt.plot(x,y)
plt.plot(x_g, a*(x_g**3)+ b*(x_g**2) + c*x_g + d )
plt.show()
# import pdb; pdb.set_trace()

json_data["params"]={
    "a" : a,
    "b" : b,
    "c" : c,
    "d" : d
}
print(json_data)
with open("/home/uon/ochansol/isaac_chansol/hanhwa/robotiq_zhop_width_rate.json", "w") as f:
    json.dump(json_data,f, indent=4 )