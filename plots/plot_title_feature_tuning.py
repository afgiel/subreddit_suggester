import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.font_manager import FontProperties

title = np.array([500, 750, 1000, 1500, 2000])
text = np.array([250, 500, 1000, 2000, 3000])

result = np.array([
# title 500
[.69, .71, .72, .73, .72],
# title 750
[.68, .70, .73, .74, .72],
# title 1000
[.69, .72, .72, .72, .73],
# title 1500
[.68, .70, .71, .71, .73],
#title 2000
[.68, .71, .69, .73, .71]
])

X, Y = np.meshgrid(title, text)
Z = result.T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=1, antialiased=False)
ax.set_xlabel('Number Title Unigram Features')
ax.set_ylabel('Number Body Unigram Features')
ax.set_zlabel('F1')
ax.set_title('Tuning Title Split Parameters')
plt.show()