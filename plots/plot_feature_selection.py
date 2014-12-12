import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

x = [50, 100, 250, 500, 1000, 3000, 5000, 10000]

train_mi = [.46, .52, .61, .69, .74, .82, .84, .87] 
train_pca = [.61, .65, .69, .72, .77, .84, .87, .90]
train_random = [.06, .04, .12, .16, .21, .44, .48, .58]

dev_mi = [.45, .48, .58, .66, .69, .70, .70, .72]
dev_pca = [.60, .64, .64, .68, .67, .69, .70, .72]
dev_random = [.06, .03, .11, .15, .16, .34, .41, .45]

fig = plt.figure(1)
ax = fig.add_subplot(111)

ax.plot(x, train_mi, color='g', linestyle='--', label='train mi')
ax.plot(x, train_pca, color='r', linestyle='--', label='train pca')
ax.plot(x, train_random, color='b', linestyle='--', label='train random')

ax.plot(x, dev_mi, color='g', linestyle='-', label='dev mi')
ax.plot(x, dev_pca, color='r', linestyle='-', label='dev pca')
ax.plot(x, dev_random, color='b', linestyle='-', label='dev random')

ax.set_xscale('log')

fontP = FontProperties()
fontP.set_size('small')
#, bbox_to_anchor=(0.7,-0.1
lgd = ax.legend(prop=fontP, loc=4)
plt.title('Reducing Dimennsionality of Feature Vectors')
plt.xlabel('Number of dimensions')
plt.ylabel('F1')
ax.grid('on')
fig.savefig('dimension_of_vectors_plot', bbox_extra_artists=(lgd,), bbox_inches='tight')
