import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

x = [12, 25, 50, 100, 150, 200]

train = [.36, .38, .40, .43, .42, .40]
dev = [.36, .36, .39, .44, .42, .40]

fig = plt.figure(1)
ax = fig.add_subplot(111)

ax.plot(x, train, color='b', linestyle='--', label='train')
ax.plot(x, dev, color='b', linestyle='-', label='dev')

fontP = FontProperties()
fontP.set_size('small')
#, bbox_to_anchor=(0.7,-0.1
lgd = ax.legend(prop=fontP, loc=2)
plt.title('Determining Optimal Number of Topics')
plt.xlabel('Number of Topics')
plt.ylabel('F1')
ax.grid('on')
fig.savefig('topics_lda', bbox_extra_artists=(lgd,), bbox_inches='tight')
