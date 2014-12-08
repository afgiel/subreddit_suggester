import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

x = [.001, .05, .1, .5, 1.0, 1.5, 2.0, 5.0, 10.0]

train_tfidf = [.66, 0.79, 0.83, 0.91, .93, .95, .96, .98, .99]
train_count_tfidf = [.50, 0.80, .84, .91, .94, .95, .96, .98, .99]
train_binary = [.69, .88, .91, .96, .98, .98, .99, .99, 1.0]
train_count_binary = [.66, .88, .91, .96, .98, .98, .98, .99, .99]

dev_tfidf = [.63, 0.69, 0.71, 0.71, .71, .72, .71, .72, .73]
dev_count_tfidf = [.47, 0.73, 0.72, .74, .73, .73, .71, .72, .70]
dev_binary = [.67, .75, .76, .74, .75, .74, .72, .70, .72]
dev_count_binary = [.65, .77, .77, .78, .75, .72, .74, .71, .72]

fig = plt.figure(1)
ax = fig.add_subplot(111)

ax.plot(x, train_tfidf, color='g', linestyle='--', label='train tfidf')
ax.plot(x, train_count_tfidf, color='r', linestyle='--', label='train count tfidf')
ax.plot(x, train_binary, color='b', linestyle='--', label='train binary')
ax.plot(x, train_count_binary, color='m', linestyle='--', label='train count binary')

ax.plot(x, dev_tfidf, color='g', linestyle='-', label='dev tfidf')
ax.plot(x, dev_count_tfidf, color='r', linestyle='-', label='dev count tfidf')
ax.plot(x, dev_binary, color='b', linestyle='-', label='dev binary')
ax.plot(x, dev_count_binary, color='m', linestyle='-', label='dev count binary')

ax.set_xscale('log')

fontP = FontProperties()
fontP.set_size('small')
#, bbox_to_anchor=(0.7,-0.1
lgd = ax.legend(prop=fontP, loc=4)
plt.title('Varying Regularization')
plt.xlabel('Regularization (C)')
plt.ylabel('F1')
ax.grid('on')
fig.savefig('varying_regularization', bbox_extra_artists=(lgd,), bbox_inches='tight')
