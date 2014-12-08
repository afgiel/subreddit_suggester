import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

x = [500, 1000, 2000, 3000, 4000, 5000, 10000]

binary = [0.71, ]
tfidf = [,]
sentiment = [,]

fig = plt.figure(1)
ax = fig.add_subplot(111)

ax.plot(x, binary, color='g', linestyle='-', label='binary')
ax.plot(x, tfidf, color='r', linestyle='-', label='tfidf')
ax.plot(x, sentiment, color='b', linestyle='-', label='sentiment')


fontP = FontProperties()
fontP.set_size('small')
lgd = ax.legend(prop=fontP, bbox_to_anchor=(0.7,-0.1))
plt.title('Learning Curve')
plt.xlabel('Number of Unigram Features')
plt.ylabel('F1')
ax.grid('on')
fig.savefig('learningcurve', bbox_extra_artists=(lgd,), bbox_inches='tight')