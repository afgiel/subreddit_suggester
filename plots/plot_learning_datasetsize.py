import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

x = [12*200*.9, 12*400*.9, 12*600*.9, 12*800*.9, 12*1000*.9]

#train_binary = [0.78, 0.89, 0.95, 0.97, 0.97, 0.98, 0.98]
#train_tfidf = [0.73, 0.80, 0.86, 0.88, 0.91, 0.91, 0.93]
#train_sentiment = [0.73, 0.80, 0.86, 0.89, 0.91, 0.91, 0.93]
train_count = [0.94, 0.92, 0.91, 0.90, 0.89]

#dev_binary = [0.63, 0.66, 0.68, 0.71, 0.70, 0.70, 0.70]
#dev_tfidf = [0.66, 0.68, 0.68, 0.73, 0.70, 0.72, 0.71]
#dev_sentiment = [0.67, 0.69, 0.69, 0.70, 0.72, 0.71, 0.72]
dev_count = [0.66, 0.68, 0.72, 0.70, 0.71]

fig = plt.figure(1)
ax = fig.add_subplot(111)

#ax.plot(x, train_binary, color='g', linestyle='--', label='train binary')
#ax.plot(x, train_tfidf, color='r', linestyle='--', label='train tfidf')
#ax.plot(x, train_sentiment, color='b', linestyle='--', label='train sentiment')
ax.plot(x, train_count, color='m', linestyle='--', label='train count')

#ax.plot(x, dev_binary, color='g', linestyle='-', label='dev binary')
#ax.plot(x, dev_tfidf, color='r', linestyle='-', label='dev tfidf')
#ax.plot(x, dev_sentiment, color='b', linestyle='-', label='dev sentiment')
ax.plot(x, dev_count, color='m', linestyle='-', label='dev count')

fontP = FontProperties()
fontP.set_size('small')
#, bbox_to_anchor=(0.7,-0.1
lgd = ax.legend(prop=fontP, loc=5)
plt.title('Learning Curve')
plt.xlabel('Number of Training Samples')
plt.ylabel('F1')
ax.grid('on')
fig.savefig('learningcurvedatasize', bbox_extra_artists=(lgd,), bbox_inches='tight')