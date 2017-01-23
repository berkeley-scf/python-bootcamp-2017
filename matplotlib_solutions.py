import matplotlib.pyplot as plt

conts = np.unique(dat['continent'])
n = len(conts)
colors = ['b','g','r','c','m']
# plt.cm.rainbow(np.linspace(0,1,n))
# ['blue','green','red','cyan','magenta','yellow','black']
colscheme = dict(zip(conts, colors))
col = [colscheme[cont] for cont in dat2007['continent']]

plt.scatter(dat2007['gdpPercap'], dat2007['lifeExp'], marker = 'o', c = col)
plt.show()
