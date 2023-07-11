import numpy as np
import matplotlib.pyplot as plt
import os.path

comparison=True
save_figure=True

data_path='data/SP1_test/TC1_SV'
figure_name='SP1_TC1_energy_SV_SE.png'
file1 = os.path.join(data_path,'energy.csv')
label1='SV'

time   = np.loadtxt(file1, usecols=0)
en_tot = np.loadtxt(file1, usecols=1)

en = []
for i in range(len(en_tot)):
    en.append(en_tot[i]-en_tot[0])

if save_figure:
    path=os.path.join(data_path,'figures')
    try: 
        os.mkdir(path) 
    except OSError as error: 
        print(error)
    
    save_path=os.path.join(path, figure_name)

if comparison:
    path2='data/SP1_test/TC1_SE'
    file2 = os.path.join(path2,'energy.csv')
    label2='SE'

    time2   = np.loadtxt(file2, usecols=0)
    en_tot2 = np.loadtxt(file2, usecols=1)

    en2 = []
    for j in range(len(en_tot2)):
        en2.append(en_tot2[j]-en_tot2[0])

fig, ax1=plt.subplots()
fig.set_size_inches(8,4)
fig.set_tight_layout(True)

ax1.set_title('(a) Energy variations (CG2)', fontsize=18)

ax1.set_xlabel('$t$',fontsize=14)
ax1.set_ylabel('$E(t)-E(t_0)$',fontsize=14)

ax1.ticklabel_format(axis='y',scilimits=(0,0))
ax1.plot(time,en,'bo',label=label1)

if comparison:
    ax1.plot(time2,en2,'ro',label=label2)
    ax1.legend(loc='upper left',fontsize=14)
ax1.grid()

if save_figure:
    plt.savefig(save_path,dpi=300)
else:
    plt.show()