import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Plot() :
    path = os.getcwd()
    def __init__(self,data,savePlot=True,showPlot=False) :
        self.data = data
        self.savePlot = savePlot
        self.showPlot = showPlot
        self.path = self.path + '\\Plots'
        if not os.path.exists(self.path) :
            os.mkdir(self.path,0755)

    def accuracyPlot(self,field) :
        acc = self.data['shot_made_flag'].groupby(self.data[field]).mean()
        if self.savePlot or self.showPlot :
            acc.plot(kind='barh',title='shot_accuracy_with_'+field,figsize=(12,18))
            if self.savePlot :
                plt.savefig(self.path+'\\shot_accuracy_with_'+field+'.png')
                #print "plot saved as shot_accuracy_with_"+field+'.png'
            if self.showPlot :
                plt.show()

    def timePlot(self,step_size) :
        time_bins = np.arange(0, 721, step_size)
        intervals = pd.cut(self.data['total_time_remaining'], time_bins, right=False)
        group_by_intervals = self.data.groupby(intervals)
        # calculating mean gives % of correct shots in time intervals ex. (0s,30s) if step_size=30
        acc = group_by_intervals['shot_made_flag'].mean()
        if self.savePlot or self.showPlot :
            acc.plot(kind='bar', figsize=(12, 18), ylim=(0, 1),title='Shot accuracy over time remaining')
            if self.savePlot :
                plt.savefig(self.path+'\\shot_accuracy_vs_time_remaining_'+str(step_size)+'.png')
            if self.showPlot :
                plt.show()

'''
data = pd.read_csv('F:\Pervazive\data\data.csv')    
plot = Plot(data)
print plot.path
'''
