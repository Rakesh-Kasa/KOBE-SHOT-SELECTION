import plot
from plot import Plot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class dataAnalysis () :
    def __init__(self,data) :
        self.data = data
        self.discarded_columns = []
        
    def analyse(self,savePlot=True,showPlot=False) :
        '''
        game_event_id, game_id, lat, lon, team_id, team_name can be discarded
        lat & lon are proportional to loc_x, loc_y --> can be seen by plotting against one other
        game_event_id, game_id, team_id, team_name are same
        so can be dropped out from our data
        '''
        d_cols = ['game_event_id','game_id','team_name','team_id','lat','lon']
        '''
        minutes_remaining and seconds_remaining can be combined in to one
        '''
        self.data['total_time_remaining'] = self.data['minutes_remaining']*60 + self.data['seconds_remaining']
        d_cols.append('minutes_remaining')
        d_cols.append('seconds_remaining')
        '''
        matchup can be converted as int with home--> 0 & away-->1
        '''
        data = self.data
        data['away'] = data.matchup.str.contains('@').astype(int)
        d_cols.append('matchup')    
        plot = Plot(data,savePlot=savePlot,showPlot=showPlot)
        if savePlot or showPlot :
            sns.countplot(x='away',hue='shot_made_flag',data=data)
            if savePlot :
                plt.savefig(plot.path+'\\shots_made_in_home_away.png')
            if showPlot :
                plt.show()
        
        plot.timePlot(step_size=30)
        plot.accuracyPlot('away') # Greater accuracy in home games 
        '''
        plot of accuracy over time shows more change during last 30 seconds in the
        remaining times accuracy is almost the same, might be a useful feature
        '''
        plot.accuracyPlot('period') #seems to be consistent over the period

        '''
        much change in action_type and combined_shot_type, may be he prefers one shot over other
        '''
        plot.accuracyPlot('action_type') 
        plot.accuracyPlot('combined_shot_type')
        '''
        Looks like accuracy is almost zero for distances > 45
        Exploring this in detail
        '''
        plot.accuracyPlot('shot_distance')
        
        data['distance_mod'] = data.apply(lambda x:'distance<45' if x['shot_distance']<45 else 'distance>45',axis=1)
        if savePlot or showPlot :
            sns.countplot(x='distance_mod',hue='shot_made_flag',data=data)
            if savePlot :
                plt.savefig(plot.path+'\\shots_made_with_distance.png')
            if showPlot :
                plt.show()
        
        '''
        Looks like he took less number of shots from distance > 45, we can drop those
        Modifying shot_distance to distance by dropping distances > 45
        '''
        data['distance'] = data.apply(lambda x:x['shot_distance'] if x['shot_distance']<45 else 46,axis=1)    
        d_cols.append('shot_distance')
        d_cols.append('distance_mod')
        plot = Plot(data,savePlot=savePlot,showPlot=showPlot)
        '''
        Shot accuracy with shot_zone-> area,basic,range
        The area from where the ball is shooted should be an effective feature to
        say if a goal is made or not
        '''
        plot.accuracyPlot('shot_zone_area') #looks like kobe is good from center and wide change in accuracy
        plot.accuracyPlot('shot_zone_basic') # more accuracy from RA
        plot.accuracyPlot('shot_zone_range') # Good when near to basket

        plot.accuracyPlot('opponent') #kobe plays well with some opponents over other
        plot.accuracyPlot('playoffs') # some difference in regular and playoffs
        plot.accuracyPlot('season') # perfomance has dropped in recent years may be due to his injury

        data = data.drop(d_cols,axis=1)
        self.data = data
        self.discarded_columns = d_cols
        features = ['action_type','combined_shot_type','game_date', 'loc_x', 'loc_y', 'period', 'playoffs','season', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range','opponent','total_time_remaining', 'away', 'distance']
        return self.data,features
'''       
data = pd.read_csv('F:\Pervazive\data\data.csv')    
d = dataAnalysis(data)
data,f = d.analyse(savePlot=False,showPlot=False)
print d.discarded_columns
''' 
