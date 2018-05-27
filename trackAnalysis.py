import utils
import params as param
import networkx as nx

import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as td
import math
import matplotlib.pyplot as plt
import numpy as np


class TA:

    def __init__(self):
        prm = param.params()

        dftr = pd.read_csv('trackaaa.csv', engine='python')
        dftr = dftr.sort_values(by=['RiderID','Date','Time'])

        dftr = dftr[dftr['Time'] <= 223000]
        dftr = dftr[dftr['Time'] >= 103000]

        dftr['TS'] = dftr.apply(lambda x: utils.putTimeinDate(utils.parseDate(str(x['Date'])), utils.parseDate(str(x['Time']))), axis=1)

        dftr = dftr.drop(['Date','Time'], axis=1)
        dftr = dftr[['TS','RiderID','Lon.','Lat.','Status']]

        dftr2 = dftr.shift(-1)
        dftr2['TS-1'] = dftr['TS']
        dftr2['RiderID-1'] = dftr['RiderID']

        dftr2['Lon.-1'] = dftr['Lon.']
        dftr2['Lat.-1'] = dftr['Lat.']
        dftr2 = dftr2[:-1]

        dftr2['TimeDelta'] = dftr2.apply(lambda x: utils.getTrackTimeDelta(x), axis=1)
        dftr2['Distance'] = dftr2.apply(lambda x: utils.getTrackDist(x), axis=1)

        dftr2 = dftr2[dftr2['Lon.'] <= prm.lonMax]
        dftr2 = dftr2[dftr2['Lon.'] >= prm.lonMin]
        dftr2 = dftr2[dftr2['Lat.'] <= prm.latMax]
        dftr2 = dftr2[dftr2['Lat.'] >= prm.latMin]

        dftr2 = dftr2[dftr2['Lon.-1'] <= prm.lonMax]
        dftr2 = dftr2[dftr2['Lon.-1'] >= prm.lonMin]
        dftr2 = dftr2[dftr2['Lat.-1'] <= prm.latMax]
        dftr2 = dftr2[dftr2['Lat.-1'] >= prm.latMin]

        dftr2 = dftr2[dftr2['TimeDelta'] < 60.0]
        dftr2 = dftr2[dftr2['TimeDelta'] != np.nan]
        dftr2 = dftr2[dftr2['Distance'] > 0.0]
        dftr2['Speed'] = dftr2['Distance']/dftr2['TimeDelta'] * 3600.
        dftr2 = dftr2[dftr2['Speed'] < 100.0]

        dftr2['xInd'] = dftr2['Lon.'].apply(lambda x : utils.toInd(x, prm.lonMin, prm.lonTick))
        dftr2['yInd'] = dftr2['Lat.'].apply(lambda x : utils.toInd(x, prm.latMin, prm.latTick))

        transDic = {}

        total_len = len(dftr2.index)

        for i in range(total_len-1):
            
            row = dftr2.iloc[i]
            tup = (row['xInd'], row['yInd'])
            
            nextRow = dftr2.iloc[i+1]
            tupNext = (nextRow['xInd'], nextRow['yInd'])
            
            absdiff = abs(tup[0]-tupNext[0]) + abs(tup[1]-tupNext[1])
            
            if absdiff < prm.step/8 and absdiff > 0:
            
                try:
                    s = transDic[tup]
                    s.add(tupNext)
                    transDic[tup] = s
                except:

                    s = set()
                    s.add(tupNext)
                    transDic[tup] = s
            
        G = nx.Graph()
        G.add_nodes_from([key for key in transDic])

        for key in transDic:
            
            ss = transDic[key]
            
            for s in ss:
                G.add_edge(key, s)

        # g_con = list(nx.connected_components(G))

        self.track_df = dftr2
        self.network = G

