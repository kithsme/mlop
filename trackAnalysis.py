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
        dftr2['x-1Ind'] = dftr2['Lon.-1'].apply(lambda x : utils.toInd(x, prm.lonMin, prm.lonTick))
        dftr2['y-1Ind'] = dftr2['Lat.-1'].apply(lambda x : utils.toInd(x, prm.latMin, prm.latTick))

        dftr2['IndDiff'] = dftr2.apply(lambda x: abs(x['xInd']-x['x-1Ind']) + abs(x['yInd']-x['y-1Ind']), axis=1)

        dftr3 = dftr2[dftr2['IndDiff'] > 0]

        dftr3 = dftr3[dftr3['IndDiff'] <= prm.step/16]

        dftr4 = dftr3.groupby(['x-1Ind','y-1Ind','xInd','yInd','IndDiff']).mean()

        dftr4 = dftr4['Speed']

        dftr4 = dftr4.reset_index()

        diTransList = []

        for i, row in dftr4.iterrows():
            
            fromNode = (int(row['x-1Ind']), int(row['y-1Ind']))
            toNode = (int(row['xInd']), int(row['yInd']))
            dist = row['IndDiff']/row['Speed']
            diTransList.append((fromNode, toNode, dist))
            
        DG = nx.DiGraph()
        DG.add_weighted_edges_from(diTransList)

        self.track_df = dftr2
        self.network = DG



