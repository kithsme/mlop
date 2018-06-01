import utils
import params as param

import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as td
import math
import matplotlib.pyplot as plt
import numpy as np
import Pair

class OA:

    def __init__(self):
        prm = param.params()

        df = pd.read_csv('ORDER_DATA_2.csv', header=None)
        df['Date']=df[0].apply(lambda x: utils.parseDate(x.split(' ')[0]))
        df['UpTS'] = df.apply(lambda x: utils.putTimeinDate(x['Date'], utils.parseDate(x[4])), axis=1)
        df['CatchTS'] = df.apply(lambda x: utils.putTimeinDate(x['Date'], utils.parseDate(x[8])), axis=1)
        df['PikupTS'] = df.apply(lambda x: utils.putTimeinDate(x['Date'], utils.parseDate(x[9])), axis=1)
        df['DeliverTS'] = df.apply(lambda x: utils.putTimeinDate(x['Date'], utils.parseDate(x[10])), axis=1)
        df = df.drop([0,4,8,9,10], axis=1)
        df.columns = ['OrderID','pLon', 'pLat', 'dLon','dLat','DelivererID', 'Date','UpTS','CatchTS','PickUpTS', 'DeliverTS']
        df['CatchDelay'] = (df['CatchTS'] - df['UpTS']).apply(lambda x: x.seconds)
        df['PickupTravelTime'] = (df['PickUpTS'] - df['CatchTS']).apply(lambda x: x.seconds)
        df['DeliverTravelTime'] = (df['DeliverTS'] - df['PickUpTS']).apply(lambda x: x.seconds)
        df['TotalTravelTime'] = (df['DeliverTS'] - df['CatchTS']).apply(lambda x: x.seconds)
        df['p-dDistance'] = df.apply(lambda x: utils.lonlatDist(x['pLon'],x['pLat'],x['dLon'],x['dLat']), axis=1)

        df['TimeSlot1'] = df['UpTS'].apply(lambda x: utils.getTimeSlot(x, prm.MINS))
        df['TimeSlot2'] = df['CatchTS'].apply(lambda x: utils.getTimeSlot(x, prm.MINS))
        df['TimeSlot3'] = df['PickUpTS'].apply(lambda x: utils.getTimeSlot(x, prm.MINS))
        df['TimeSlot4'] = df['DeliverTS'].apply(lambda x: utils.getTimeSlot(x, prm.MINS))
        df = df[df['PickupTravelTime'] > 59]
        df = df[df['DeliverTravelTime'] > 59]
        
        df['pxInd'] = df['pLon'].apply(lambda x: utils.toInd(x, prm.lonMin, prm.lonTick))
        df['pyInd'] = df['pLat'].apply(lambda x: utils.toInd(x, prm.latMin, prm.latTick))
        df['dxInd'] = df['dLon'].apply(lambda x: utils.toInd(x, prm.lonMin, prm.lonTick))
        df['dyInd'] = df['dLat'].apply(lambda x: utils.toInd(x, prm.latMin, prm.latTick))

        self.order_df = df


    def getRiderWorkDays(self):
        # 이 부분은 배달원의 업무 현황으로 사용될 수 있다
        rdic = {}
        for r in self.order_df['DelivererID'].unique():
            dftm = self.order_df[self.order_df['DelivererID']==r]
            rdic[r] = dftm['Date'].unique()

        fig = plt.figure(figsize=(20,15))
        ax = fig.add_subplot(111)
        ax.grid(which='minor', axis='y')

        rlabel = []
        rloop = 0
        for r in rdic:
            ax.scatter(rdic[r], [rloop for i in range(len(rdic[r]))], label=r, marker='s', c='gray')
            rloop += 1
            rlabel.append(r)

        ax.set_yticks([i for i in range(len(rdic))],minor=True)
        ax.set_yticklabels(rlabel, minor=True)
        ax.set_xlim(utils.parseDate('2015-06-15 00:00'),utils.parseDate('2015-11-02 00:00'))
        ax.set_ylim(-1, len(rdic))
        ax.set_xlabel('Date')
        ax.set_ylabel('Deliverer ID')

        return ax

    def getOrderTimeReport(self):
        # 이 부분은 대기시간/픽업시간/배달시간 등으로 활용 가능함
        prm = param.params()
        timeBoxPlot = [[]]
        ln = 37
        labels= ['']+[str(td(seconds=37800+ 60*prm.MINS*i)) for i in range(ln)]
        fig = plt.figure(figsize=(15,10))

        for i in range(ln):
            
            df_tmp = self.order_df[self.order_df['TimeSlot2'] == i]
            
            timeBoxPlot.append(df_tmp['CatchDelay'].values/60.0)
            
        ax1 = fig.add_subplot(111)
        x = [0]+[i for i in range(0,ln)]
        height = [0]+[len(timeBoxPlot[i]) for i in range(0, ln)]
        ax1.bar(x , height, alpha=0.7)

        ax2 = ax1.twinx()

        ax2.boxplot(timeBoxPlot, showfliers=False)

        ax1.set_xticklabels(labels, rotation=45)
        ax1.set_xlabel('Time slot')
        ax1.set_ylabel('Number of Service Requests (EA)')
        ax2.set_ylabel('Minutes')


        return ax1, ax2

    