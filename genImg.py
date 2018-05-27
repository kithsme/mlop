import utils
import params
import Pair
import orderAnalysis
import trackAnalysis
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

class Background:

    def __init__(self):
        prm = params.params()
        # order_df = orderAnalysis.OA().order_df
        track_df = trackAnalysis.TA().track_df
        track_graph = trackAnalysis.TA().network

        fig = plt.figure(figsize=(15,15),facecolor='black')
        ax = fig.add_subplot(111)
        ax.set_facecolor("k")
        ax.set_ylim((prm.latMin, prm.latMax))
        ax.set_xlim((prm.lonMin, prm.lonMax))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.scatter(track_df['Lon.'], track_df['Lat.'], color='grey', s=10, alpha=1)

        fig.subplots_adjust(bottom = 0)
        fig.subplots_adjust(top = 1)
        fig.subplots_adjust(right = 1)
        fig.subplots_adjust(left = 0)

        self.background = fig
        self.network = track_graph
        
        track_df_mean = track_df.groupby(['xInd','yInd']).mean()
        self.map_df = track_df_mean


class PairImage:

    def __init__(self, line, bg):
        prm = params.params()
        G = bg.network
        p = Pair.Pair(line)
        c = p.refinedCoords
        idx = p.Id
        color = p.colors

        p1 = c[0:2]
        p2 = c[2:4]
        p3 = c[4:6]
        p4 = c[6:]

        indp1 = (utils.toInd(p1[0],prm.lonMin, prm.lonTick), utils.toInd(p1[1], prm.latMin, prm.latTick))
        indd1 = (utils.toInd(p2[0],prm.lonMin, prm.lonTick), utils.toInd(p2[1], prm.latMin, prm.latTick))
        indp2 = (utils.toInd(p3[0],prm.lonMin, prm.lonTick), utils.toInd(p3[1], prm.latMin, prm.latTick))
        indd2 = (utils.toInd(p4[0],prm.lonMin, prm.lonTick), utils.toInd(p4[1], prm.latMin, prm.latTick))

        target = nx.connected_components(G)
        target = list(target)
        target = target[0]
        
        if indp1 not in target:
            indp1 = utils.approximate(indp1, target)
                
        if indd1 not in target:
            indd1 = utils.approximate(indd1, target)
        
        if indp2 not in target:
            indp2 = utils.approximate(indp2, target)
            
        if indd2 not in target:
            indd2 = utils.approximate(indd2, target)
            
        ashortest = nx.shortest_path(G, indp1, indd1)
        ashortest1 = nx.shortest_path(G, indd1, indp2)
        ashortest2 = nx.shortest_path(G, indp2, indd2)

        zz = [utils.mapItOn(z, bg.map_df) for z in ashortest]
        zz = np.asarray(zz)
        zz1 = [utils.mapItOn(z, bg.map_df) for z in ashortest1]
        zz1 = np.asarray(zz1)
        zz2 = [utils.mapItOn(z, bg.map_df) for z in ashortest2]
        zz2 = np.asarray(zz2)
        
        zp1 = bg.background.axes[0].plot(zz[:,0], zz[:,1], lw=20, c=color[0], alpha=1)
        zp2 = bg.background.axes[0].plot(zz1[:,0], zz1[:,1], lw=20, c=color[1],alpha=1)
        zp3 = bg.background.axes[0].plot(zz2[:,0], zz2[:,1], lw=20, c=color[2],alpha=1)
        
        bg.background.savefig('out.png', bbox_inches='tight', pad_inches=0)
        img = Image.open('out.png')
        img = img.resize((prm.step, prm.step), Image.ANTIALIAS) # resizes image in-place
        
        bg.background.axes[0].lines.pop(0)
        bg.background.axes[0].lines.pop(0)
        bg.background.axes[0].lines.pop(0)

        self.x = np.asarray(img)[:,:,:-1]
        self.y = 1 - p.type[0]

def prepareLearningXYsFromFiles(positive_dir, negative_dir):
    """ get xy instances from png file already created """
    xys = []

    for file in tqdm(os.listdir(positive_dir),desc='Positive Instances '):
        if file.endswith(".png"):
            img = Image.open(file)
            x = np.asarray(img)[:,:,:-1]
            y = 1
            xys.append((x,y))
    
    for file in tqdm(os.listdir(negative_dir),desc='Negative Instances '):
        if file.endswith(".png"):
            img = Image.open(file)
            x = np.asarray(img)[:,:,:-1]
            y = 0
            xys.append((x,y))

    return xys

def prepareLearningXYs():
    
    def getData(df):
        strong, weak = [], []
        for der in df['DelivererID'].unique():
    
            df_tmp = df[df['DelivererID']==der]
            df_tmp = df_tmp.reset_index()
            
            for i, row in df_tmp.iterrows():
                
                df_tmp2 = df_tmp[ (df_tmp['UpTS'] > row['UpTS']) & (df_tmp['UpTS'] < row['DeliverTS']) ]
                
                try:
                    for j, row2 in df_tmp2.iterrows():
                        
                        if utils.isStrong(row, row2):
                            strong.append((row, row2))
                        
                        if utils.isWeak(row, row2):
                            weak.append((row, row2))
                
                except:
                    pass
        
        return strong, weak

    print('Generating Image for Learning Started!')

    order_df = orderAnalysis.OA().order_df

    strong, weak = getData(order_df)
    strong_good, strong_bad = [], []
    for s in strong:
        if utils.onTime(s[0]) and utils.onTime(s[1]):
            strong_good.append(s)
        else:
            strong_bad.append(s)
            
    # weak_good, weak_bad = [], []
    # for w in weak:
    #     if utils.onTime(w[0]) and utils.onTime(w[1]):
    #         weak_good.append(w)
    #     else:
    #         weak_bad.append(w)        
        
    gg = Background()
    xys = []

    for s in tqdm(strong_good,desc='Positive Instances '):
        pair = PairImage(s, gg)
        xys.append((pair.x, pair.y))
    
    for s in tqdm(strong_bad,desc='Negative Instances '):
        pair = PairImage(s, gg)
        xys.append((pair.x, pair.y))
    print('Generating Image for Learning Completed!')
    return xys
