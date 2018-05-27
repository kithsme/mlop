import numpy as np
import pandas as pd
import utils


class Pair:

    def getType(self,line):
        # ppdd or pdpd
        
#         catch = [line[0]['CatchTS'], line[1]['CatchTS']]
        pick = [line[0]['PickUpTS'], line[1]['PickUpTS']]
        deliver = [line[0]['DeliverTS'], line[1]['DeliverTS']]
        
#         maxCatch = np.max(catch)
        minPick = np.min(pick)
        maxPick = np.max(pick)
        minDeliver = np.min(deliver)
        
        argminPick = np.argmin(pick)
        argminDeliver = np.argmin(deliver)
        
        wsType = 0
        pdType = -1
        if (maxPick > minDeliver) : 
            wsType = 1
                        
        if (argminPick == 0 and argminDeliver == 0):
            pdType = 0
        elif (argminPick == 1 and argminDeliver == 0):
            pdType = 1
        elif (argminPick == 0 and argminDeliver == 1):
            pdType = 2
        else:
            pdType = 3
        
        return (wsType, pdType)
    
    def getRefinedCoords(self):
        ws,pd = self.type
        c = self.coords
        c1,c2,c3,c4 = c[0:2], c[2:4], c[4:6], c[6:]
        
        if ws == 1:
            if pd == 0:
                return c1 + c2 + c3 + c4
            else:
                return c3 + c4 + c1 + c2
        else:
            
            if pd == 0:
                return c1 + c3 + c2 + c4
            
            elif pd == 1:
                return c3 + c1 + c2 + c4
            
            elif pd == 2:
                return c1 + c3 + c4 + c2
            
            else :
                return c3 + c1 + c4 + c2
    
    def getOverheadDist(self):
        
        rc = self.refinedCoords
        c = self.coords
        
        origin = 0.0
        origin += utils.lonlatDist(c[0], c[1], c[2], c[3])
        origin += utils.lonlatDist(c[4], c[5], c[6], c[7])
        
        dist = 0.0
        dist += utils.lonlatDist(rc[0], rc[1], rc[2], rc[3])
        dist += utils.lonlatDist(rc[2], rc[3], rc[4], rc[5])
        dist += utils.lonlatDist(rc[4], rc[5], rc[6], rc[7])
        
        return (dist, origin)
    
    def getColorCode(self):
        
        ws,pd = self.type
        
        if ws == 1:
            if pd == 0:
                return ['y','cyan','r']
            else:
                return ['r','b','y']
        else:
            
            if pd == 0:
                return ['y','r','cyan']
            
            elif pd == 1:
                return ['r','y','cyan']
            
            elif pd == 2:
                return ['y','r','b']
            
            else :
                return ['r','y','b']
        
    def __init__(self, line):
        self.Id = line[0]['OrderID'] +'_'+ line[1]['OrderID']
        self.type = self.getType(line)
        self.coords = list(line[0][['pLon','pLat','dLon','dLat']]) + list(line[1][['pLon','pLat','dLon','dLat']])
        self.refinedCoords = self.getRefinedCoords()
        self.catchGap = (max(line[0]['CatchTS'], line[1]['CatchTS']) - min(line[0]['CatchTS'], line[1]['CatchTS'])).seconds
        self.overhead = self.getOverheadDist()
        self.colors = self.getColorCode()
        