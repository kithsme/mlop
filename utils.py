import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as td
import math
import matplotlib.pyplot as plt
import numpy as np


def putTimeinDate(dtt, tt):
    a = dtt
    h = tt.hour
    mi = tt.minute
    s = tt.second
    a = a.replace(hour=h, minute=mi, second=s)
    
    return a

def parseDate(dtstr):
    if type(dtstr) is type(dt):
        return dtstr
    
    for fmt in ('%Y-%m-%d %H:%M', '%Y.%m.%d %H:%M', '%H:%M:%S','%H%M%S', '%Y.%m.%d','%Y%m%d','%H%M%S'):
        try:
            return dt.strptime(dtstr, fmt)
        except ValueError:
            pass
    raise ValueError('No valid date format')
    
    
def lonlatDist(lon1, lat1, lon2, lat2):
    """ return km """
    R = 6371
    radLon1 = math.radians(lon1)
    radLat1 = math.radians(lat1)
    radLon2 = math.radians(lon2)
    radLat2 = math.radians(lat2)
    
    x = (radLon2-radLon1) * math.cos(0.5*(radLat2+radLat1))
    y = radLat2 - radLat1
    
    return R*math.sqrt(x*x + y*y)

def getTime(x):
    return 3600*x.hour + 60*x.minute + x.second

def getTimeSlot(x, mins=20):
    """ start from AM 10:30 (=37800 sec) """
    stime = 37800 
    # etime = 81000 # 22:30
    t = getTime(x)
    return (t - stime) // (mins*60)

def getTrackTimeDelta(x):
    
    if x['RiderID'] == x['RiderID-1'] and 0 < (x['TS']-x['TS-1']).seconds and x['TS'].date() == x['TS-1'].date():
        return (x['TS'] - x['TS-1']).seconds
    else:
        return np.nan
        
def toInd(x, mn, tick):
    
    return int((x-mn)/tick)

def mapItOn(key, base):
    
    row = base.loc[key]
    
    return [row['Lon.'], row['Lat.']]

def approximate(p, connected_comps):
    
    min_c = None
    min_val = 9999
    
    for c in connected_comps:
        
        a = abs(c[0]-p[0])
        b = abs(c[1]-p[1])
        
        if a+b < min_val:
            min_c = c
            min_val = a+b
    
    return min_c
        

def getTrackDist(x):
    """ getTrackTimeDelta should be called before this """
    if 0.0 < x['TimeDelta'] :
        return lonlatDist(x['Lon.'],x['Lat.'],x['Lon.-1'],x['Lat.-1'])
    else:
        return np.nan

def isStrong(r1, r2):
    
    maxCTS = max([r1['CatchTS'], r2['CatchTS']])
    minPTS = min([r1['PickUpTS'], r2['PickUpTS']])
    maxPTS = max([r1['PickUpTS'], r2['PickUpTS']])
    minDTS = min([r1['DeliverTS'], r2['DeliverTS']])
    
    if maxCTS < minPTS and maxPTS < minDTS:
        return True
    else:
        return False


def isWeak(r1, r2):
    
    maxCTS = max([r1['CatchTS'], r2['CatchTS']])
    minPTS = min([r1['PickUpTS'], r2['PickUpTS']])
    maxPTS = max([r1['PickUpTS'], r2['PickUpTS']])
    minDTS = min([r1['DeliverTS'], r2['DeliverTS']])
    
    if maxCTS > minPTS and maxPTS > minDTS:
        return True
    else:
        return False

def onTime(r):
    
    if (r['DeliverTS'] - r['PickUpTS']).seconds > 30. * 60.:
        return False
    else:
        return True
    