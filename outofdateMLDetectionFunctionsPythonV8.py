#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division
import numpy as np
from numpy import loadtxt
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d
import scipy
from scipy.fftpack import fft
from scipy.signal import find_peaks
from pylab import plot, ylim, xlim, show, xlabel, ylabel, grid
from numpy import linspace, ones, convolve
from scipy.signal import argrelextrema
import sys
from numpy import NaN, Inf, arange, isscalar, array
import math
import pandas as pd
from numpy.random import randint
from matplotlib.widgets import Slider
from scipy.interpolate import splrep, splev
from sklearn import preprocessing
from sklearn import cluster
import pickle

from tqdm.notebook import tqdm as counter


# In[ ]:


from tqdm import tqdm


def getdata(datafile): #function to load in the datafile when file is in .txt format
    data = np.loadtxt(datafile)
    return data

def timesignal(data): #split the data into a time stream and a signal stream
    time = data[:]#,0]
    signal = data[:]#,1]
    return time, signal

def movingaverage(interval, window_size): #function taked from stackoverflow to use a moving average
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def plotsmooth(datafile, time, signal, y_av): #function to plot the rolling average signal versus the raw signal
    plt.title("Plot of Smooth {}".format(datafile)) 
    plt.xlabel("Time") 
    plt.ylabel("Signal")
    plt.plot(time, signal, linewidth=0.1)
    plt.plot(time, y_av, "r", linewidth=3)
    plt.rcParams['figure.figsize'] = [100, 50]
    plt.show()
    return

def peakdet(v, delta, x): #function taken from stackoverflow to mark the min and max peaks of a signal
    '''
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    Returns two arrays
    function [maxtab, mintab]=peakdet(v, delta, x) 
    '''
    maxtab = []
    mintab = []
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    lookformax = True

    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True
    return array(maxtab), array(mintab)

def get_signal_points(y_av_ultra_smooth, y_av, maxtab, mintab, time): #function that trims down min and max points to just major min and max points.
    deviation2 = y_av - y_av_ultra_smooth                             #function switches to plotting the difference from the average then converts back to raw signal line
    series=deviation2

    x = np.linspace(0, 10, len(series))
    maxtab, mintab = peakdet(series, .3, x)   # it is very important to give the correct x to `peakdet`, y_av2 changed from series

    thresh = 50


    maxtabx = maxtab[:,0]
    maxtaby = maxtab[:,1]

    redmaxtabx = np.delete(maxtabx, np.where(abs(maxtaby) <= thresh))
    redmaxtaby = np.delete(maxtaby, np.where(abs(maxtaby) <= thresh))

    redmaxtabx = np.delete(redmaxtabx, np.where(redmaxtaby <= 0))
    redmaxtaby = np.delete(redmaxtaby, np.where(redmaxtaby <= 0))


    mintabx = mintab[:,0]
    mintaby = mintab[:,1]

    redmintabx = np.delete(mintabx, np.where(abs(mintaby) <= thresh))
    redmintaby = np.delete(mintaby, np.where(abs(mintaby) <= thresh))

    redmintabx = np.delete(redmintabx, np.where(redmintaby >= 0))
    redmintaby = np.delete(redmintaby, np.where(redmintaby >= 0))
    
    Reposredmaxtaby = redmaxtaby
    Reposredmaxtabx = redmaxtabx

    Reposredmaxtabx = (Reposredmaxtabx/10)*len(time)
    Reposredmaxtabx = np.round_(Reposredmaxtabx)
    Reposredmaxtabx = Reposredmaxtabx.astype(np.int)

    for i in range(len(Reposredmaxtaby)):
        Reposredmaxtaby[i] = y_av[Reposredmaxtabx[i]]
        
    
    Reposredmintaby = redmintaby
    Reposredmintabx = redmintabx

    Reposredmintabx = (Reposredmintabx/10)*len(time)
    Reposredmintabx = np.round_(Reposredmintabx)
    Reposredmintabx = Reposredmintabx.astype(np.int)

    for i in range(len(Reposredmintaby)):
        Reposredmintaby[i] = y_av[Reposredmintabx[i]]
        
    
    return redmaxtabx, Reposredmaxtaby, redmintabx, Reposredmintaby, x

def plot_signal_point(redmaxtabx, Reposredmaxtaby, redmintabx, redmintaby, x, y_av): #function plots the major min and max points from prev function
    plt.plot(x, y_av, '-', color='black')   # use the same x for plotting, y_av2 changed from series
    plt.scatter(redmaxtabx, Reposredmaxtaby, color='blue') # the x-coordinates used in maxtab need to be the same as those in plot
    plt.scatter(redmintabx, redmintaby, color='red')
    plt.rcParams['figure.figsize'] = [100, 50]
    plt.show()
    
def plot_signal_point_plus(redmaxtabx, Reposredmaxtaby, redmintabx, redmintaby, x, y_av, y_av_ultra_smooth): #function plots the major min and max points from prev function with ultra smoothed as well
    plt.plot(x, y_av, '-', color='black')   # use the same x for plotting, y_av2 changed from series
    plt.plot(x, y_av_ultra_smooth, '-', color='green')
    plt.scatter(redmaxtabx, Reposredmaxtaby, color='blue') # the x-coordinates used in maxtab need to be the same as those in plot
    plt.scatter(redmintabx, redmintaby, color='red')
    plt.rcParams['figure.figsize'] = [100, 50]
    plt.show()

def pipeline(data, v_smooth_num, av_num, delta, start_crop, end_crop): #function with datafile as only input and runs full pipeline to get the final min and max points. Can then rn the plotting function after to see plot
    #data = getdata(datafile)
    
    data_crop = data[start_crop:end_crop]
    
    time, signal = timesignal(data_crop)
    
    y_av_ultra_smooth = movingaverage(signal, v_smooth_num)
    
    y_av = movingaverage(signal, av_num)
    
    #deviation = y_av - y_av_ultra_smooth
    #series=deviation
    
    series = y_av
    x = np.linspace(0, 10, len(series))

    maxtab, mintab = peakdet(series, delta, x) #Delta is min height to be a peak
    
    #redmaxtabx, Reposredmaxtaby, redmintabx, Reposredmintaby, x = get_signal_points(y_av_ultra_smooth, y_av, maxtab, mintab, time)
    
    if maxtab.ndim == 1:
        #maxtab = np.append(maxtab,0)
        maxtab = np.resize(maxtab,(1,2))
        
    if mintab.ndim == 1:
        #mintab = np.append(mintab,0)
        mintab = np.resize(mintab,(1,2))    
    
    maxtabx = maxtab[:,0]
    maxtaby = maxtab[:,1]
    
    mintabx = mintab[:,0]
    mintaby = mintab[:,1]
    
    return maxtabx, maxtaby, mintabx, mintaby, x, y_av, y_av_ultra_smooth

def DetectionScan(curvedata, realdata, COIlen, LenBeforeEvent):
    
    data1 = getdata(curvedata)
    data2 = realdata #getdata(realdata)
    
    combdata = data2
    
    combtime, combsignal = timesignal(combdata)
    curvetime, curvesignal = timesignal(data1)
    
    y_av_curve = movingaverage(curvesignal,15000)
    y_av = movingaverage(combsignal,15000)
    y_av_less_smooth = movingaverage(combsignal,2000)
    curvelen = len(curvesignal)
    
    crosscorr = scipy.signal.correlate(y_av, y_av_curve)
    peaks, _ = find_peaks(crosscorr, distance=500000, height=1e10)
    maxpoints = peaks

    
    TESTnumber = 0
    
    for Z in range(len(maxpoints)): #need to alter so range goes up to like 500 but no loop error
    #if np.int(maxpoints[X]) > 0:
        #curvesection'X' = y_av[np.int(maxpoints[X])-curvelen:np.int(maxpoints[X])]
        globals()['curvesection%s' % Z] = combdata[np.int(maxpoints[Z])-(curvelen-2)-LenBeforeEvent:np.int(maxpoints[Z])-(curvelen-2)+COIlen] #need to alter so size can change
        TESTnumber = TESTnumber + 1
    
    if TESTnumber == 0:
        print('No Curves of Interest Found. Try altering input parameters to detect curves.')
        
    else:
        numsigs = Z+1
    
        #ADD LOOP SO CAN HAVE ANY NUMBER OF SIGNALS HERE
        extractedsignals = np.empty([COIlen+LenBeforeEvent,2,0]) #changed from 899999. need to alter though
        for Y in range(numsigs):
            extractedsignals = np.dstack((extractedsignals,globals()['curvesection%s' % Y]))
    
    
        print('Number of Curves of Interest is: %s' % str(Z+1))
        print('Curves of Interest Highlighted:')
    
        plt.plot(combdata)
        for J in range(len(maxpoints)):  #again loop so number of loops is not issue
            plt.axvspan(np.int(maxpoints[J])-(curvelen-2)-LenBeforeEvent, np.int(maxpoints[J])-(curvelen-2)+COIlen, color='green', alpha=0.5)
    
        plt.plot(y_av)
        #for J in range(len(maxpoints)):  #again loop so number of loops is not issue
        #plt.axvspan(np.int(maxpoints[J])-(curvelen-2), np.int(maxpoints[J]), color='red', alpha=0.5)
        
        
        return extractedsignals

def DetectionScanSmallPeak(smallpeakdata, realdata, COIlen, peakheight):
    
    #data1 = getdata(smallpeakdata)
    #data2 = getdata(realdata)
    
    combdata = realdata
    
    combtime, combsignal = timesignal(combdata)
    curvetime, curvesignal = timesignal(smallpeakdata)
    
    y_av_curve = movingaverage(curvesignal,100)
    y_av = movingaverage(combsignal,100)
    y_av_less_smooth = movingaverage(combsignal,2000)
    curvelen = len(curvesignal)
    
    crosscorr = scipy.signal.correlate(y_av, y_av_curve)
    peaks, _ = find_peaks(crosscorr, distance=20000, height=peakheight) #5e7
    maxpoints = peaks

    TESTnum = 0
    
    for Z in range(len(maxpoints)): #need to alter so range goes up to like 500 but no loop error
    #if np.int(maxpoints[X]) > 0:
        #curvesection'X' = y_av[np.int(maxpoints[X])-curvelen:np.int(maxpoints[X])]
        globals()['smallcurvesection%s' % Z] = combdata[np.int(maxpoints[Z])-(curvelen-2):np.int(maxpoints[Z])-(curvelen-2)+COIlen] #need to alter so size can change
        TESTnum = TESTnum +1
        
        
    if TESTnum == 0:
        print('No Smaller Peaks Found')
        
    else:
        numsigs = Z+1
    
        #ADD LOOP SO CAN HAVE ANY NUMBER OF SIGNALS HERE
        smallcurveextractedsignals = np.empty([COIlen,2,0]) #changed from 899999. need to alter though
        for Y in range(numsigs):
            smallcurveextractedsignals = np.dstack((smallcurveextractedsignals,globals()['smallcurvesection%s' % Y]))
    
    
        print('Number of Curves of Interest is: %s' % str(Z+1))
        print('Curves of Interest Highlighted:')
    
        plt.plot(combdata)
        for J in range(len(maxpoints)):  #again loop so number of loops is not issue
            plt.axvspan(np.int(maxpoints[J])-(curvelen-2), np.int(maxpoints[J])-(curvelen-2)+COIlen, color='red', alpha=0.5)
    
        plt.plot(y_av)
        #for J in range(len(maxpoints)):  #again loop so number of loops is not issue
            #plt.axvspan(np.int(maxpoints[J])-(curvelen-2), np.int(maxpoints[J]), color='red', alpha=0.5)
        
        
        return smallcurveextractedsignals
        

def GetStartPoint(extractedsignals,Thresh):

    GlobMinIndex = np.argmin(extractedsignals[:,1,0])



    BeforeMinArray = extractedsignals[:GlobMinIndex,:,0]
    AfterMinArray = extractedsignals[GlobMinIndex:,:,0] #GlobMinIndex+1: if not wanting to include the global min point


    BeforeMinArrayTime, BeforeMinArraySignal = timesignal(BeforeMinArray)

    BeforeMinArray_HighS = movingaverage(BeforeMinArraySignal, 5000)
    
    BeforeMinArray_LowS = movingaverage(BeforeMinArraySignal, 50)


    AfterMinArrayTime, AfterMinArraySignal = timesignal(AfterMinArray)

    AfterMinArray_HighS = movingaverage(AfterMinArraySignal, 5000)
    
    AfterMinArray_LowS = movingaverage(AfterMinArraySignal, 50)
    
    BeforeMinArray_LowS_ForDensity = movingaverage(BeforeMinArraySignal[0:len(BeforeMinArraySignal)], 500)

    BeforeMinGrad = np.gradient(BeforeMinArray_LowS_ForDensity)

    BeforeMinGrad2 = np.gradient(BeforeMinGrad)

    R = np.linspace(0, len(BeforeMinArray_LowS_ForDensity), len(BeforeMinArray_LowS_ForDensity))

    BeforeMinGradmaxtab, BeforeMinGradmintab = peakdet(BeforeMinArray_LowS_ForDensity, 40, R) #Delta is min height to be a peak
     
    BeforeMinGradmaxtabx = BeforeMinGradmaxtab[:,0]
    BeforeMinGradmaxtaby = BeforeMinGradmaxtab[:,1]*0+100
    
    BeforeMinGradmintabx = BeforeMinGradmintab[:,0]
    BeforeMinGradmintaby = BeforeMinGradmintab[:,1]*0+100

    
    BeforeMinArrayDensityArr = BeforeMinArray_LowS_ForDensity*0

    for Q in range(len(BeforeMinGradmaxtabx.astype(int))):
        BeforeMinArrayDensityArr[BeforeMinGradmaxtabx.astype(int)[Q]] = 50000

    for Q in range(len(BeforeMinGradmintabx.astype(int))):
        BeforeMinArrayDensityArr[BeforeMinGradmintabx.astype(int)[Q]] = 50000
    
    BeforeMinArrayDensity = movingaverage(BeforeMinArrayDensityArr, len(BeforeMinArrayDensityArr.astype(int))/100)
    
    BeforeMinArrayDensityX = np.arange(len(BeforeMinArrayDensity))

    poly = np.polyfit(x=BeforeMinArrayDensityX,y=BeforeMinArrayDensity,deg=30)

    yfit = np.polyval(poly,BeforeMinArrayDensityX)
    
    BeforeMinIndMax = np.argmax(yfit)

#find first place yfit is equal to half of max

    BeforeMinDenistyCrop = yfit[:BeforeMinIndMax]

#for F in range(len(BeforeMinDenistyCrop)):
#    if BeforeMinDenistyCrop[-F] <= (yfit[BeforeMinIndMax])/2:
#        StartPointTEST = F


#When this made into function can have division as input so can have a way to alter starts

    StartPointTEST = np.where(BeforeMinDenistyCrop <= (yfit[BeforeMinIndMax])/Thresh,BeforeMinDenistyCrop+20,yfit[BeforeMinIndMax]+30)

    PossibleStarts = np.where(StartPointTEST != yfit[BeforeMinIndMax]+30)
    StartPoint = np.max(PossibleStarts)

#Maybe use find peaks on before min data and then look at density of peaks to get start point

#Find global minimum of the selected area
#Have ultra long smooth and less smooth
#Work back and forward from global minimum
#going back find point that less smooth is within X of ultra smooth for N frames
#going forward find point that less smooth is within X of 0 line for N frames

    return StartPoint
    
    
def GetEndPoint (extractedsignals):


    GlobMinIndex = np.argmin(extractedsignals[:,1,0])



    BeforeMinArray = extractedsignals[:GlobMinIndex,:,0]
    AfterMinArray = extractedsignals[GlobMinIndex:,:,0] #GlobMinIndex+1: if not wanting to include the global min point


    BeforeMinArrayTime, BeforeMinArraySignal = timesignal(BeforeMinArray)

    BeforeMinArray_HighS = movingaverage(BeforeMinArraySignal, 5000)
    
    BeforeMinArray_LowS = movingaverage(BeforeMinArraySignal, 50)


    AfterMinArrayTime, AfterMinArraySignal = timesignal(AfterMinArray)

    AfterMinArray_HighS = movingaverage(AfterMinArraySignal, 5000)
    
    AfterMinArray_LowS = movingaverage(AfterMinArraySignal, 50)
    
    AfterMinArrayGrad = np.gradient(AfterMinArray_HighS)

    EndPointArr_test = AfterMinArray_HighS*0

    EndPointArr_test = np.where((np.absolute(AfterMinArray_HighS)>=40),EndPointArr_test,1500)

    Z = 1

    for N in range(len(EndPointArr_test)):
        if EndPointArr_test[N] == 1500:
            if EndPointArr_test[N-1]==0:
                EndPointArr_test[N] = EndPointArr_test[N] + Z
                Z = Z+1

    EndPointArr = np.where((EndPointArr_test!=1503),AfterMinArray_HighS+200,1600)

#Select the 3rd point at which EndPoint_test is True?

    EndPointTwo = np.argmax(EndPointArr)

#Have end point be middle between global min between global max and when 3rd EndPoint_test is true

    AfterMaxInd = np.argmax(AfterMinArray_HighS)

    SecondCutAfter = AfterMinArray_HighS[AfterMaxInd:EndPointTwo]
    
    if not SecondCutAfter:
        SecondCutAfterMinInd=0+AfterMaxInd
        
    else:

        SecondCutAfterMinInd = np.argmin(SecondCutAfter)+AfterMaxInd

    EndPoint = np.floor((SecondCutAfterMinInd+EndPointTwo)/2) + GlobMinIndex
    
    return EndPoint


def TESTplot_signal_point_plus(redmaxtabx, Reposredmaxtaby, redmintabx, redmintaby, x, y_av, y_av_ultra_smooth): #function plots the major min and max points from prev function with ultra smoothed as well
    plt.plot(x, y_av, '-', color='black')   # use the same x for plotting, y_av2 changed from series
    plt.plot(x, y_av_ultra_smooth, '-', color='green')
    plt.scatter(redmaxtabx, Reposredmaxtaby, color='blue') # the x-coordinates used in maxtab need to be the same as those in plot
    plt.scatter(redmintabx, redmintaby, color='red')
    plt.axvspan(maxtabx[0]-Addedlen, mintabx[len(mintabx)-1], color='red', alpha=0.2)
    plt.rcParams['figure.figsize'] = [100, 50]
    plt.show()        
        

def CreateDataframe(realdata):

    time, signal = timesignal(realdata)

    y_av750 = movingaverage(signal[:,1],750)

    y_av10k = movingaverage(signal[:,1],10000)

    TrainingDFtest = pd.DataFrame(columns=['Yval', 'Yvalgrad','Yval750', 'Yval750grad', 'Yval10k', 'Yval10kgrad'])

    TrainingDFtest['Yval'] = abs(signal[:,1])

    TrainingDFtest['Yvalgrad'] = abs(np.gradient(signal[:,1]))

    TrainingDFtest['Yval750'] = abs(y_av750)

    TrainingDFtest['Yval750grad'] = abs(np.gradient(y_av750))

    TrainingDFtest['Yval10k'] = abs(y_av10k)

    TrainingDFtest['Yval10kgrad'] = abs(np.gradient(y_av10k))

    TrainingDFtest['Yval'] = TrainingDFtest['Yval']/5000
    TrainingDFtest['Yvalgrad'] = TrainingDFtest['Yvalgrad']/np.max(TrainingDFtest['Yvalgrad'])

    TrainingDFtest['Yval750'] = TrainingDFtest['Yval750']/5000
    TrainingDFtest['Yval750grad'] = TrainingDFtest['Yval750grad']/np.max(TrainingDFtest['Yval750grad'])

    TrainingDFtest['Yval10k'] = TrainingDFtest['Yval10k']/5000
    TrainingDFtest['Yval10kgrad'] = TrainingDFtest['Yval10kgrad']/np.max(TrainingDFtest['Yval10kgrad'])

    DFnorm = pd.DataFrame(TrainingDFtest.values)

    DFnorm = DFnorm*1000
    
    return DFnorm, signal


def PredictSignal(DFnorm,signal,gap):

    loaded_model = pickle.load(open('5000MaxValDiv75010k.sav', 'rb')) # LowerAve_NORM_TEST_ML_MODE.sav

    Prediction = loaded_model.predict(DFnorm)

    plt.plot(signal)
    plt.plot(Prediction*1000)
    plt.rcParams['figure.figsize'] = [50, 25]
    plt.show()

    Detection = Prediction*0

    labelar = Prediction*10000

    for i in tqdm(range(len(Prediction))):
        if labelar[i] > 9900:
            Detection[i] = 8000

    signalcount = np.empty(len(signal))
    gapcount = np.empty(len(signal))
    frame = np.arange(len(signal))

    signo = 0
    gapno = 0

    n = 0
    u = 0

    for i in tqdm(range(len(signal))):
    
        if Detection[i] == 0:
        
            signalcount[int(u):int(i)] = signo
        
            gapno = gapno+1
            signo = 0
        
            #signalcount[i] = signo
            gapcount[i] = gapno
        
            u = i

        

        
        #gapcount = np.append(gapcount,gapno)
        #signalcount = np.append(signalcount,0)
        
        #frame = np.append(frame,i)
        
    
        if Detection[i] > 0:
        
            gapcount[int(n):int(i)] = gapno
        
            gapno = 0
            signo = signo+1
        
            signalcount[i] = signo
        #gapcount[int(n):int(i)] = gapno
        
            n = i
        
        
        if i == len(signal)-1:
        
            gapcount[int(n):int(i)] = gapno
        

        
        #gapcount = np.append(gapcount,0)
        #signalcount = np.append(signalcount,signo)
        
        #frame = np.append(frame,i)
    

    for i in tqdm(range(len(signal))):
    
        if gapcount[i] < gap:
            #signalcount[i] = gapcount[i]
            gapcount[i] = 0
            signalcount[i] = 1

    for i in tqdm(range(len(signal))):
    
        if signalcount[i] > 0:
        
            signalcount[i] = 1
        
    
        if gapcount[i] > 0:
        
            gapcount[i] = 1




    signalCURVES = np.empty(len(signal))

    for i in tqdm(range(len(signal))):
    
        signalCURVES[i] = signal[i,1]*signalcount[i]


    ExtractedSignalCountStart = 0
    ExtractedSignalCountEnd = 0

    i=0

    StartVals = []
    EndVals = []

    for x in range(100): #20000 only 50 until loop problem sorted
    
        print(x)
    
        for i in tqdm(range(ExtractedSignalCountEnd+1,len(signalCURVES))):
            if i == ExtractedSignalCountStart: #this is editable
                    x = 50-3
                    break
            if signalcount[i] == 1:
                ExtractedSignalCountStart = i
                StartVals = np.append(StartVals,ExtractedSignalCountStart)
                #print(ExtractedSignalCountStart)
                print('i')
                print(len(signalCURVES)-ExtractedSignalCountStart+1)
                print(len(signalCURVES))
                print(len(signalCURVES)+ExtractedSignalCountStart+1)
                print(ExtractedSignalCountStart)
                print(i)
                break
            
        
        for q in tqdm(range(ExtractedSignalCountStart+1,len(signalCURVES))):
            if q == ExtractedSignalCountEnd: #this is editable
                    x = 50-3
                    #ExtractedSignalCountEnd = q #this is editable
                    #EndVals = np.append(EndVals,ExtractedSignalCountEnd) #this is editable
                    break
            if signalcount[q] == 0:
                ExtractedSignalCountEnd = q
                EndVals = np.append(EndVals,ExtractedSignalCountEnd)
                #print(ExtractedSignalCountEnd) 
                print('q')
                print(len(signalCURVES)-ExtractedSignalCountStart+1)
                print(len(signalCURVES))
                print(len(signalCURVES)+ExtractedSignalCountStart+1)
                print(ExtractedSignalCountStart)
                print(q)    
                break
    
    

    print(StartVals)  
    print(EndVals)
    
    if len(StartVals) - len(EndVals) == 1:
        EndVals = np.append(EndVals,len(signalCURVES))
    
    print(StartVals)  
    print(EndVals)


    return StartVals, EndVals, signalCURVES
    

def PredictSignalInFS(DFnorm,signal,gap):

    loaded_model = pickle.load(open('5000MaxValDiv75010k.sav', 'rb')) # LowerAve_NORM_TEST_ML_MODE.sav

    Prediction = loaded_model.predict(DFnorm)

    #plt.plot(signal)
    #plt.plot(Prediction*1000)
    #plt.rcParams['figure.figsize'] = [50, 25]
    #plt.show()

    Detection = Prediction*0

    labelar = Prediction*10000

    for i in tqdm(range(len(Prediction))):
        if labelar[i] > 9900:
            Detection[i] = 8000

    signalcount = np.empty(len(signal))
    gapcount = np.empty(len(signal))
    frame = np.arange(len(signal))

    signo = 0
    gapno = 0

    n = 0
    u = 0

    for i in tqdm(range(len(signal))):
    
        if Detection[i] == 0:
        
            signalcount[int(u):int(i)] = signo
        
            gapno = gapno+1
            signo = 0
        
            #signalcount[i] = signo
            gapcount[i] = gapno
        
            u = i

        

        
        #gapcount = np.append(gapcount,gapno)
        #signalcount = np.append(signalcount,0)
        
        #frame = np.append(frame,i)
        
    
        if Detection[i] > 0:
        
            gapcount[int(n):int(i)] = gapno
        
            gapno = 0
            signo = signo+1
        
            signalcount[i] = signo
        #gapcount[int(n):int(i)] = gapno
        
            n = i
        
        
        if i == len(signal)-1:
        
            gapcount[int(n):int(i)] = gapno
        

        
        #gapcount = np.append(gapcount,0)
        #signalcount = np.append(signalcount,signo)
        
        #frame = np.append(frame,i)
    

    for i in tqdm(range(len(signal))):
    
        if gapcount[i] < gap:
            #signalcount[i] = gapcount[i]
            gapcount[i] = 0
            signalcount[i] = 1

    for i in tqdm(range(len(signal))):
    
        if signalcount[i] > 0:
        
            signalcount[i] = 1
        
    
        if gapcount[i] > 0:
        
            gapcount[i] = 1




    signalCURVES = np.empty(len(signal))

    for i in tqdm(range(len(signal))):
    
        signalCURVES[i] = signal[i,1]*signalcount[i]


    ExtractedSignalCountStart = 0
    ExtractedSignalCountEnd = 0

    i=0

    StartVals = []
    EndVals = []

    print()
    print()
    print()
    breakval = 0

    for x in tqdm(range(50)): #20000 only 50 until loop problem sorted 1 might work
        print(len(signalCURVES)-ExtractedSignalCountStart+1)
        print(len(signalCURVES)-ExtractedSignalCountStart+1-3)
        print('This one:')
        print(len(signalCURVES)-ExtractedSignalCountEnd-2)
        print('This one 2:')
        print(len(signalCURVES)-ExtractedSignalCountStart-2)
        circuitbreakval = len(signalCURVES)-ExtractedSignalCountStart-2
        
        
        if breakval == len(signalCURVES):
            break
        
        breakval = 0
        print('First Loop')
        for i in tqdm(range(ExtractedSignalCountEnd+1,len(signalCURVES))):
            #breakval = 0
            #if i == ExtractedSignalCountStart: #this is editable
                    #break         
            if signalcount[i] == 1:
                ExtractedSignalCountStart = i
                StartVals = np.append(StartVals,ExtractedSignalCountStart)
                break
            if i == len(signalCURVES)-ExtractedSignalCountStart+1-2:
                print('1')
                print(len(signalCURVES)-ExtractedSignalCountEnd-2)
                breakval = len(signalCURVES)
                #break
            if i == len(signalCURVES)-ExtractedSignalCountStart-2:
                print('2')
                print(len(signalCURVES)-ExtractedSignalCountStart-2)
                breakval = len(signalCURVES)
                #break
                
        if breakval == len(signalCURVES):
            break
        print('Second Loop:')
        print(len(signalCURVES)-ExtractedSignalCountStart-2) 
        circuitbreakval2 = len(signalCURVES)-ExtractedSignalCountStart-2
        print(circuitbreakval2)
        print(circuitbreakval)   
        #breakval = 0
        for q in tqdm(range(ExtractedSignalCountStart+1,len(signalCURVES))):

            #if q == len(signalCURVES)-ExtractedSignalCountEnd-2:
                #print('3')
                #print(len(signalCURVES)-ExtractedSignalCountEnd-2)
                #breakval = len(signalCURVES)
                
            #if q == 100: #circuitbreakval2
                #print('4')
                #print(q)
                #print(circuitbreakval2)
                #breakval = len(signalCURVES)  
                              
            #if q == ExtractedSignalCountEnd: #this is editable
                    #break  
                    
            if q > circuitbreakval2-10: #circuitbreakval2 100
                print('4')
                print(q)
                print(circuitbreakval2)
                breakval = len(signalCURVES)
                break  
                                                
            if signalcount[q] == 0:
                ExtractedSignalCountEnd = q
                EndVals = np.append(EndVals,ExtractedSignalCountEnd)
                print(ExtractedSignalCountEnd)
                print(len(signalCURVES)-ExtractedSignalCountEnd-1)
                break
            
            if q == len(signalCURVES)-ExtractedSignalCountEnd-2:
                print('3')
                print(len(signalCURVES)-ExtractedSignalCountEnd-2)
                breakval = len(signalCURVES)
                break 
            #if q > circuitbreakval2-100: #circuitbreakval2 100
                #print('4')
                #print(q)
                #print(circuitbreakval2)
                #breakval = len(signalCURVES)
                #break  



            #if q == len(signalCURVES)-ExtractedSignalCountEnd-2:
                #print('3')
                #print(len(signalCURVES)-ExtractedSignalCountEnd-2)
                #breakval = len(signalCURVES)
                #break 
            #if q == len(signalCURVES)-ExtractedSignalCountStart-2:
                #print('4')
                #print(len(signalCURVES)-ExtractedSignalCountStart-2)
                #breakval = len(signalCURVES)
                
        print(StartVals)
        print(EndVals)
        print(breakval)
                
        #if breakval == len(signalCURVES):
            #break
        
        print()
        print()
        print()
    
    

    #print(StartVals)  
    #print(EndVals)
    
    if len(StartVals) - len(EndVals) == 1:
        EndVals = np.append(EndVals,len(signalCURVES))
    
    
    print(StartVals)  
    print(EndVals)


    return StartVals, EndVals, signalCURVES
    
    
def AnalyseSignals(StartVals,EndVals,signal,signalCURVES,signalcurveave):

    column_names_main = ['SignalNumber','SignalLength','NoPeaks','NoDips','NoPeaksBeforeGlobal','NoPeaksAfterGlobal','NoDipsBeforeGlobal','NoDipsAfterGlobal','DipDepth','MedianPeakDifference']
    MainSignalDF = pd.DataFrame(columns = column_names_main, dtype=object)


    numCOI = len(StartVals)
    extractedsignals = signalCURVES




#MAX LENGTH FOR END CROP IS 300000





    for K in range(numCOI):
        print()
        print()
    


        data = signal[int(StartVals[K]):int(EndVals[K]),1]
    

    
        ultra_smooth_param = 10000 #CAN EDIT
        smooth_param = 500 #CAN EDIT
        peakdipheight = 50 #CAN EDIT
    
        start_crop = 0 #CAN EDIT  MUST BE SAME AS TEST ABOVE
        end_crop = len(extractedsignals) #CAN EDIT  MUST BE SAME AS TEST ABOVE
    
        #data from scan, ultra_smooth param, smooth param, height of peaks/dips
        maxtabx, maxtaby, mintabx, mintaby, x, y_av, y_av_ultra_smooth = pipeline(data, 
                                                                              ultra_smooth_param, 
                                                                              smooth_param, 
                                                                              peakdipheight, 
                                                                              start_crop, 
                                                                              end_crop)
    

    #print('Graph for Un-Cropped Extracted Signal %s' %str(K+1))
    #plt.plot(extractedsignals[:,:,K])
    
        DipIndex = np.where(mintaby == mintaby.min())
        DipIndex = DipIndex[0]
    
    
        DipBeforeValsMin = (mintaby[:int(DipIndex)])
        DipAfterValsMin = (mintaby[int(DipIndex)+1:])
        DipRemoveMin = np.concatenate((DipBeforeValsMin,DipAfterValsMin))
    
        DipBeforeValsMax = (maxtaby[:int(DipIndex)])
        DipAfterValsMax = (maxtaby[int(DipIndex)+1:])
        DipRemoveMax = np.concatenate((DipBeforeValsMax,DipAfterValsMax))
    
    #Av Peak Height is the difference between each peak and corresponding dip excluding the global dip
    #So that it is not skewed
    
        maxtabxresize = maxtabx
        mintabxresize = mintabx
    
        maxtabyresize = maxtaby
        mintabyresize = mintaby
    
    #Currently only deleting 1 peak/dip
    #Need it so it is more robust if difference is more than 1
    
        if len(maxtabx) > len(mintabx) :
            print('The max is longer')
            #delete the longer of two down until equal length
            #maxtabxresize = maxtabx
            np.delete(maxtabx,len(maxtabx)-1) #np.delete(maxtabx,int(maxtabx[len(maxtabx)-1]))
            maxtabxresize=np.resize(maxtabx,len(maxtabx)-1)
        
            np.delete(maxtaby,len(maxtabx)-1) #changed y to x   np.delete(maxtaby,int(maxtabx[len(maxtabx)-1]))
            maxtabyresize=np.resize(maxtaby,len(maxtaby)-1)
        
        
        if len(mintabx) > len(maxtabx) :
            print('The min is longer')
        #delete the longer of two down until equal length
            mintabxresize = mintabx
        #np.delete(mintabx,int(mintabx[len(mintabx)-1]))
        #mintabxresize=np.resize(mintabx,len(mintabx)-1)
        
        #np.delete(mintaby,int(mintaby[len(mintaby)-1]))
        #mintabyresize=np.resize(mintaby,len(mintaby)-1)
        
    
        DipIndex = np.where(mintabyresize == mintabyresize.min())
        DipIndex = DipIndex[0]
    
    
        DipBeforeValsMin = (mintabyresize[:int(DipIndex)])
        DipAfterValsMin = (mintabyresize[int(DipIndex)+1:])
        DipRemoveMin = np.concatenate((DipBeforeValsMin,DipAfterValsMin))
    
        DipBeforeValsMax = (maxtabyresize[:int(DipIndex)])
        DipAfterValsMax = (maxtabyresize[int(DipIndex)+1:])
        DipRemoveMax = np.concatenate((DipBeforeValsMax,DipAfterValsMax))
    
        peakdip = np.stack((maxtabxresize,mintabxresize))
    
        PeaksBeforeMainDip = np.where(peakdip[0,:] <= mintabxresize[int(DipIndex)])
    
    
    
        print('Graph for Cropped Extracted Signal %s' %str(K+1))
        print('Length of Signal %s is Approximately %ss (%s Frames)' %(str(K+1), (int(EndVals[K]) - int(StartVals[K]))/5000, int(EndVals[K]) - int(StartVals[K])))
        print('Number of Peaks in Signal %s is %s' %(str(K+1), str(len(maxtabx))))
        print('Number of Dips in Signal %s is %s' %(str(K+1), str(len(mintabx))))
    
        print('Peaks Before Dip in Signal %s is %s' %(str(K+1), str(len(PeaksBeforeMainDip[0]))))
        print('Dips Before Dip in Signal %s is %s' %(str(K+1), str(int(DipIndex))))
    
        print('Peaks After Dip in Signal %s is %s' %(str(K+1), str(len(mintaby) - len(PeaksBeforeMainDip[0]))))
        print('Dips After Dip in Signal %s is %s' %(str(K+1), str(int(len(mintaby) - DipIndex - 1))))
        print('Median Peak Height in Signal %s is %s' %(str(K+1), np.median((DipRemoveMax+10000)-(DipRemoveMin+10000))))
        print('Dip Depth in Signal %s is %s' %(str(K+1), np.min(mintaby)))
    
        if np.min(mintaby) > -500:
            print()
            print('REMOVED DUE TO INCORRECT SIGNAL')
            print()
    

        if (int(EndVals[K]) - int(StartVals[K])) < 10000:
            print('SIGNAL TOO SHORT')
            continue

        plot_signal_point_plus(maxtabx, maxtaby, mintabx, mintaby, x, y_av, y_av_ultra_smooth)
    
        #crosscorr = scipy.signal.correlate(y_av_ultra_smooth, signalcurveave)
    
        #if np.max(crosscorr) > 1.5*10**10:
            #print('There is a cheeky signal nestled in here')
            #print()
            #print()
            #print()
            #ccstart = crosscorr.argmax()-500000
            #if ccstart < 0:
                #ccstart = 0
        
            #ccend = crosscorr.argmax()+500000
            #if ccend > len(crosscorr):
                #ccend = len(crosscorr)
    
        #if np.max(crosscorr) < 1.5*10**10:
            #ccstart = 0
            #ccend = 0
    
        #plt.plot(crosscorr)
        #plt.axvspan(ccstart, ccend, color='red', alpha=0.5)
        #plt.show()
        #print(np.max(crosscorr))
        #if np.max(crosscorr) > 1.5*10**10:
            #print('There is a cheeky signal nestled in here')
            #print()
            #print()
            #print()
            #print()
            #print()
    #plt.plot(y_av[ccstart:ccend])
    #plt.show()
        #if np.max(crosscorr) > 1.5*10**10:
            #normcrosscorr = crosscorr/np.max(crosscorr)
            #plt.plot(normcrosscorr[900000:])
            #plt.plot(y_av/np.max(y_av))
            #plt.show()
        #if np.max(crosscorr) > 1.5*10**10:
            #crosscorrsig = y_av[ccstart-900000:ccend]#-900000]
            #plt.plot(crosscorrsig)
            #plt.show()

    
        if np.min(mintaby) > -500:
            continue
    
        MainSignalDF.loc[K] = ['Signal' + str(K+1)]+[int(EndVals[K]) - int(StartVals[K])]+[len(maxtabx)]+[len(mintabx)]+[str(len(PeaksBeforeMainDip[0]))]+[str(len(mintabx) - len(PeaksBeforeMainDip[0]))]+[int(DipIndex)]+[int(len(mintaby) - DipIndex - 1)]+[np.min(mintaby)]+[np.median((DipRemoveMax+10000) - (DipRemoveMin+10000))]
       
    #for Q in range(numCOI): 
        #reducedsignal = signal[int(StartVals[Q]):int(EndVals[Q]),1]
    
    reducedsignal = signal[:,1] - signalCURVES
    reducedsignal = reducedsignal[reducedsignal != 0]
    newrowsig = np.arange(len(reducedsignal))
    reducedsignalconc = np.stack((newrowsig, reducedsignal), axis=0)
    reducedsignalconc = reducedsignalconc.T
                    


    return MainSignalDF, reducedsignalconc




def AnalyseSignalsFailsafeVer(StartVals,EndVals,signal,signalCURVES,DFnorm,signalcurveave,gap):

    column_names_main = ['SignalNumber','SignalLength','NoPeaks','NoDips','NoPeaksBeforeGlobal','NoPeaksAfterGlobal','NoDipsBeforeGlobal','NoDipsAfterGlobal','DipDepth','MedianPeakDifference']
    MainSignalDF = pd.DataFrame(columns = column_names_main, dtype=object)
    MainSignalDFcc = pd.DataFrame(columns = column_names_main, dtype=object)


    numCOI = len(StartVals)
    extractedsignals = signalCURVES




#MAX LENGTH FOR END CROP IS 300000



    StartValsReserve = StartVals
    EndValsReserve = EndVals
    StartValsReserve2 = StartVals
    EndValsReserve2 = EndVals
    
    
    
    for K in range(numCOI):
        print()
        print()
    


        data = signal[int(StartVals[K]):int(EndVals[K]),1]
        
        fulllength = (int(EndVals[K]) - int(StartVals[K]))
    

    
        ultra_smooth_param = 10000 #CAN EDIT
        smooth_param = 500 #CAN EDIT
        peakdipheight = 50 #CAN EDIT
    
        start_crop = 0 #CAN EDIT  MUST BE SAME AS TEST ABOVE
        end_crop = len(extractedsignals) #CAN EDIT  MUST BE SAME AS TEST ABOVE
    
        #data from scan, ultra_smooth param, smooth param, height of peaks/dips
        maxtabx, maxtaby, mintabx, mintaby, x, y_av, y_av_ultra_smooth = pipeline(data, 
                                                                              ultra_smooth_param, 
                                                                              smooth_param, 
                                                                              peakdipheight, 
                                                                              start_crop, 
                                                                              end_crop)
    

    #print('Graph for Un-Cropped Extracted Signal %s' %str(K+1))
    #plt.plot(extractedsignals[:,:,K])
    
        DipIndex = np.where(mintaby == mintaby.min())
        DipIndex = DipIndex[0]
    
    
        DipBeforeValsMin = (mintaby[:int(DipIndex)])
        DipAfterValsMin = (mintaby[int(DipIndex)+1:])
        DipRemoveMin = np.concatenate((DipBeforeValsMin,DipAfterValsMin))
    
        DipBeforeValsMax = (maxtaby[:int(DipIndex)])
        DipAfterValsMax = (maxtaby[int(DipIndex)+1:])
        DipRemoveMax = np.concatenate((DipBeforeValsMax,DipAfterValsMax))
    
    #Av Peak Height is the difference between each peak and corresponding dip excluding the global dip
    #So that it is not skewed
    
        maxtabxresize = maxtabx
        mintabxresize = mintabx
    
        maxtabyresize = maxtaby
        mintabyresize = mintaby
    
    #Currently only deleting 1 peak/dip
    #Need it so it is more robust if difference is more than 1
    
        if len(maxtabx) > len(mintabx) :
            print('The max is longer')
            #delete the longer of two down until equal length
            #maxtabxresize = maxtabx
            np.delete(maxtabx,len(maxtabx)-1) #np.delete(maxtabx,int(maxtabx[len(maxtabx)-1]))
            maxtabxresize=np.resize(maxtabx,len(maxtabx)-1)
        
            np.delete(maxtaby,len(maxtabx)-1) #changed y to x   np.delete(maxtaby,int(maxtabx[len(maxtabx)-1]))
            maxtabyresize=np.resize(maxtaby,len(maxtaby)-1)
        
        
        if len(mintabx) > len(maxtabx) :
            print('The min is longer')
        #delete the longer of two down until equal length
            mintabxresize = mintabx
        #np.delete(mintabx,int(mintabx[len(mintabx)-1]))
        #mintabxresize=np.resize(mintabx,len(mintabx)-1)
        
        #np.delete(mintaby,int(mintaby[len(mintaby)-1]))
        #mintabyresize=np.resize(mintaby,len(mintaby)-1)
        
    
        DipIndex = np.where(mintabyresize == mintabyresize.min())
        DipIndex = DipIndex[0]
    
    
        DipBeforeValsMin = (mintabyresize[:int(DipIndex)])
        DipAfterValsMin = (mintabyresize[int(DipIndex)+1:])
        DipRemoveMin = np.concatenate((DipBeforeValsMin,DipAfterValsMin))
    
        DipBeforeValsMax = (maxtabyresize[:int(DipIndex)])
        DipAfterValsMax = (maxtabyresize[int(DipIndex)+1:])
        DipRemoveMax = np.concatenate((DipBeforeValsMax,DipAfterValsMax))
    
        peakdip = np.stack((maxtabxresize,mintabxresize))
    
        PeaksBeforeMainDip = np.where(peakdip[0,:] <= mintabxresize[int(DipIndex)])
        
        crosscorrorig = scipy.signal.correlate(y_av_ultra_smooth, signalcurveave)
    
    
    
        print('Graph for Cropped Extracted Signal %s' %str(K+1))
        print('Length of Signal %s is Approximately %ss (%s Frames)' %(str(K+1), (int(EndVals[K]) - int(StartVals[K]))/5000, int(EndVals[K]) - int(StartVals[K])))
        print('Number of Peaks in Signal %s is %s' %(str(K+1), str(len(maxtabx))))
        print('Number of Dips in Signal %s is %s' %(str(K+1), str(len(mintabx))))
    
        print('Peaks Before Dip in Signal %s is %s' %(str(K+1), str(len(PeaksBeforeMainDip[0]))))
        print('Dips Before Dip in Signal %s is %s' %(str(K+1), str(int(DipIndex))))
    
        print('Peaks After Dip in Signal %s is %s' %(str(K+1), str(len(mintaby) - len(PeaksBeforeMainDip[0]))))
        print('Dips After Dip in Signal %s is %s' %(str(K+1), str(int(len(mintaby) - DipIndex - 1))))
        print('Median Peak Height in Signal %s is %s' %(str(K+1), np.median((DipRemoveMax+10000)-(DipRemoveMin+10000))))
        print('Dip Depth in Signal %s is %s' %(str(K+1), np.min(mintaby)))
        
        if (int(EndVals[K]) - int(StartVals[K])) < 10000:
            print('SIGNAL TOO SHORT')
            StartValsReserve2[K] = -1
            EndValsReserve2[K] = -1
            continue
        
        plot_signal_point_plus(maxtabx, maxtaby, mintabx, mintaby, x, y_av, y_av_ultra_smooth)
        
        if np.max(crosscorrorig) > 1.5*10**10:
            print()
            print(np.max(crosscorrorig))
            print('High Chance of Good Signal')
        
        if np.max(crosscorrorig) < 1.5*10**10:
            print()
            print(np.max(crosscorrorig))
            print('High Chance of Bad Signal')
            np.delete(StartValsReserve,[K])
            np.delete(EndValsReserve,[K])
            
            StartValsReserve2[K] = -1
            EndValsReserve2[K] = -1
    
        if np.min(mintaby) > -500:
            print()
            print('REMOVED DUE TO INCORRECT SIGNAL')
            print()
            StartValsReserve2[K] = -1
            EndValsReserve2[K] = -1
            
    

        #if (int(EndVals[K]) - int(StartVals[K])) < 10000:
            #print('SIGNAL TOO SHORT')
            #StartValsReserve2[K] = -1
            #EndValsReserve2[K] = -1
            #continue

        #plot_signal_point_plus(maxtabx, maxtaby, mintabx, mintaby, x, y_av, y_av_ultra_smooth)
        
        
        if (int(EndVals[K]) - int(StartVals[K])) > 1250000:
        
    
            crosscorr = scipy.signal.correlate(y_av_ultra_smooth, signalcurveave)
            
    
            if np.max(crosscorr) > 1.5*10**10:
                print('There may be a signal here')
                print()
                print()
                print()
                ccstart = crosscorr.argmax()-500000
                if ccstart < 0:
                    ccstart = 0
        
                ccend = crosscorr.argmax()+500000
                if ccend > len(crosscorr):
                    ccend = len(crosscorr)
    
            if np.max(crosscorr) < 1.5*10**10:
                ccstart = 0
                ccend = 0
                
                np.delete(StartValsReserve,[K])
                np.delete(EndValsReserve,[K])
                
                StartValsReserve2[K] = -1
                EndValsReserve2[K] = -1
    
            plt.plot(crosscorr)
            plt.axvspan(ccstart, ccend, color='red', alpha=0.5)
            plt.show()
            print('Max CrossCorr value: %s' %np.max(crosscorr))
            if np.max(crosscorr) > 1.5*10**10:
                print('There is a signal here')
                #print()
                #print()
                #print()
                #print()
                #print()
    #plt.plot(y_av[ccstart:ccend])
    #plt.show()
            #if np.max(crosscorr) > 1.5*10**10: #switching from 900000 to 600000
               # normcrosscorr = crosscorr/np.max(crosscorr)
                #plt.plot(normcrosscorr[int(+ccstart-600000):int(ccend)]) #900000 and ccstart
                #plt.plot(y_av/np.max(y_av))
                #plt.show()
        #if np.max(crosscorr) > 1.5*10**10:
            #crosscorrsig = y_av[int(StartVals[K]+ccstart-900000):int(StartVals[K]+ccend)]#-900000] and ccstart
            #plt.plot(crosscorrsig)
            #plt.show()
            if np.max(crosscorr) > 1.5*10**10:
                #print(StartVals[K]+ccstart-600000)
                #print(StartVals[K]+ccend)
                if StartVals[K]+ccstart < 600001: #startvalsk added
                    ccstart = StartVals[K]+600001 #startvalsk added
                StartValsFS, EndValsFS, signalCURVESFS = PredictSignalInFS(DFnorm[int(StartVals[K]+ccstart-600000):int(StartVals[K]+ccend)],signal[int(StartVals[K]+ccstart-600000):int(StartVals[K]+ccend)],gap)
                #print(StartValsFS)
                #print(EndValsFS)
                #print(StartValsFS+int(StartVals[K]+ccstart-600000))
                #print(EndValsFS+int(StartVals[K]+ccend)) #EndValsFS+int(StartVals[K]+ccend)
                if np.max(crosscorr) > 1.5*10**10:
                    extractedccsig = signal[int(StartVals[K]+ccstart-600000+(StartValsFS[0])):int(StartVals[K]+ccstart-600000+(EndValsFS[0]))]#-900000] and ccstart    np.min(StartValsFS) np.max(EndValsFS)
                    plt.plot(extractedccsig[:,1])
                    plt.show()
                    
                    #maxtabxetr, maxtabyextr, mintabxextr, mintabyextr, xextr, y_avextr, y_av_ultra_smoothextr = pipeline(extractedccsig[:,1], 
                                                                              #10000, 
                                                                              #500, 
                                                                              #50, 
                                                                              #0, 
                                                                              #len(extractedccsig))
                    #print('plot of UltraSmooth')                                                   
                    #plt.plot(y_av_ultra_smoothextr)
                    #plt.show()
                    
                    #crosscorrsigextract = scipy.signal.correlate(y_av_ultra_smoothextr, signalcurveave)
                    #print('CrossCorr Value:')
                    #print(np.max(crosscorrsigextract))
                    #print()
                    #print()
                    
                StartValsFS = StartValsFS+int(StartVals[K]+ccstart-600000)
                EndValsFS = EndValsFS+int(StartVals[K]+ccstart-600000) #EndValsFS+int(StartVals[K]+ccend)
                StartValsReserve[K] = StartValsFS[0]
                EndValsReserve[K] = EndValsFS[0]
                StartValsReserve2[K] = StartValsFS[0]
                EndValsReserve2[K] = EndValsFS[0]
                MainSignalDFFS = AnalyseSignalsnocc(StartValsFS,EndValsFS,signal,signalCURVESFS,signalcurveave,K)
                MainSignalDFcc = pd.concat([MainSignalDFcc,MainSignalDFFS])

    
        if np.min(mintaby) > -500:
            continue
            
    
        MainSignalDF.loc[K] = ['Signal' + str(K+1)]+[int(EndVals[K]) - int(StartVals[K])]+[len(maxtabx)]+[len(mintabx)]+[str(len(PeaksBeforeMainDip[0]))]+[str(len(mintabx) - len(PeaksBeforeMainDip[0]))]+[int(DipIndex)]+[int(len(mintaby) - DipIndex - 1)]+[np.min(mintaby)]+[np.median((DipRemoveMax+10000) - (DipRemoveMin+10000))]
        
        if MainSignalDF.loc[K]['SignalLength'] == 0:
            MainSignalDF = MainSignalDF.drop([K])
            
        if fulllength > 1250000:
            #print('CrossCorr for maybe deleting:')
            #print(np.max(crosscorrorig))
            if np.max(crosscorrorig) > 1.5*10**10:
                MainSignalDF = MainSignalDF.drop([K])
        
    reducedsignal = signal[:,1]                
    #for Q in range(numCOI):
 
        #reducedsignalcurves = np.append(reducedsignal,signal[int(StartValsReserve[Q]):int(EndValsReserve[Q]),1])
        #print(Q)
        #print(StartValsReserve)
        #print(EndValsReserve)
        #print()
        #print(reducedsignal[int(StartValsReserve[Q]):int(EndValsReserve[Q])])
        #print(np.shape(reducedsignal[int(StartValsReserve[Q]):int(EndValsReserve[Q])]))
        #print(len(reducedsignal[int(StartValsReserve[Q]):int(EndValsReserve[Q])]))
        #print()
        #print(signal[int(StartValsReserve[Q]):int(EndValsReserve[Q]),1])
        #print(np.shape(signal[int(StartValsReserve[Q]):int(EndValsReserve[Q]),1]))
        #print()
        
        #reducedsignal = reducedsignal[int(StartValsReserve[Q]):int(EndValsReserve[Q])] - signal[int(StartValsReserve[Q]):int(EndValsReserve[Q]),1] #try *0, removed ,1 from first part
        
        #reducedsignal = reducedsignal[int(StartValsReserve[Q]):int(EndValsReserve[Q])]*0 #,1 removed
        #plt.plot(reducedsignal[int(StartValsReserve[Q]):int(EndValsReserve[Q])])
        #plt.show()
    
    print('Before Editing:')
    print(StartValsReserve)
    print(EndValsReserve)
    reducedsignal = signal[:,1]# - reducedsignalcurves
    
    #reducedsignal = reducedsignal[reducedsignal != 0]
    
    StartValsReserve2 = StartValsReserve2[StartValsReserve2 != -1]
    EndValsReserve2 = EndValsReserve2[EndValsReserve2 != -1]
    print('After Editing:')
    print(StartValsReserve2)
    print(EndValsReserve2)
    
    for Q in range(len(StartValsReserve2)):
        print()
        print()
        print('Plot of Curve %s' %Q)
        plt.plot(reducedsignal[int(StartValsReserve2[Q]):int(EndValsReserve2[Q])])
        plt.show()
        
    #reducedsignal = np.arange(len(signal))*0 #old
    for Q in range(len(StartValsReserve2)):
        #reducedsignal[int(StartValsReserve2[Q]):int(EndValsReserve2[Q])] = signal[int(StartValsReserve2[Q]):int(EndValsReserve2[Q]),1] #old
        
        reducedsignal[int(StartValsReserve2[Q]):int(EndValsReserve2[Q])] = 0 #new
    
    #reducedsignalblank = reducedsignal
    reducedsignal = reducedsignal[reducedsignal != 0] #new
    newrowsig = np.arange(len(reducedsignal))
    reducedsignal = np.stack((newrowsig, reducedsignal), axis=0) #reducedsignalconc
    reducedsignal = reducedsignal.T #reducedsignalconc


    return MainSignalDF,MainSignalDFcc, reducedsignal#,reducedsignalblank, reducedsignalcurves
    
    
def AnalyseSignalsnocc(StartVals,EndVals,signal,signalCURVES,signalcurveave,signum):

    column_names_main = ['SignalNumber','SignalLength','NoPeaks','NoDips','NoPeaksBeforeGlobal','NoPeaksAfterGlobal','NoDipsBeforeGlobal','NoDipsAfterGlobal','DipDepth','MedianPeakDifference']
    MainSignalDF = pd.DataFrame(columns = column_names_main, dtype=object)


    numCOI = len(EndVals)
    extractedsignals = signalCURVES




#MAX LENGTH FOR END CROP IS 300000





    for K in range(numCOI):
        print()
        print()
    


        data = signal[int(StartVals[K]):int(EndVals[K]),1]
    

    
        ultra_smooth_param = 10000 #CAN EDIT
        smooth_param = 500 #CAN EDIT
        peakdipheight = 50 #CAN EDIT
    
        start_crop = 0 #CAN EDIT  MUST BE SAME AS TEST ABOVE
        end_crop = len(extractedsignals) #CAN EDIT  MUST BE SAME AS TEST ABOVE
    
        #data from scan, ultra_smooth param, smooth param, height of peaks/dips
        maxtabx, maxtaby, mintabx, mintaby, x, y_av, y_av_ultra_smooth = pipeline(data, 
                                                                              ultra_smooth_param, 
                                                                              smooth_param, 
                                                                              peakdipheight, 
                                                                              start_crop, 
                                                                              end_crop)
    

    #print('Graph for Un-Cropped Extracted Signal %s' %str(K+1))
    #plt.plot(extractedsignals[:,:,K])
    
        DipIndex = np.where(mintaby == mintaby.min())
        DipIndex = DipIndex[0]
    
    
        DipBeforeValsMin = (mintaby[:int(DipIndex)])
        DipAfterValsMin = (mintaby[int(DipIndex)+1:])
        DipRemoveMin = np.concatenate((DipBeforeValsMin,DipAfterValsMin))
    
        DipBeforeValsMax = (maxtaby[:int(DipIndex)])
        DipAfterValsMax = (maxtaby[int(DipIndex)+1:])
        DipRemoveMax = np.concatenate((DipBeforeValsMax,DipAfterValsMax))
    
    #Av Peak Height is the difference between each peak and corresponding dip excluding the global dip
    #So that it is not skewed
    
        maxtabxresize = maxtabx
        mintabxresize = mintabx
    
        maxtabyresize = maxtaby
        mintabyresize = mintaby
    
    #Currently only deleting 1 peak/dip
    #Need it so it is more robust if difference is more than 1
    
        if len(maxtabx) > len(mintabx) :
            print('The max is longer')
            #delete the longer of two down until equal length
            #maxtabxresize = maxtabx
            np.delete(maxtabx,len(maxtabx)-1) #np.delete(maxtabx,int(maxtabx[len(maxtabx)-1]))
            maxtabxresize=np.resize(maxtabx,len(maxtabx)-1)
        
            np.delete(maxtaby,len(maxtabx)-1) #changed y to x   np.delete(maxtaby,int(maxtabx[len(maxtabx)-1]))
            maxtabyresize=np.resize(maxtaby,len(maxtaby)-1)
        
        
        if len(mintabx) > len(maxtabx) :
            print('The min is longer')
        #delete the longer of two down until equal length
            mintabxresize = mintabx
        #np.delete(mintabx,int(mintabx[len(mintabx)-1]))
        #mintabxresize=np.resize(mintabx,len(mintabx)-1)
        
        #np.delete(mintaby,int(mintaby[len(mintaby)-1]))
        #mintabyresize=np.resize(mintaby,len(mintaby)-1)
        
    
        DipIndex = np.where(mintabyresize == mintabyresize.min())
        DipIndex = DipIndex[0]
    
    
        DipBeforeValsMin = (mintabyresize[:int(DipIndex)])
        DipAfterValsMin = (mintabyresize[int(DipIndex)+1:])
        DipRemoveMin = np.concatenate((DipBeforeValsMin,DipAfterValsMin))
    
        DipBeforeValsMax = (maxtabyresize[:int(DipIndex)])
        DipAfterValsMax = (maxtabyresize[int(DipIndex)+1:])
        DipRemoveMax = np.concatenate((DipBeforeValsMax,DipAfterValsMax))
    
        peakdip = np.stack((maxtabxresize,mintabxresize))
    
        PeaksBeforeMainDip = np.where(peakdip[0,:] <= mintabxresize[int(DipIndex)])
    
    
    
        print('Graph for Cropped Extracted Signal %s' %str(K+1))
        print('Length of Signal %s is Approximately %ss (%s Frames)' %(str(K+1), (int(EndVals[K]) - int(StartVals[K]))/5000, int(EndVals[K]) - int(StartVals[K])))
        print('Number of Peaks in Signal %s is %s' %(str(K+1), str(len(maxtabx))))
        print('Number of Dips in Signal %s is %s' %(str(K+1), str(len(mintabx))))
    
        print('Peaks Before Dip in Signal %s is %s' %(str(K+1), str(len(PeaksBeforeMainDip[0]))))
        print('Dips Before Dip in Signal %s is %s' %(str(K+1), str(int(DipIndex))))
    
        print('Peaks After Dip in Signal %s is %s' %(str(K+1), str(len(mintaby) - len(PeaksBeforeMainDip[0]))))
        print('Dips After Dip in Signal %s is %s' %(str(K+1), str(int(len(mintaby) - DipIndex - 1))))
        print('Median Peak Height in Signal %s is %s' %(str(K+1), np.median((DipRemoveMax+10000)-(DipRemoveMin+10000))))
        print('Dip Depth in Signal %s is %s' %(str(K+1), np.min(mintaby)))
        

        #print('plot of UltraSmooth')                                                   
        #plt.plot(y_av_ultra_smooth)
        #plt.show()
                    
        crosscorrsigextract = scipy.signal.correlate(y_av_ultra_smooth, signalcurveave)
        print('CrossCorr Value:')
        print(np.max(crosscorrsigextract))
        print()
        print()
        
        if np.max(crosscorrsigextract) > 1.5*10**10:
            print('Signal in this extraction')
        
        if np.max(crosscorrsigextract) < 1.5*10**10:
            print('Signal not in this extraction')
            #continue
            
            #np.delete(StartValsReserve,[K])
            #np.delete(EndValsReserve,[K])
                
            #StartValsReserve2[K] = -1
            #EndValsReserve2[K] = -1
    
        if np.min(mintaby) > -500:
            print()
            print('REMOVED DUE TO INCORRECT SIGNAL')
            print()
    

        if (int(EndVals[K]) - int(StartVals[K])) < 10000:
            print('SIGNAL TOO SHORT')
            continue

        plot_signal_point_plus(maxtabx, maxtaby, mintabx, mintaby, x, y_av, y_av_ultra_smooth)
        
        if np.max(crosscorrsigextract) < 1.5*10**10:
            continue

    
        if np.min(mintaby) > -500:
            continue
            
        
    
        MainSignalDF.loc[K] = ['Signal' + str(signum+1) + '.' + str(K+1)]+[len(y_av_ultra_smooth)]+[len(maxtabx)]+[len(mintabx)]+[str(len(PeaksBeforeMainDip[0]))]+[str(len(mintabx) - len(PeaksBeforeMainDip[0]))]+[int(DipIndex)]+[int(len(mintaby) - DipIndex - 1)]+[np.min(mintaby)]+[np.median((DipRemoveMax+10000) - (DipRemoveMin+10000))]
        
        #MainSignalDF.loc[K] = ['Signal' + str(K+1)]+[int(EndVals[K]) - int(StartVals[K])]+[len(maxtabx)]+[len(mintabx)]+[str(len(PeaksBeforeMainDip[0]))]+[str(len(mintabx) - len(PeaksBeforeMainDip[0]))]+[int(DipIndex)]+[int(len(mintaby) - DipIndex - 1)]+[np.min(mintaby)]+[np.median((DipRemoveMax+10000) - (DipRemoveMin+10000))]


    return MainSignalDF
    



def AnalyseSignalFullData(signal):

    column_names_main = ['SignalLength (s)','NoPeaks','NoDips','MedianPeakDifference','AvePeakSeparation (s)']
    MainSignalDF = pd.DataFrame(columns = column_names_main, dtype=object)


    #numCOI = len(EndVals)
    #extractedsignals = signalCURVES




#MAX LENGTH FOR END CROP IS 300000





    print()
    print()
    


    data = signal[:,1]
    

    
    ultra_smooth_param = 10000 #CAN EDIT
    smooth_param = 500 #CAN EDIT
    peakdipheight = 350 #CAN EDIT
    
    start_crop = 0 #CAN EDIT  MUST BE SAME AS TEST ABOVE
    end_crop = len(signal) #CAN EDIT  MUST BE SAME AS TEST ABOVE
    
        #data from scan, ultra_smooth param, smooth param, height of peaks/dips
    maxtabx, maxtaby, mintabx, mintaby, x, y_av, y_av_ultra_smooth = pipeline(data, 
                                                                              ultra_smooth_param, 
                                                                              smooth_param, 
                                                                              peakdipheight, 
                                                                              start_crop, 
                                                                              end_crop)
    

    #print('Graph for Un-Cropped Extracted Signal %s' %str(K+1))
    #plt.plot(extractedsignals[:,:,K])
    
    DipIndex = np.where(mintaby == mintaby.min())
    DipIndex = DipIndex[0]
    
    
    #Av Peak Height is the difference between each peak and corresponding dip excluding the global dip
    #So that it is not skewed
    
    maxtabxresize = maxtabx
    mintabxresize = mintabx
    
    maxtabyresize = maxtaby
    mintabyresize = mintaby
    
    #Currently only deleting 1 peak/dip
    #Need it so it is more robust if difference is more than 1
    
    if len(maxtabx) > len(mintabx) :
        print('The max is longer')
        #delete the longer of two down until equal length
        #maxtabxresize = maxtabx
        np.delete(maxtabx,len(maxtabx)-1) #np.delete(maxtabx,int(maxtabx[len(maxtabx)-1]))
        maxtabxresize=np.resize(maxtabx,len(maxtabx)-1)
        
        np.delete(maxtaby,len(maxtabx)-1) #changed y to x   np.delete(maxtaby,int(maxtabx[len(maxtabx)-1]))
        maxtabyresize=np.resize(maxtaby,len(maxtaby)-1)
        
        
    if len(mintabx) > len(maxtabx) :
        print('The min is longer')
        #delete the longer of two down until equal length
        mintabxresize = mintabx
        #np.delete(mintabx,int(mintabx[len(mintabx)-1]))
        #mintabxresize=np.resize(mintabx,len(mintabx)-1)
        
        #np.delete(mintaby,int(mintaby[len(mintaby)-1]))
        #mintabyresize=np.resize(mintaby,len(mintaby)-1)
        
    
    DipIndex = np.where(mintabyresize == mintabyresize.min())
    DipIndex = DipIndex[0]
    

    
    peakdip = np.stack((maxtabxresize,mintabxresize))

    
    
    
    print('Graph of Signal')
    print('Length of Signal is Approximately %ss (%s Frames)' % ((len(signal))/5000, len(signal)))
    print('Number of Peaks in Signal is %s' %str(len(maxtabx)))
    print('Number of Dips in Signal is %s' %str(len(mintabx)))
    

    print('Median Peak Height in Signal is %s' %(np.median(maxtaby)))

    

    if len(signal) < 10000:
       print('SIGNAL TOO SHORT')
       

    plot_signal_point_plus(maxtabx, maxtaby, mintabx, mintaby, x, y_av, y_av_ultra_smooth)

    
    MainSignalDF.loc[0] = [len(signal)/5000]+[len(maxtabx)]+[len(mintabx)]+[np.median(maxtaby)]+[(len(signal)/5000)/len(maxtabx)]


    return MainSignalDF




