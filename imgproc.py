# Image processing for Toolkit
# The software suite is released under the CCO License, with 
# acceptable parts compatable with the GNU Public License.  If you 
# use the software as is or as a basis for expansion, the original 
# author (Ash Turza) would appreciate acknowledgement but it is not 
# required.  
#
# Ash Turza can be found at kaazrut@gmail.com.
#
# This is the toplevel script for the software toolkit.  Check the 
# README for detailed instructions on use.


import os
import glob
import color_detect
import size_calibration
import numpy as np
import matplotlib.pyplot as plt
import sys
import logging
from scipy.signal import savgol_filter, filtfilt
from scipy.spatial import distance
import time
import tkinter as tk
from tkinter import filedialog
import ipdb

#change per test run
#PX2IN = 2460 #px to inches based on image reference, taken from matlab's distance tool
bag = 9.8 #weight of bag in g

#initialize variables; don't touch these.
#PX2MM = PX2IN/25.4  #px to mm conversion
fileList = []
forceVal = []
distVal = []
FORCE_LABEL = 'Force (N)'
prop_cycle = plt.rcParams['axes.prop_cycle']

fileCycle = [0,0] #num, size
NUM_COLORS = 20

#creating logger; especially do not touch this unless you know what you're doing
logger = logging.getLogger(__name__)
handler = logging.FileHandler('temp/imgproc.log')
logger.setLevel(logging.INFO)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

#create plots for single test data
def singresplot(xvar, yvar, xax, yax, plotTitle, testName, path):
    print('Generating plots...')
    plt.figure()
    plt.plot(xvar, yvar, linewidth=2.0)
    plt.xlabel(xax, fontsize=14)
    plt.ylabel(yax, fontsize=14)
    plt.title(plotTitle, fontsize=16)
    plt.savefig(os.path.join(path, testName + '_raw.png'))

    plt.show()

#create plots with multiple test data
def multiplot(xvar, yvar, xax, yax, plotTitle, testName, path):
    print('Generating single plot...')
    plt.figure()
    ax = plt.subplot(111)
    cmap = plt.get_cmap('jet')
    ax.set_color_cycle(cmap(2*i/NUM_COLORS) for i in range(NUM_COLORS))
    for j, k in enumerate(yvar):
        ax.plot(xvar, yvar[j], linewidth=1.0, label='Trial %s' % j)

    ax.set_xlabel(xax, fontsize=14)
    ax.set_ylabel(yax, fontsize=14)
    ax.set_title(plotTitle, fontsize=16)    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(path, testName + '-all.png'))
    plt.close()

#create plots of both the mean for multiple tests and filtered linear-regression of multiple tests
def meanplot(xvar, yvar, stdError, xax, yax, plotTitle, testName, sgFiltered, flag, path):
    if flag =='dist':
        testName = testName + 'Dist'
    else:
        testName = testName + 'DeltaD'

    print('Generating error plots...')
    plt.figure()
    plt.errorbar(xvar, yvar, yerr=stdError, linewidth=1.0)
    plt.xlabel(xax, fontsize=12)
    plt.ylabel(yax, fontsize=12)
    plt.title(plotTitle + ' Mean, Raw', fontsize=14)
    plt.savefig(os.path.join(path, testName + '-errorbar.png'))
    plt.clf()

    plt.errorbar(xvar, sgFiltered, yerr=stdError, linewidth=1.0)
    plt.xlabel(xax, fontsize=12)
    plt.ylabel(yax, fontsize=12)
    plt.title(plotTitle, fontsize=14)
    plt.savefig(os.path.join(path, testName + '-sg-errorbar.png'))
    plt.clf()

    plt.errorbar(xvar, yvar, yerr=stdError, linewidth=1.0, label='Raw')
    plt.plot(xvar, sgFiltered, linewidth=1.0, label='Smoothed')
    plt.xlabel(xax, fontsize=12)
    plt.ylabel(yax, fontsize=12)
    #plt.xlim((0 ,1.05)) #for axis manipulation
    plt.title(plotTitle + ' Mean', fontsize=14)
    plt.legend(loc=0)
    plt.savefig(os.path.join(path, testName +'-means.png'))
    plt.close()

#code for the compression tests
def compression(path, px2mm, flag):
    distVal = []
    forceVal= []
    deltaVal=[]
    i = 0

    for infile in glob.glob (os.path.join(path, '*.jpg')):
        currentImage = os.path.basename(infile)
        basepath = os.path.dirname(infile)
        pixelVec = color_detect.cdetect(basepath, currentImage, flag)
        fileList.append(infile)
        fsize = os.path.getsize(infile)
        com = pixelVec[1]
        #calculate vertical displacement
        diff = abs(com[0][1] - com[1][1])/px2mm
        distVal.append(diff)
        if i == 0:
            delta = 0
        else:
            delta = abs(distVal[0] - diff)
        force = i*2.5*10**-3*9.81 # # of pennies * weight * g
        forceVal.append(force)
        deltaVal.append(delta)
        i = i+1 #thank you python for being 0-index
        fileCycle[0] = fileCycle[0] + 1
        fileCycle[1] = fileCycle[1] + fsize
    return(distVal, forceVal, deltaVal)    

#tests that require using an angle: shear and bending; use flag to determine which one
def thetadep(path, px2mm, flag):
    distVal = []
    forceVal= []
    deltaVal= []
    i = 0
    j = 0

    for infile in glob.glob (os.path.join(path, '*.jpg')):
        currentImage = os.path.basename(infile)
        basepath = os.path.dirname(infile)
        pts = color_detect.cdetect(basepath, currentImage, flag)
        fileList.append(infile)
        fsize = os.path.getsize(infile)

        if flag == 2:
            if len(pts) < 2:
                diff = np.nan
            else:    
                x1 = pts[1][0] - pts[0][0]
                y1 = pts[1][1] - pts[0][1]
                theta = np.arctan2(y1, x1)
                hyp = distance.euclidean(pts[0], pts[1])
                diff = abs(hyp*np.sin(theta))/px2mm
        else:
            u1 = (pts[0][0][1] - pts[0][1][0])
            u2 = (pts[0][1][1] - pts[0][0][1])
            vector1 = (u1, u2)
            v1 = (pts[1][0][1] - pts[1][1][0])
            v2 = (pts[1][1][1] - pts[1][0][1])
            vector2 = (v1, v2)
            dotV = np.dot(vector1, vector2)
            mag1 = np.sqrt(np.dot(vector1, vector1))
            mag2 = np.sqrt(np.dot(vector2, vector2))
            #why it's measuring from 180 I have no idea
            diff = np.rad2deg(np.arccos(dotV/(mag1*mag2)))
            if diff-180 >= 0:
                diff = 360 - diff

        distVal.append(np.float64(diff))

        if j == 0:
            force = 0
            delta = 0
        else:
            delta = abs(distVal[0] - diff)
            force = ((i*2.5)+bag)*10**-3*9.81
            i = i+1

        forceVal.append(force)
        deltaVal.append(delta)
        j = j+1
        fileCycle[0] = fileCycle[0] + 1
        fileCycle[1] = fileCycle[1] + fsize
    return(distVal, forceVal, deltaVal)    

def statanalysis(subDist, forceVal, testNum, studyID, testName, distLabel, path, flag):
    #and here we do the actual statistical analysis of the aggretate data
    subDist = np.array(subDist)
    for x in np.nditer(subDist, op_flags=['readwrite']):
        if flag == 1:
            if x < 1.:
                x[...] = np.nan
        elif flag == 2:
            if x > 20:
                x[...] = np.nan
        else:
            continue

    distMean = np.nanmean(subDist, axis=0, dtype=np.float64)
    std = np.nanstd(subDist, axis=0, dtype=np.float64)
    stdError = std/np.sqrt(testNum)

    #Savitzky-Golay filtering (window size: 13, polyorder: 5)
    sgDistFilt = savgol_filter(distMean, 13, 5, axis=0)
    #manipulate arrays to export raw data to .csv files for importation into other software
    if flag == 1:
        forV = np.array(forceVal)
        multiplot(forceVal, subDist, FORCE_LABEL, distLabel, studyID, testName, path)
        meanplot(forceVal, distMean, stdError, FORCE_LABEL, distLabel, studyID, testName, sgDistFilt, 'dist', path)
    else:
        forV = np.array(forceVal[0])
        multiplot(forceVal[0], subDist, FORCE_LABEL, distLabel, studyID, testName, path)
        meanplot(forceVal[0], distMean, stdError, FORCE_LABEL, distLabel, studyID, testName, sgDistFilt, 'dist', path)
    dataCombo = np.c_[forV, subDist.T]  
    np.savetxt(os.path.join(path,testName +'.csv'), dataCombo, delimiter=',')
    np.savetxt(testName +'.csv', dataCombo, delimiter=',')


#the difference here is that this section takes the delta distance (displacement) and plots it
def statdisplacement(sDisplace, forceVal, testNum, studyID, testName, dispLabel, path):
    sDisplace = np.array(sDisplace)
    for x in np.nditer(sDisplace, op_flags=['readwrite']):
        if x > 100:
            x[...] = np.nan
    displaceMean = np.nanmean(sDisplace, axis=0, dtype=np.float64)
    dispSTD = np.nanstd(sDisplace, axis=0, dtype=np.float64)
    dispSTDError = dispSTD/np.sqrt(testNum)
    sgDisplaceFilt = savgol_filter(displaceMean, 13, 5, axis=0)
    meanplot(forceVal, displaceMean, dispSTDError, FORCE_LABEL, dispLabel, studyID, testName, sgDisplaceFilt, 'disp', path)        

def statdisplacementalt(sDisplace, subDist, forceVal, testNum, studyID, testName, dispLabel, path, flag):
    subDist = np.array(subDist)
    sDisplace = np.array(sDisplace)
    for x in np.nditer(sDisplace, op_flags=['readwrite']):
        if x > 100:
            x[...] = np.nan
    for x in np.nditer(subDist, op_flags=['readwrite']):
        if flag == 1:
            if x < 1.:
                x[...] = np.nan
        elif flag == 2:
            if x > 20:
                x[...] = np.nan
        else:
            continue
    distMean = np.nanmean(subDist, axis=0, dtype=np.float64)
    displaceMean = distMean - distMean[0]         
    dispSTD = np.nanstd(subDist, axis=0, dtype=np.float64)
    dispSTDError = dispSTD/np.sqrt(testNum)
    sgDisplaceFilt = savgol_filter(displaceMean, 13, 5, axis=0)
    meanplot(forceVal, displaceMean, dispSTDError, FORCE_LABEL, dispLabel, studyID, testName, sgDisplaceFilt, 'disp', path)

def main():
    compBool = False
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(parent=root,initialdir="/",title='Please select a directory')
    if not path:
        sys.exit()

    while True:
    #python why are you case sensitive?
        testType = input('Test type (COMpression, SHear, BENding): ')
        if testType not in {'COM', 'SH', 'BEN', 'com', 'sh', 'ben'}:
            logger.warn('Incorrect input; entered as %s', testType)
            continue
        else:
            break
    while True:        
        compTest = input('Comprehensive (y/n) ?')
        if compTest not in {'y', 'n', 'Y', 'N'}:
            compTest = input('Comprehensive (y/n)?')
            continue
        if compTest in ['y', 'Y']:
            compBool = True 
            break
        else:
            break
    while True:            
        testName = input('Save as filename.png: ')
        if not testName:
            testName = input('Save as filename.png: ')
            continue
        else:
            break

    #this is the worst hack ever
    if compBool:
        subDist = []
        subForce = []
        sDisplace = []
        topdir = path
        subdir = [os.path.abspath(x[0]) for x in os.walk(topdir)]
        subdir.remove(os.path.abspath(topdir))
        testNum = len(subdir)
        logger.info('Compilation run start')
        startTime = time.time()

        if testType in ('COM', 'com'):
            typeFlag = 1
            for currentdir in subdir:
                print(currentdir)
                topimg = os.listdir(currentdir)[0]
                #px2mm = PX2MM #toggle if reference swatch does not exist
                px2mm = size_calibration.sizecalib(currentdir, topimg, typeFlag)
                distVal, forceVal, deltaVal = compression(currentdir, px2mm, typeFlag)
                subDist.append(distVal)
                subForce.append(forceVal)
                sDisplace.append(deltaVal)
            studyID = 'Compression Test (trials: {0})'.format(testNum)
            distLabel = 'Distance (mm)'
            dispLabel = 'Displacement $\Delta d$ (mm)'
            statanalysis(subDist, forceVal, testNum, studyID, testName, distLabel, path, typeFlag)
            statdisplacement(sDisplace, forceVal, testNum, studyID, testName, dispLabel,path)

        elif testType in ('SH', 'sh'):
            planeID = input('Shear plane: ')
            typeFlag = 2
            for currentdir in subdir:
                print(currentdir)               
                topimg = os.listdir(currentdir)[0]
                px2mm = size_calibration.sizecalib(currentdir, topimg, typeFlag)
                distVal, forceVal, deltaVal = thetadep(currentdir, px2mm, typeFlag)
                subDist.append(distVal)
                subForce.append(forceVal)
                sDisplace.append(deltaVal)
            studyID = 'Shear Test (trials: {0}), {shear}-plane'.format(testNum, shear = planeID)
            testName = testName + '-{shear}'.format(shear = planeID)
            distLabel = 'Distance (mm)'
            dispLabel = 'Displacement $\Delta d$ (mm)'
            statanalysis(subDist, subForce, testNum, studyID, testName, distLabel, path, typeFlag)
            #I don't know why the displacement adds a factor of ten to a simple subtraction, dear self, wtf
            #statdisplacement(sDisplace, forceVal, testNum, studyID, testName, dispLabel, path)
            #for when statdisplacement isn't working normally
            statdisplacementalt(sDisplace, subDist, forceVal, testNum, studyID, testName, dispLabel,path,typeFlag)

        else:
            typeFlag = 3
            for currentdir in subdir:
                print(currentdir)
                topimg = os.listdir(currentdir)[0]
                px2mm = size_calibration.sizecalib(currentdir, topimg, typeFlag)
                distVal, forceVal, delta = thetadep(currentdir, px2mm, typeFlag)
                subDist.append(distVal)
                subForce.append(forceVal)
                sDisplace.append(delta)
            studyID = 'Bending Test (trials: {0})'.format(testNum)
            distLabel = (r'Angle ${\theta}$ (deg)')
            dispLabel = (r'${\Delta} {\theta} $ (deg)')
            statanalysis(subDist, subForce, testNum, studyID, testName, distLabel, path, typeFlag)
            statdisplacement(sDisplace, forceVal, testNum, studyID, testName, dispLabel, path)
        
        logger.info('Compilation run end')
        print('Processed {} files ({} bytes). Runtime: {} seconds.'.format(
            fileCycle[0] , fileCycle[1], (time.time() - startTime)))

    else:

        while True:
            if not os.path.isdir(path):
                print('Not a valid subdirectory.')
                logger.warn('Input: Nonexistant subdir')
                continue
            else:
                currentdir = path
                startTime = time.time()
                topimg = os.listdir(currentdir)[0]
                

                if testType in ('COM', 'com'):
                    typeFlag = 1
                    px2mm = size_calibration.sizecalib(currentdir, topimg, typeFlag)
                    distVal, forceVal, deltaVal = compression(path, px2mm, typeFlag)
                    studyID = 'Compression Test'
                elif testType in ('SH', 'sh'):
                    planeID = input('Shear plane: ')
                    typeFlag = 2
                    px2mm = size_calibration.sizecalib(currentdir, topimg, typeFlag)
                    distVal, forceVal, deltaVal = thetadep(path, px2mm, typeFlag)
                    studyID = 'Shear Test'
                    testName = testname + '_{}'.format(planeID)
                else:
                    typeFlag = 3
                    px2mm = size_calibration.sizecalib(currentdir, topimg, typeFlag)
                    distVal, forceVal, deltaVal = thetadep(path, px2mm, typeFlag)
                    studyID = 'Bending Test'
        
        singresplot(forceVal, distVal, FORCE_LABEL, distLabel, studyID, testName, path)
        print('Processed {} files ({} bytes). Runtime: {} seconds.'.format(
            fileCycle[0] , fileCycle[1], (time.time() - startTime)))  

main()