import SimpleITK as sitk
import numpy as np
import pandas as pd
import os

import ntpath

# To get the data:
# wget https://www.dropbox.com/sh/mtip9dx6zt9nb3z/AAAs2wbJxbNM44-uafZyoMVca/subset5.zip
# The files are 7-zipped. Regular linux unzip won't work to uncompress them. Use 7za instead.
# 7za e subset5.zip

DATA_DIR = "/Volumes/data/tonyr/dicom/LUNA16/"
cand_path = 'CSVFILES/candidates_with_annotations.csv'

def extractCandidates(img_file):
    
    # Get the name of the file
    subjectName = ntpath.splitext(ntpath.basename(img_file))[0]  # Strip off the .mhd extension
    
    # Read the list of candidate ROI
    dfCandidates = pd.read_csv(DATA_DIR+cand_path)
    
    numCandidates = dfCandidates[dfCandidates['seriesuid']==subjectName].shape[0]
    print('There are {} candidate nodules in this file.'.format(numCandidates))
    
    numNonNodules = sum(dfCandidates[dfCandidates['seriesuid']==subjectName]['class'] == 0)
    numNodules = sum(dfCandidates[dfCandidates['seriesuid']==subjectName]['class'] == 1)
    #print('{} are true nodules (class 1) and {} are non-nodules (class 0)'.format(numNodules, numNonNodules))
    
    # Read if the candidate ROI is a nodule (1) or non-nodule (0)
    candidateValues = dfCandidates[dfCandidates['seriesuid']==subjectName]['class'].values
    
    # Get the world coordinates (mm) of the candidate ROI center
    worldCoords = dfCandidates[dfCandidates['seriesuid']==subjectName][['coordX', 'coordY', 'coordZ']].values
    
    # Use SimpleITK to read the mhd image
    itkimage = sitk.ReadImage(img_file)
    
    # Get the real world origin (mm) for this image
    originMatrix = np.tile(itkimage.GetOrigin(), (numCandidates,1))  # Real world origin for this image (0,0)
    
    # Subtract the real world origin and scale by the real world (mm per pixel)
    # This should give us the X,Y,Z coordinates for the candidates
    candidatesPixels = (np.round(np.absolute(worldCoords - originMatrix) / itkimage.GetSpacing())).astype(int)
    
    # Replace the missing diameters with the 50th percentile diameter 
    
    candidateDiameter = dfCandidates['diameter_mm'].fillna(dfCandidates['diameter_mm'].quantile(0.5)).values / itkimage.GetSpacing()[1]
     
    candidatePatches = []
    
    imgAll = sitk.GetArrayFromImage(itkimage) # Read the image volume
    
    for candNum in range(numCandidates):
        
        #print('Extracting candidate patch #{}'.format(candNum))
        candidateVoxel = candidatesPixels[candNum,:]
        xpos = int(candidateVoxel[0])
        ypos = int(candidateVoxel[1])
        zpos = int(candidateVoxel[2])
        
        # Need to handle the candidates where the window would extend beyond the image boundaries
        windowSize = 32
        x_lower = np.max([0, xpos - windowSize])  # Return 0 if position off image
        x_upper = np.min([xpos + windowSize, itkimage.GetWidth()]) # Return  maxWidth if position off image
        
        y_lower = np.max([0, ypos - windowSize])  # Return 0 if position off image
        y_upper = np.min([ypos + windowSize, itkimage.GetHeight()]) # Return  maxHeight if position off image
         
        # SimpleITK is x,y,z. Numpy is z, y, x.
        imgPatch = imgAll[zpos, y_lower:y_upper, x_lower:x_upper]
        
        # Normalize to the Hounsfield units
        # TODO: I don't think we should normalize into Housefield units
        imgPatchNorm = imgPatch #normalizePlanes(imgPatch)
        
        candidatePatches.append(imgPatchNorm)  # Append the candidate image patches to a python list

    return candidatePatches, candidateValues, candidateDiameter



from scipy.misc import toimage

# We need to save the array as an image.
# This is the easiest way. Matplotlib seems to like adding a white border that is hard to kill.

def SavePatches(img_file, patchesArray, valuesArray):
    
    saveDir = ntpath.dirname(img_file) + '/patches'

    try:
        os.stat(saveDir)
    except:
        os.mkdir(saveDir) 

    subjectName = ntpath.splitext(ntpath.basename(img_file))[0]

    print('Saving image patches for file {}.'.format(subjectName))
    for i in range(len(valuesArray)):

        print('\r{} of {}'.format(i+1, len(valuesArray))),
        im = toimage(patchesArray[i])
        im.save(saveDir + '/{}_{}_{}.png'.format(subjectName, i, valuesArray[i]))

    print('Finished {}'.format(subjectName))

i = 0

for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if (file.endswith('.mhd')) & ('__MACOSX' not in root):  # Don't get the Macintosh directory
            
            img_file = os.path.join(root, file)
            patchesArray, valuesArray, noduleDiameter = extractCandidates(img_file)   
             
            SavePatches(img_file, patchesArray, valuesArray)
            
           



