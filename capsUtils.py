import csv
import math
import pandas
import numpy as np
from matplotlib import pyplot as plt

def plotLog(fileNM, showPlot=True):
    #Read CSV File
    dataCSV = pandas.read_csv(fileNM)
    #Initialize Plot
    figPlot = plt.figure(figsize=(4,6))
    #Initialize Sub Plot
    figPlot.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    #Add Sub Plot
    figPlot.add_subplot(211)
    #Add Data of Training Loss
    for keyVal in dataCSV.keys():
        if keyVal.find('loss') >= 0 and not keyVal.find('val') >= 0:  
            plt.plot(dataCSV['epoch'].values, dataCSV[keyVal].values, label=keyVal)
    #show Plot Legend
    plt.legend()
    #Set Plot Title
    plt.title('Training loss')
    #Add Sub Plot
    figPlot.add_subplot(212)
    #Add Data of Accuracy
    for keyVal in dataCSV.keys():
        if keyVal.find('acc') >= 0:  
            plt.plot(dataCSV['epoch'].values, dataCSV[keyVal].values, label=keyVal)
    #show Plot Legend
    plt.legend()
    #Set Plot Title
    plt.title('Training and validation accuracy')
    # figPlot.savefig('result/fig.png')
    # Show Plot 
    if showPlot:
        plt.show()


def combineIMG(createdIMG, heightIMG=None, widthIMG=None):
    #No. Of Images
    numIMG = createdIMG.shape[0]
    #Width and Height given OR Not
    if widthIMG is None and heightIMG is None:
        widthIMG = int(math.sqrt(numIMG))
        heightIMG = int(math.ceil(float(numIMG)/widthIMG))
    elif widthIMG is not None and heightIMG is None:  
        heightIMG = int(math.ceil(float(numIMG)/widthIMG))
    elif heightIMG is not None and widthIMG is None:  
        widthIMG = int(math.ceil(float(numIMG)/heightIMG))

    shape = createdIMG.shape[1:3]
    imageF = np.zeros((heightIMG*shape[0], widthIMG*shape[1]), dtype=createdIMG.dtype)
    for index, img in enumerate(createdIMG):
        i = int(index/widthIMG)
        j = index % widthIMG
        imageF[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return imageF

if __name__=="__main__":
    plotLog('result/log.csv')



