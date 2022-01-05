import numpy as np
import pandas as pd

def load_data(trainFile, testFile):
    trainData = pd.read_csv(trainFile)
    testData = pd.read_csv(testFile)
    
    return [trainData, testData]

def save_to_file(result, testData, saveFile):
    finalRes = pd.DataFrame({'Id': testData['Id'], 'result':result})
    finalRes.to_csv(saveFile, index = False)
