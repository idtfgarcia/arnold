__author__ = 'Ivan'

import inspect as insp

import numpy as np
from matplotlib import pyplot as plt

import graphUtils as gutil
from dataAssimilation import ensembleMethods as enm
from dataAssimilation.reducedLinearModel_scikit import reducedOrderModel as romclass
from ioUtils import ioUtils

frameinfo = insp.getframeinfo(insp.currentframe())

def cls(): print("\n" * 100)

sizeEnsemble = 200
baseFolder = '.\\..\\..\\Writings\\paper_enmor\\matlab\\raw_data\\'
plt.ioff()

# ========================================================================================
# - Upload the model runs
# ----------------------------------------------------------------------------------------

# Process the model ensemble data
modelEnsemble = ioUtils.csv2array(baseFolder + '20141101_results.txt', delimiter=' ')  #Load data
modelEnsemble = np.array(modelEnsemble)
xShoreCoord   = modelEnsemble[:, 0].reshape([20, 11, int(modelEnsemble.shape[0]/20/11)], order = 'F').copy()
xShoreCoord   = np.squeeze(xShoreCoord[:,0,0])
modelEnsemble = 1000*modelEnsemble[:, 1].reshape([20, 11, int(modelEnsemble.shape[0]/20/11)], order = 'F')

modelEnsemble = np.delete(np.delete(modelEnsemble, 0, 0), 18, 0)
xShoreCoord   = np.delete(np.delete(xShoreCoord, 0, 0), 18, 0)


# process the input data
ensembleInputs = ioUtils.csv2array(baseFolder + '20141101_parameters.txt', delimiter='\n')
ensembleInputs = np.array(ensembleInputs)
ensembleInputs = ensembleInputs[:, 0].reshape([20, 11, int(ensembleInputs.shape[0] / 20 / 11)], order='F')

ensembleInputs = np.delete(ensembleInputs, range(9, 20, 1), 0)
ensembleInputs = np.delete(ensembleInputs, range(1, 20, 1), 1)
ensembleInputs = np.squeeze(ensembleInputs, axis=1)
# ----------------------------------------------------------------------------------------





# ========================================================================================
# - Get rid of faulty model runs
# ----------------------------------------------------------------------------------------
modelErrors = np.unique(np.array(np.where(np.isnan(modelEnsemble))[2], dtype=int))
modelEnsemble = np.delete(modelEnsemble, modelErrors, 2)
ensembleInputs = np.delete(ensembleInputs, modelErrors, 1)
modelErrors = [ 308,  310,  355,  416,  509,  535,  622,  689,  725,  754,  755, 756,
                774,  798,  802,  805,  845,  850,  856,  912,  984,  985, 995, 1091]
modelEnsemble = np.delete(modelEnsemble, modelErrors, 2)
ensembleInputs = np.delete(ensembleInputs, modelErrors, 1)
# ----------------------------------------------------------------------------------------



# ========================================================================================
# - Get reference and observations
# ----------------------------------------------------------------------------------------

# Let's forget about the first time-step, assume that it is a system start process.
modelEnsemble = modelEnsemble[:,1:,:]

# Choose a reference run that will be assumed to be the initial guess.
referenceSimulation = modelEnsemble[:, :, 0].copy()
referenceInput      = ensembleInputs[:, [0]].copy()
modelEnsemble       = np.delete(modelEnsemble, 0, 2)
ensembleInputs      = np.delete(ensembleInputs, 0, 1)

# Choose a set of observations: ** Notice
observation_indices = range(25,1000,25)
observationSet      = modelEnsemble[:, :, observation_indices].copy()
observationInput    = ensembleInputs[:, observation_indices].copy()
observationOperator = np.eye(referenceSimulation.shape[0])
# ----------------------------------------------------------------------------------------



# ========================================================================================
# - Select one observation model and define its error properties
# ----------------------------------------------------------------------------------------

# Which observation set do you want to use?
iset = 2

initialStateCov = np.diag(np.var(modelEnsemble[:, 0, :], axis=1))
inputCov        = np.diag(np.var(ensembleInputs, axis=1))

numObsX = observationSet.shape[0]
numObsT = observationSet.shape[1]
numObs = observationSet.shape[2]

observationCov = np.ndarray([numObsX, numObsX, numObsT])
obsStd = np.ones(observationSet[:, :, iset].shape)*0.05

for it in range(0, numObsT):
    observationCov[:,:,it] = np.diag(obsStd[:,it]*obsStd[:,it])

for iset in range(0, numObs):
    observationSet[:,:,iset] = observationSet[:,:,iset] + obsStd * np.random.randn(
        numObsX, numObsT )
#-----------------------------------------------------------------------------------------




# ========================================================================================
# - Build the reduced order model
# ----------------------------------------------------------------------------------------
import dataAssimilation.ensembleUtils as eu


# Get the closest ensemble members to the reference simulation
reference = np.concatenate((referenceSimulation[:,[0]],referenceInput), axis=0)
ensemble  = np.concatenate((modelEnsemble[:,0,:],ensembleInputs), axis=0)
ids = eu.knn4vector(reference, ensemble, sizeEnsemble)


# Keep only the number of ensemble members desired
other_members = np.delete(modelEnsemble, ids.id.astype(int), 2)
other_inputs = np.delete(ensembleInputs, ids.id.astype(int), 1)

modelEnsemble = modelEnsemble[:,:,ids.id.astype(int)]
ensembleInputs = ensembleInputs[:,ids.id.astype(int)]

rom = romclass('Profile model', 'Idealized model', referenceInput, referenceSimulation,
               modelEnsemble, ensembleInputs, "percent", 0.99)
# ----------------------------------------------------------------------------------------



# ========================================================================================
# Some graphs to describe the performance of the ROM with respect to training ensemble
# and validation ensemble.

e = ensembleInputs.mean(axis=1)





error_performance_plots = False

if error_performance_plots:
    f1 = gutil.romPerformacePcolor(modelEnsemble, ensembleInputs, rom)
    f1.savefig('C:\\Users\\Ivan\\Dropbox\\Writings\\paper_enmor\\manuscript\\figures'
               '\\performance_rom_pcolor.pdf')

    f1 = gutil.romPerformaceMAE(modelEnsemble, ensembleInputs, other_members, other_inputs,
                                rom)
    f1.savefig('C:\\Users\\Ivan\\Dropbox\\Writings\\paper_enmor\\manuscript\\figures'
               '\\performance_rom_mae.pdf')

    f1,f2 = gutil.romErrorAnalysis(rom, modelEnsemble, ensembleInputs, xShoreCoord)

    f1.savefig('C:\\Users\\Ivan\\Dropbox\\Writings\\paper_enmor\\manuscript\\figures'
               '\\performance_rom_profile_errors.pdf')

    f2.savefig('C:\\Users\\Ivan\\Dropbox\\Writings\\paper_enmor\\manuscript\\figures'
               '\\performance_rom_histograms_per_time.pdf')

    plt.show()
    plt.close()
# ----------------------------------------------------------------------------------------




# ========================================================================================
# Assimilate observations
assimilate = False

if assimilate:
    [analyzedState, analyzedInput, analysisCost] = enm.enMOR(observationSet[:,:,iset],
                                                         observationOperator,
                                                         observationCov,
                                                         referenceInput,
                                                         referenceSimulation[:,[0]],
                                                         inputCov,
                                                         initialStateCov,
                                                         rom,
                                                         modelEnsemble,
                                                         ensembleInputs)


a = np.concatenate((analyzedState,analyzedInput),axis=0)
e = np.concatenate((modelEnsemble[:,0,:], ensembleInputs), axis=0)
d = e - np.repeat(a,e.shape[1],axis=1)
n = [np.linalg.norm(d[:,[i]]) for i in range(0,d.shape[1])]
ind = np.argmin(n)
# ----------------------------------------------------------------------------------------




# ========================================================================================
# Some graphs to describe the performance of the data assimilation procedure.
dataAssimilationGraphs = False

if dataAssimilationGraphs:

    gutil.graphicAnalysis(modelEnsemble, observationSet[:,:,iset],
                      observationInput[:,[iset]], referenceSimulation, referenceInput)

    gutil.graphicResultAnalysis(analyzedState, analyzedInput, observationSet[:,:,iset],
                            observationInput[:, [iset]], referenceSimulation,
                            referenceInput, rom)
#-----------------------------------------------------------------------------------------