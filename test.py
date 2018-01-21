
def cls(): print("\n" * 100)

import numpy as np

numSteps = 3
numStates = 3
numInputs = 5
numEnsMembers = 300

modelEnsemble = np.zeros([numStates, numSteps + 1, numEnsMembers])
inputVector = np.zeros([numInputs, numEnsMembers])

A = np.random.rand(numStates, numStates, numSteps) / 100
B = np.random.rand(numStates, numInputs, numSteps) / 100

# ================================================
# Run the linear model to produce the ensemble
# ================================================
for iens in range(0, numEnsMembers, 1):
    modelEnsemble[:, [0], iens] = np.random.rand(numStates, 1)
    inputVector[:, [iens]] = np.random.rand(numInputs, 1)
    for iStep in range(0, numSteps, 1):
        modelEnsemble[:, iStep + 1, iens] = A[:, :, iStep].dot(
            modelEnsemble[:, iStep, iens]) + \
                                            B[:, :, iStep].dot(inputVector[:, iens])

referenceSimulation = modelEnsemble[:, :, 0]
referenceInput = inputVector[:, [0]]

modelEnsemble = np.delete(modelEnsemble, 0, 2)
inputVector = np.delete(inputVector, 0, 1)

from sklearn import linear_model
regr = linear_model.LinearRegression()

regr.fit(modelEnsemble[:,0,:], modelEnsemble[:,1,:])

# The coefficients
print('Coefficients: \n', regr.coef_)