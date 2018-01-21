def cls(): print("\n" * 100)

def test_linearReg_sk():

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

def test_components_estimation():

    import numpy as np
    from dataAssimilation.reducedLinearModel import reducedLinearModel as romClass
    from matplotlib import pyplot as plt

    numSteps = 3
    numStates = 3
    numInputs = 5
    numEnsMembers = 300

    modelEnsemble = np.zeros([ numStates,numSteps+1,numEnsMembers ])
    inputVector = np.zeros([numInputs,numEnsMembers])

    A = np.random.rand(numStates, numStates, numSteps)/100
    B = np.random.rand(numStates, numInputs, numSteps)/100

    # ================================================
    # Run the linear model to produce the ensemble
    # ================================================
    for iens in range(0, numEnsMembers, 1):

        modelEnsemble[:,[0],iens] = np.random.rand(numStates,1)
        inputVector[:,[iens]] = np.random.rand(numInputs,1)
        for iStep in range(0, numSteps, 1):
            modelEnsemble[:, iStep + 1, iens] = A[:, :, iStep].dot(modelEnsemble[:, iStep, iens]) + \
                                                B[:, :, iStep].dot(inputVector[:, iens])

    referenceSimulation = modelEnsemble[:, :, 0]
    referenceInput = inputVector[:, [0]]

    modelEnsemble = np.delete(modelEnsemble, 0, 2)
    inputVector = np.delete(inputVector, 0, 1)

    rom = romClass('Profile model', 'Idealized model', inputVector, modelEnsemble, referenceInput, \
                   referenceSimulation, "percent", 1.0, maxiter=30, factr=1e-6,
                   pgtol=1e-9, verbose=1)

    costA = np.ndarray([numSteps])
    costB = np.ndarray([numSteps])
    for i in range(0,numSteps):
        PA = rom.projectionMatrix.transpose().dot(A[:, :, i]).dot(rom.projectionMatrix)
        costA[i] =  np.sum((PA - rom.dynamics[:,:,i]) * (PA - rom.dynamics[:,:,i]))

        PB = rom.projectionMatrix.transpose().dot(B[:, :, i])
        costB[i] = np.sum((PB - rom.sensitivities[:,:,i]) *
                          (PB - rom.sensitivities[:,:,i]))

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 2, 1)
    y, x = np.mgrid[slice(0, A.shape[0], 1), slice(0, A.shape[1] + 1, 1)]
    z = rom.projectionMatrix.transpose().dot(A[:,:,0]).dot(rom.projectionMatrix)
    im = ax.pcolor(x, y, z, cmap='RdBu', vmin = -0.01, vmax = 0.01)
    ax.set_title('A')
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(1, 2, 2)
    y, x = np.mgrid[slice(0, A.shape[0], 1), slice(0, A.shape[1] + 1, 1)]
    z = (rom.dynamics[:, :, 0])
    im2 = ax.pcolor(x, y, z, cmap='RdBu', vmin = -0.01, vmax = 0.01)
    ax.set_title('Dynamics')
    plt.colorbar(im2, ax=ax)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 2, 1)
    y, x = np.mgrid[slice(0, B.shape[0], 1), slice(0, B.shape[1] + 1, 1)]
    z = rom.projectionMatrix.transpose().dot(B[:, :, 0])
    im = ax.pcolor(x, y, z, cmap='RdBu', vmin = -0.01, vmax = 0.01)
    ax.set_title('B')
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(1, 2, 2)
    y, x = np.mgrid[slice(0, B.shape[0], 1), slice(0, B.shape[1] + 1, 1)]
    z = (rom.sensitivities[:, :, 0])
    im2 = ax.pcolor(x, y, z, cmap='RdBu', vmin = -0.01, vmax = 0.01)
    ax.set_title('Sensitivities')
    plt.colorbar(im2, ax=ax)

    return A, B, rom, costA, costB

'''       finalCost = 0
    for iens in range(0, 50, 1):
        finalCost = finalCost + np.sum(0.5 * (
        rom.evaluateROM(modelEnsemble[:, [0], iens], inputVector[:, [iens]]) - modelEnsemble[:, :, iens]) * (
                                       rom.evaluateROM(modelEnsemble[:, [0], iens],
                                                       inputVector[:, [iens]]) - modelEnsemble[:, :, iens]))
'''

#self.assertEqual(fun(3), 4)