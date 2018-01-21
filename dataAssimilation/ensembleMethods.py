import numpy as np
import copy


def __assimilationCost(optimArg, observationSet, observationOperator, \
                     invObservationCovariance, invStateCovariance,
                     invInputCovariance, priorState, priorInput, rom):

    initialCondition = np.reshape(optimArg[:rom.numStates], [rom.numStates, 1])
    inputVector = np.reshape(optimArg[rom.numStates:], [rom.numControls, 1])

    cost = 0.0
    simulation = rom.evaluateROM(initialCondition, inputVector)

    #TODO: How can we accept non-linear operators
    innovation = observationSet - observationOperator.dot(simulation)

    #TODO: Currently we need an observation at every time
    for itime in range(1,innovation.shape[1],1):
        cost = cost + 0.5*innovation[:,[itime]].transpose().dot(
            invObservationCovariance[:,:,itime]).dot(innovation[:,[itime]]).squeeze()

    dx = priorState - initialCondition
    cost = cost + \
           0.5*(dx).transpose().dot(invStateCovariance).dot(dx).squeeze()

    dx = priorInput - inputVector
    cost = cost + \
           0.5*(dx).transpose().dot(invInputCovariance).dot(dx).squeeze()

    return cost

def enMOR(observationSet, observationOperator, observationCov, priorInput,
          priorState, inputCov, initialStateCov, rom, modelEnsemble, ensembleInputs):

    from scipy import optimize
    from numpy.linalg import inv
    from dataAssimilation.reducedLinearModel_scikit import reducedOrderModel as romclass


    assert type(inputCov) is np.ndarray, \
        "inputVectorEnsemble is not a np.array: %r" % inputCov
    assert type(initialStateCov) is np.ndarray, \
        "inputVectorEnsemble is not a np.array: %r" % initialStateCov
    assert type(observationCov) is np.ndarray, \
        "inputVectorEnsemble is not a np.array: %r" % observationCov
    assert type(observationSet) is np.ndarray, \
        "inputVectorEnsemble is not a np.array: %r" % observationSet
    assert type(observationOperator) is np.ndarray, \
        "inputVectorEnsemble is not a np.array: %r" % observationOperator
    assert type(priorInput) is np.ndarray, \
        "inputVectorEnsemble is not a np.array: %r" % priorInput
    assert type(priorState) is np.ndarray, \
        "inputVectorEnsemble is not a np.array: %r" % priorState

    print('')
    print('Starting data assimilation')

    invInputCovariance = inv(inputCov)
    invStateCovariance = inv(initialStateCov)

    invObsCovariance = np.ndarray([rom.numStates, rom.numStates,
                                   observationSet.shape[1]])

    for iobs in range(0, observationCov.shape[2]):
        invObsCovariance[:, :, iobs] = inv(observationCov[:, :, iobs])

    # Project initial state to sub-space and reformat for optimization algorithm
    optimArg = np.concatenate((priorState, priorInput), axis=0)

    # TODO: The use of the observation operator is still not possible.
    observationOperator = np.eye(rom.numStates, rom.numStates)

    rom_temp = copy.copy(rom)
    print('\n\n\nData assimilation outer loop: 0')
    for i in range(0,1):

        print('\nData assimilation outer process:')
        print(' - Initial Cost: %r' % __assimilationCost(optimArg,
                                                         observationSet,
                                                         observationOperator,
                                                         invObsCovariance,
                                                         invStateCovariance,
                                                         invInputCovariance,
                                                         priorState,
                                                         priorInput,
                                                         rom_temp))

        [optimArg, cost, spec] = optimize.fmin_l_bfgs_b( __assimilationCost,
                                                         optimArg,
                                                         approx_grad=True,
                                                         args=(observationSet,
                                                              observationOperator,
                                                              invObsCovariance,
                                                              invStateCovariance,
                                                              invInputCovariance,
                                                              priorState,
                                                              priorInput,
                                                              rom_temp),
                                                         iprint=0,
                                                         disp=0,
                                                         epsilon=1e-2)

        print(' - Final cost: ' + str(cost))
        if spec['warnflag'] == 0:
            print(' - Optimization converged')
        elif spec['warnflag'] == 1:
            print(
                ' - Optimization stopped due to too many function evaluations or too many iterations')
        elif spec['warnflag'] == 2:
            print(' - Optimization did not converged due to unknown reason')
        print(' - ' + spec['task'].decode(encoding='UTF-8'))

        '''
        inputsEns = np.concatenate((modelEnsemble[:,0,:], ensembleInputs), axis=0)

        # How far away are we from the prior information:
        d = np.concatenate( (rom_temp.referenceSimulation[:,[0]],rom_temp.referenceInput),
                           axis=0) - optimArg
        l2norm = np.sum(d*d)

        ind = None
        for iEns in range(0,inputsEns.shape[1]):

            # How far away are we from this ensemble member:
            d = inputsEns[:, [iEns]] - optimArg
            l2norm_temp = np.sum(d*d)

            if l2norm_temp < l2norm:
                l2norm = l2norm_temp
                ind = iEns

        if ind is None:
            break

        print(' - Nearest neighbour index: %r' % ind)

        newReferenceSimulation = modelEnsemble[:,:,ind].copy()
        newReferenceInput = ensembleInputs[:,[ind]].copy()

        modelEnsemble[:,:,ind] = rom_temp.referenceSimulation.copy()
        ensembleInputs[:,[ind]] = rom_temp.referenceInput.copy()

        print('\n\n\nData assimilation outer loop: %i' % (i+1))
        rom_temp = romclass('Temporay model', 'Temporary model', newReferenceInput,
                            newReferenceSimulation, modelEnsemble, ensembleInputs,
                            rom_temp.energyType, rom_temp.energy)

        optimArg = np.concatenate((newReferenceSimulation[:,[0]], newReferenceInput),
                                  axis=0)
        '''


    # initialState = np.dot(self.projectionMatrix, np.reshape(optimArg[:self.numReducedStates], [self.numReducedStates, 1])) + self.referenceSimulation[:, [0]]
    # inputVector = np.reshape(optimArg[self.numReducedStates:], [self.numControls, 1]) + self.referenceInput

    return [np.reshape(optimArg[:rom.numStates], [rom.numStates, 1]),
            np.reshape(optimArg[rom.numStates:], [rom.numControls, 1]),
            cost]