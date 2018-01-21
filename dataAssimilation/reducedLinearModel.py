__author__ = 'Ivan'

import numpy as np

class reducedLinearModel(object):

    def __init__(self, name, metadata, inputVectors, modelEnsemble, referenceInput,
                 referenceSimulation, energyType, energy, maxiter = 500, factr = 1e7,
                 pgtol = 1e-5, verbose = 0):

        assert type(name) is str, "name is not a string: %r" % name
        assert type(metadata) is str, "metadata is not a string: %r" % metadata
        assert type(modelEnsemble) is np.ndarray, "modelEnsemble is not a np.ndarray: " \
                                                  "%r" % modelEnsemble
        assert type(inputVectors) is np.ndarray, "inputVectors is not a np.ndarray: %r" \
                                                 % inputVectors
        assert type(referenceSimulation) is np.ndarray, "referenceSimulation is not a " \
                                                        "np.ndarray: %r" % modelEnsemble
        assert type(referenceInput) is np.ndarray, "referenceInput is not a " \
                                                   "np.ndarray: %r" % inputVectors


        # Define relevant members
        self._metadata = metadata
        self._name = name

        self.referenceSimulation = referenceSimulation.copy()
        self.referenceInput = referenceInput.copy()

        self.sizeEns = modelEnsemble.shape[2]
        self.dimDt = modelEnsemble.shape[1] - 1

        self.numStates = referenceSimulation.shape[0]
        self.numControls = referenceInput.shape[0]

        self.modelEnsemble = modelEnsemble
        self.inputVectors = inputVectors

        self.energyType = energyType
        self.energy = energy

        self.maxiter = maxiter
        self.factr = factr
        self.pgtol = pgtol
        self.verbose = verbose

        self.cost, self.projectionMatrix, self.spectralEnergy, self.romSingularValues, \
        self.dynamics, self.sensitivities, self.dimr = \
            self.estimateComponents( modelEnsemble, inputVectors, referenceSimulation,
                                     referenceInput, energyType, energy, maxiter,
                                     factr, pgtol, verbose )

    def estimateComponents(self, modelEnsemble, inputVectors, referenceSimulation,
                           referenceInput, energyType, energy, maxiter=500, factr=1e7,
                           pgtol=1e-5, verbose=0):
        '''
        :rtype : numpy.ndarray
        :param inputVectorEnsemble: 2D np.array  numParameters by numEnsembleMem
        :param initialStateEnsemble: 2D np.array numStateElem by numEnsembleMem
        :param snapshots: 3D np.array with numStateElem by numTimes by numEnsembleMem
        :return:
        '''

        from scipy import optimize
        import time

        # Process the data into the right format. The analysis requires
        # snapshot matrix rather than an ensemble of simulations.
        [snapshots, snapshotInputs] = \
            self.getSnapshotMatrix(modelEnsemble, inputVectors, referenceSimulation,
                                   referenceInput)


        # Model order reduction:
        # Compute the sub-basis of the space spanned by the ensemble
        [snapshotRedMat, projectionMatrix, spectralEnergy, romSingularValues] = \
            self.modelOrderReduction(snapshots, energyType, energy)

        dimr = snapshotRedMat.shape[0]

        print(' - %r dimensional model.' % dimr)
        print(' - %r controls detected.' % self.numControls)
        print(' - %r required components.' % self.dimDt)

        weights = np.ndarray([dimr,dimr,self.dimDt+1])
        for t in range(0, self.dimDt+1, 1):
            weights[:,:,t] = np.eye(dimr, dimr)


        print('')
        print('Linear model identification process:')

        print(' - Initializing components.')

        dynamics = np.zeros([dimr, dimr, self.dimDt])
        sensitivities = np.zeros([dimr, self.numControls, self.dimDt])

        for iStep in range(0, self.dimDt, 1):
            C = np.concatenate((snapshotRedMat[:, iStep, :], snapshotInputs),
                               axis=0)
            rhs = C.dot(C.transpose())
            lhs = snapshotRedMat[:, iStep + 1, :].dot(C.transpose())
            system = lhs.dot(np.linalg.pinv(rhs))

            dynamics[:, :, iStep] = system[:, :dimr].copy()
            sensitivities[:, :, iStep] = system[:, dimr:].copy()

        optimArg = np.concatenate(
            (np.reshape(dynamics, [dimr * dimr * self.dimDt, 1]),
             np.reshape(sensitivities, [dimr * self.numControls * self.dimDt, 1])),
            axis=0)

        '''optimArg = np.zeros([dimr * dimr * self.dimDt
                             + dimr * self.numControls * self.dimDt, 1])'''

        cost = self.componentEstimationCost(optimArg, snapshotRedMat, snapshotInputs,
                                            weights, dimr)
        print(' - Initial Cost: %r ' % cost)
        print(' - Minimization started (This process may take long time...)')
        tic = time.time()

        '''
         - factr (float): The iteration stops when
                (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps,
            where eps is the
            machine precision, which is automatically generated by the code. Typical
            values for factr are:1e12 for low accuracy; 1e7 for moderate accuracy; 10.0
            for extremely high accuracy.

         -  pgtol (float): The iteration will stop when
                max{|proj g_i | i = 1, ..., n} <= pgtol
            where pg_i is the i-th component of the projected gradient.

         - Callback (callable, optional): Called after each iteration, as callback(xk),
            where xk is the current parameter vector
        '''
        [optimArg, cost, spec] = \
            optimize.fmin_l_bfgs_b(self.componentEstimationCost, optimArg,
                                   fprime=self.componentEstimationCostDerivative,
                                   args=(snapshotRedMat, snapshotInputs, weights, dimr),
                                   disp=verbose, maxiter=maxiter, factr=factr,
                                   pgtol=pgtol, iprint=-1)

        print(' - Final cost: ' + str(cost))
        print(' - Time elapsed: ' + str(time.time() - tic))
        if spec['warnflag'] == 0:
            print(' - Optimization converged')
        elif spec['warnflag'] == 1:
            print(
                ' - Optimization stopped due to too many function evaluations or too many iterations')
        elif spec['warnflag'] == 2:
            print(' - Optimization did not converged due to unknown reason')
        print(' - ' + spec['task'].decode(encoding='UTF-8'))

        dynamics = np.reshape(optimArg[:(dimr * dimr * self.dimDt)],
                                   [dimr, dimr, self.dimDt])
        sensitivities = np.reshape(optimArg[(dimr * dimr * self.dimDt):],
                                        [dimr, self.numControls, self.dimDt])

        # print('')
        # print(' - Building adjoint model.')
        # self.buildAdjoint()
        self.romCost = cost

        return cost, projectionMatrix, spectralEnergy, \
               romSingularValues, dynamics, sensitivities, dimr


    def getSnapshotMatrix(self, modelEnsemble, inputVectors, referenceSimulation,
                          referenceInput):
        '''
        Build the snapshots with the reference run previously chosen.
        '''
        import numpy.matlib as npmatlib

        assert type(modelEnsemble) is np.ndarray, \
            "modelEnsemble is not a np.ndarray: %r" % type(modelEnsemble)
        assert type(inputVectors)  is np.ndarray,\
            "modelEnsemble is not a np.ndarray: %r" % type(inputVectors)

        snapshots = np.zeros(modelEnsemble.shape)
        snapshotInputs = np.zeros(inputVectors.shape)

        for iEns in range(0,modelEnsemble.shape[2],1):
            snapshots[:,:,iEns] = modelEnsemble[:,:,iEns] - referenceSimulation
            snapshotInputs[:,[iEns]] =  inputVectors[:,[iEns]] - referenceInput

        return snapshots, snapshotInputs

    def modelOrderReduction(self, snapshots, energyType, energy, normalize=True):
        '''
        Computes the proper orthogonal decomposition of the ensembe.

        :param ensemble: np.dim(ensemble) = 3
        :param energy: 0 < energy < 1
        :return: sub-basis of the space spanned by ensemble.
        '''

        assert energyType == "percent" or energyType == "dimensions", \
            "Invalid energType value. Allowed values: ['percent','dimensions']"
        assert np.ndim(snapshots) == 3, \
            "Member snapshots should be a 3 dimensional np.ndarray"
        if energyType == 'percent':
            assert energy <= 1.0 and energy > 0.0, \
                "Energy should be a value in (0,1]: %r"
        else:
            assert type(energy) is int, \
                "EnergyType = 'Dimensions'. Dimensions should be an integer: %r" % energy

        print('')
        print('Spectral analysis started:')
        print(' - This may take a while...')

        temp = snapshots.copy()

        dims = temp.shape
        temp = temp.reshape([dims[0], dims[1] * dims[2]], order='F')

        if normalize:
            norms = np.sqrt(np.sum(temp * temp, axis=0))
            for icol in range(0, temp.shape[1], 1):
                temp[:, icol] = temp[:, icol] / norms[icol]

        [projectionMatrix, s, v] = np.linalg.svd(temp)

        if energyType == 'percent':

            if s[0] / np.sum(s) >= energy:
                energy = s[0] / np.sum(s)
                print("WARNING: Energy choice overwritten to Minimum value: %r" % energy)

            spectralEnergy = np.cumsum(s) / max(np.cumsum(s))
            projectionMatrix = projectionMatrix[:, spectralEnergy <= energy]
            spectralEnergy = spectralEnergy[spectralEnergy <= energy]

            print(" - Size of the subspace: %r " % projectionMatrix.shape[1])

        else:
            assert min(snapshots.shape[0], snapshots.shape[1]) >= energy, \
                "Dimensions maximum value: %r" % min(dims[0], dims[1])

            spectralEnergy = np.cumsum(s) / max(np.cumsum(s))
            projectionMatrix = projectionMatrix[:, 0:energy]
            spectralEnergy = spectralEnergy[0:energy]

            print(" - Size of the subspace: %r " % projectionMatrix.shape[1])

        snapshotRedMat = np.zeros([projectionMatrix.shape[1], snapshots.shape[1],
                                   snapshots.shape[2]])

        for iEns in range(0, snapshots.shape[2], 1):
            snapshotRedMat[:, :, iEns] = \
                projectionMatrix.transpose().dot(snapshots[:, :, iEns])

        return snapshotRedMat, projectionMatrix, spectralEnergy, s

    def componentEstimationCost(self, optimArg, snapshots, inputVectors, weights, dimr):

        dynamics = np.reshape(optimArg[:(dimr*dimr*self.dimDt)].copy(),
                              [dimr,dimr,self.dimDt])

        sensitivities = np.reshape(optimArg[(dimr*dimr*self.dimDt):].copy(),
                                   [dimr,self.numControls,self.dimDt])

        cost = 0.0
        for imodel in range(0,snapshots.shape[2],1):
            simulation = self.__evaluateROM( snapshots[:,[0],imodel],
                                             inputVectors[:,[imodel]],
                                             dynamics, sensitivities, dimr )

            d = simulation - snapshots[:,:,imodel]

            for t in range(1, self.dimDt+1):
                cost = cost + \
                       0.5*(d[:,[t]].transpose().dot(weights[:,:,t])).dot(d[:,
                                                                        [t]]).squeeze()

        ''' d/dX (tr(XXT)) = 2X
            X can be rectangular
        '''
        # print('Cost: ' + str(cost))
        return cost

    def componentEstimationCostDerivative(self, optimArg, snapshots, inputVectors,
                                          weights, dimr):

        # 1.Intialize the cost:
        dj_dyn = np.zeros([dimr, dimr, self.dimDt])
        dj_sen = np.zeros([dimr, self.numControls, self.dimDt])

        dynamics = np.reshape(optimArg[:(dimr * dimr *
                                         self.dimDt)].copy(),
                              [dimr, dimr, self.dimDt])
        sensitivities = np.reshape(optimArg[(dimr * dimr *
                                             self.dimDt):].copy(),
                                   [dimr, self.numControls, self.dimDt])


        for imodel in range(0,snapshots.shape[2],1):

            # 2. Compute the innovation.
            simulation = self.__evaluateROM( snapshots[:,[0],imodel],
                                             inputVectors[:,[imodel]], dynamics,
                                             sensitivities, dimr)

            for it in range(0, self.dimDt+1, 1):

                innovation = weights[:,:,it].dot(simulation[:,[it]] - snapshots[:,[it],
                                                                   imodel])


                # Dynamic components:
                for idyn in range(0, dynamics.shape[2]):
                    dj_dyn[:,:,idyn] = dj_dyn[:,:,idyn] \
                                       + self.diff_rom_dk( snapshots[:,[0],imodel],
                                                           dynamics,
                                                           sensitivities, it,
                                                           inputVectors[:,[imodel]],
                                                           idyn, innovation, dimr)

                # Sensitivity components:
                for isen in range(0, sensitivities.shape[2]):
                    dj_sen[:,:,isen] = dj_sen[:,:,isen] \
                                       + self.diff_rom_sk( dynamics, it,
                                                           inputVectors[:,[imodel]],
                                                           isen, innovation, dimr)

        dJ =  np.concatenate( (np.reshape(dj_dyn,[np.size(dj_dyn)]), np.reshape(dj_sen,[np.size(dj_sen)])), axis=0)

        # print('Cost derivative: ' + str(np.sum(dJ)))
        return dJ

    def evaluateROM(self, initialState, inputVector):

        assert type(initialState) is np.ndarray, \
            "InitialState is not an np.array: %r" % initialState
        assert type(inputVector) is np.ndarray, \
            "InputVector is not an np.array: %r" % inputVector
        assert initialState.shape[0] == self.numStates, \
            " initialState.shape[0] does not match ROM definition: %r." %self.numStates
        assert inputVector.shape[0] == self.numControls, \
            "inputVector.shape[0] does not match ROM definition: %r." %self.numControls

        inpVec = inputVector - self.referenceInput

        simulation  = np.zeros([self.dimr, self.dimDt + 1])
        simulation[:,[0]] = \
            self.projectionMatrix.transpose().dot(initialState -
                                                  self.referenceSimulation[:,[0]])

        for istep in range(0, self.dimDt, 1):
            simulation[:, [istep + 1]] = \
                self.dynamics[:, :, istep].dot(simulation[:, [istep]]) + \
                self.sensitivities[:, :, istep].dot(inpVec)

        simulation = self.projectionMatrix.dot(simulation) + self.referenceSimulation

        return simulation

    def __evaluateROM(self, initialState, inputVector, dynamics, sensitivities, dimr):

        assert type(initialState) is np.ndarray, \
            "InitialState is not an np.array: %r" % initialState
        assert type(inputVector) is np.ndarray, \
            "InputVector is not an np.array: %r" % inputVector
        assert initialState.shape[0] == dimr, \
            "initialState.shape[0] doesn't match ROM definition: %r." % dimr
        assert inputVector.shape[0] == self.numControls, \
            "inputVector.shape[0] doesn't match ROM definition: %r." % self.numControls

        simulation  = np.zeros([dimr, self.dimDt + 1])
        simulation[:,[0]] = initialState.copy()

        for istep in range(0, self.dimDt, 1):
            simulation[:, [istep + 1]] = dynamics[:,:,istep].dot(simulation[:,[istep]]) \
                                         + sensitivities[:,:,istep].dot(inputVector)

        return simulation

    def diff_rom_dk(self, initialState, dynamics, sensitivities, index_end, alpha, k,
                    innovation, dimr):

        assert type(initialState) is np.ndarray, "InitialState is not an np.array: %r" % initialState
        assert type(innovation) is np.ndarray, "Innovation is not an np.array: %r" % innovation

        # initialState = np.reshape(initialState,[self.numStates,1])
        # alpha = np.reshape(alpha,[self.numControls,1])
        # innovation = np.reshape(innovation,[innovation.shape[0],1])

        # Dynamic part:
        dj_dvec = np.zeros([dimr, dimr])

        if (0 > k) or (index_end <= k):
            return dj_dvec

        # Dynamic part:
        kronleft = initialState
        for iindex in range(0,k,1):
            kronleft = dynamics[:,:,iindex].dot(kronleft)

        kronright = np.eye(dimr)
        for iindex in range(k+1,index_end,1):
            kronright = dynamics[:,:,iindex].dot(kronright)

        #   A = kronright
        #   Cu = kronleft

        # 2. Calculate the Jacobian
        dj_dvec = kronright.transpose().dot(innovation).dot(kronleft.transpose())


        # Sensitivity part:
        for iterm in range(0,k,1):

            kronleft = sensitivities[:, :, iterm].dot(alpha)
            for iindex in range(iterm + 1, k, 1):
                kronleft = dynamics[:,:,iindex].dot(kronleft)

            kronright = np.eye(dimr)
            for iindex in range(k + 1, index_end, 1):
                kronright = dynamics[:,:,iindex].dot(kronright)

            # 3. Calculate the Jacobian
            dj_dvec = dj_dvec + \
                      kronright.transpose().dot(innovation).dot(kronleft.transpose())

        return dj_dvec

    def diff_rom_sk(self, dynamics, index_end, alpha, k, innovation, dimr):
        ''' dr_dvec = diff_rom_dk(r_ini,index_ini,index_end,alpha,k,rom)
        where r_ini is the initial condition of the state vector, index_ini is the
        index/time corresponding to the initial condition, index_end is the
        index/time corresponding to the final state (r_j), alpha is the input
        vector, k is index/time of the term that the derivative is being taken of
        and rom is a reduced order model as produced by the model reduced 4DVar. '''

        #Produced by Ivan Garcia (idtfgarcia@gmail.com)

        assert type(alpha) is np.ndarray, "InitialState is not an np.array: %r" % alpha
        assert type(innovation) is np.ndarray, "Innovation is not an np.array: %r" % innovation

        #alpha = np.reshape(alpha,[self.numControls,1])
        #innovation = np.reshape(innovation,[self.numStates,1])


        # Dynamic part:
        dj_dvec = np.zeros([dimr, self.numControls])

        if (0 > k) or (index_end <= k): #I think this should be strictly less than
            return dj_dvec

        # Sensitivity part:

        kronleft = alpha

        kronright = np.eye(dimr)
        for iindex in range(k+1,index_end,1):
            kronright = dynamics[:,:,iindex].dot(kronright)

        dj_dvec = kronright.transpose().dot(innovation.dot(kronleft.transpose()))

        return dj_dvec

    def buildAdjoint(self, dimr, verbose=False):

        for itime in range(0, self.dimDt, 1):

            if verbose: renglon = ''

            self.stateAdjoint[:,:,itime] = np.eye(dimr)
            for iStep in range(0,itime+1,1):
                if verbose: renglon = 'D' + str(iStep) + renglon
                self.stateAdjoint[:,:,itime] = np.dot(self.dynamics[:,:,iStep], self.stateAdjoint[:,:,itime])
            if verbose: print('stateAdjoint[:,:,' + str(itime)+'] = ' + renglon)

            if verbose: print('inputAdjoint[:,:,' + str(itime)+'] = ')
            self.inputAdjoint[:, :, itime] = np.zeros([dimr, self.numControls])
            for iterm in range(0, itime+1, 1):

                if verbose: renglon = 'S' + str(iterm)
                inputTerm = self.sensitivities[:, :, iterm]
                for iStep in range( (iterm+1), itime+1, 1):
                    if verbose: renglon = 'D' + str(iStep) + renglon
                    inputTerm =  np.dot(self.dynamics[:, :, iStep], inputTerm)

                self.inputAdjoint[:, :, itime] = self.inputAdjoint[:, :, itime] + inputTerm
                if verbose: print(renglon + '+')
            if verbose: print('')

    def assimilationCost(self, optimArg, observationSet, observationOperator, \
                         invObservationCovariance, invStateCovariance, invInputCovariance):

        initialCondition = np.reshape(optimArg[:self.numStates], [self.numStates, 1])
        inputVector = np.reshape(optimArg[self.numStates:], [self.numControls, 1])

        cost = 0.0

        simulation = self.evaluateROM(initialCondition, inputVector)

        #TODO: How can we accept non-linear operators
        innovation = observationSet - observationOperator.dot(simulation)

        #TODO: Currently we need an observation at every time
        for itime in range(1,innovation.shape[1],1):
            cost = cost + 0.5*innovation[:,[itime]].transpose().dot(
                invObservationCovariance[:,:,itime]).dot(innovation[:,[itime]]).squeeze()

        cost = cost + \
               0.5*(self.referenceSimulation[:,[0]] - initialCondition).transpose(
                   ).dot(invStateCovariance).dot(self.referenceSimulation[:,[0]] -
                                                 initialCondition).squeeze()
        cost = cost + \
               0.5*(self.referenceInput - inputVector).transpose().dot(
                   invInputCovariance).dot(self.referenceInput - inputVector).squeeze()

        return cost

    def assimilateData(self, observationSet, observationOperator, observationCov,
                       inputVector, initialState, inputCov, initialStateCov):

        from scipy import optimize
        from numpy.linalg import inv
        from matplotlib import pyplot as plt

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
        assert type(inputVector) is np.ndarray, \
            "inputVectorEnsemble is not a np.array: %r" % inputVector
        assert type(initialState) is np.ndarray, \
            "inputVectorEnsemble is not a np.array: %r" % initialState

        print('')
        print('Starting data assimilation')

        invInputCovariance = inv(inputCov)
        invStateCovariance = inv(initialStateCov)

        invObsCovariance = np.ndarray([self.numStates, self.numStates,
                                       observationSet.shape[1]])
        for iobs in range(0, observationCov.shape[2]):
            invObsCovariance[:, :, iobs] = inv(observationCov[:, :, iobs])

        # Project initial state to sub-space and reformat for optimization algorithm
        optimArg = np.zeros([self.numStates + self.numControls, 1])
        optimArg[0:self.numStates, [0]] = initialState
        optimArg[self.numStates:, [0]] = inputVector

        # TODO: The use of the observation operator is still not possible.
        observationOperator = np.eye(self.numStates, self.numStates)

        print(' - Initial Cost: %r' \
        % self.assimilationCost(optimArg, observationSet, observationOperator,
                                invObsCovariance, invStateCovariance, invInputCovariance))

        inputsEns = np.concatenate((self.modelEnsemble[:,0,:],self.inputVectors), axis=0)

        [optimArg, cost, spec] = optimize.fmin_l_bfgs_b(self.assimilationCost, optimArg,
                                                        approx_grad=True,
                                                        args=(observationSet,
                                                              observationOperator,
                                                              invObsCovariance,
                                                              invStateCovariance,
                                                              invInputCovariance),
                                                        iprint=0,disp=0)

        d =  np.concatenate((self.referenceSimulation[:,[0]],self.referenceInput),
                            axis=0) - optimArg


        l2norm = np.sum(d*d)
        ind = None
        for iEns in range(0,inputsEns.shape[1]):

            d = inputsEns[:, [iEns]] - optimArg

            '''fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(d, optimArg, '.b')
            ax.axis('equal')
            ax.axis([0,1500,0,1500])
            plt.show()'''

            l2norm_temp = np.sum(d*d)
            if l2norm_temp < l2norm:
                l2norm = l2norm_temp.copy()
                ind = iEns

        newReferenceSimulation = self.modelEnsemble[:,:,ind]
        newReferenceInput = self.inputVectors[:,[ind]]

        self.modelEnsemble[:,:,ind] = self.referenceSimulation.copy()
        self.inputVectors[:,[ind]] = self.referenceInput.copy()


        cost, projectionMatrix, spectralEnergy, romSingularValues, dynamics, \
        sensitivities, dimr = self.estimateComponents( self.modelEnsemble,
                                                       self.inputVectors,
                                                       newReferenceSimulation,
                                                       newReferenceInput,
                                                       self.energyType,
                                                       self.energy,
                                                       self.maxiter,
                                                       self.factr,
                                                       self.pgtol,
                                                       self.verbose)

        print('indice: %r' %ind)
        print('l2norm: %r' %l2norm)

        print(' - Final cost: ' + str(cost))
        if spec['warnflag'] == 0:
            print(' - Optimization converged')
        elif spec['warnflag'] == 1:
            print(
                ' - Optimization stopped due to too many function evaluations or too many iterations')
        elif spec['warnflag'] == 2:
            print(' - Optimization did not converged due to unknown reason')

        print(' - ' + spec['task'].decode(encoding='UTF-8'))

        # initialState = np.dot(self.projectionMatrix, np.reshape(optimArg[:self.numReducedStates], [self.numReducedStates, 1])) + self.referenceSimulation[:, [0]]
        # inputVector = np.reshape(optimArg[self.numReducedStates:], [self.numControls, 1]) + self.referenceInput
        initialState = np.reshape(optimArg[:self.numStates], [self.numStates, 1])
        inputVector = np.reshape(optimArg[self.numStates:], [self.numControls, 1])

        return [initialState, inputVector, cost]

