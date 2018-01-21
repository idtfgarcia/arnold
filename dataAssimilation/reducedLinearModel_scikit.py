__author__ = 'Ivan'

import numpy as np
from sklearn import linear_model


class reducedOrderModel:

    def __init__(self, name, metadata, referenceInput, referenceSimulation,
                 modelEnsemble, ensembleInputs, energyType, energy):

        assert type(name) is str, \
            "name is not a string: %r" %type(name)

        assert type(metadata) is str, \
            "metadata is not a string: %r" %type(metadata)

        assert type(modelEnsemble) is np.ndarray, \
            "modelEnsemble is not a np.ndarray: %r" %type(modelEnsemble)

        assert type(ensembleInputs) is np.ndarray, \
            "inputVectors is not a np.ndarray: %r" %type(ensembleInputs)

        assert type(referenceSimulation) is np.ndarray,\
            "referenceSimulation is not a np.ndarray: %r" %type(referenceSimulation)

        assert type(referenceInput) is np.ndarray, \
            "referenceInput is not a np.ndarray: %r" %type(referenceInput)

        self.__metadata = metadata
        self.__name = name

        self.referenceSimulation = referenceSimulation.copy()
        self.referenceInput = referenceInput.copy()

        self.numStates = modelEnsemble.shape[0]
        self.numt = modelEnsemble.shape[1]
        self.numControls = ensembleInputs.shape[0]

        self.energyType = energyType
        self.energy = energy

        # Model order reduction:
        # Compute the sub-basis of the space spanned by the ensemble

        # Process the data into the right format. The analysis requires
        # snapshot matrix rather than an ensemble of simulations.
        [snapshots, snapshotInputs] = \
            self.__getSnapshotMatrix(modelEnsemble, ensembleInputs, referenceSimulation,
                                     referenceInput)

        [snapshotRedMat, self.projectionMatrix, self.spectralEnergy,
         self.romSingularValues] = self.__modelOrderReduction(snapshots, energyType,
                                                              energy)

        self.dimr = snapshotRedMat.shape[0]
        self.dimDt = modelEnsemble.shape[1] - 1
        [self.dynamics, self.sensitivities] = self.__estimateComponents(snapshotRedMat,
                                                            snapshotInputs)


    def __getSnapshotMatrix(self, modelEnsemble, ensembleInputs, referenceSimulation,
                            referenceInput):
        ''' Build the snapshots with the reference run previously chosen. '''

        snapshots = np.zeros(modelEnsemble.shape)
        snapshotInputs = np.zeros(ensembleInputs.shape)

        for iEns in range(0,modelEnsemble.shape[2],1):
            snapshots[:,:,iEns] = modelEnsemble[:,:,iEns] - referenceSimulation
            snapshotInputs[:,[iEns]] = ensembleInputs[:,[iEns]] - referenceInput

        return snapshots, snapshotInputs

    def __modelOrderReduction(self, snapshots, energyType, energy, normalize=True):
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
                                   snapshots.shape[2] ])

        for iEns in range(0, snapshots.shape[2], 1):
            snapshotRedMat[:, :, iEns] = \
                projectionMatrix.transpose().dot(snapshots[:, :, iEns])

        return snapshotRedMat, projectionMatrix, spectralEnergy, s

    def __estimateComponents(self, snapshotRedMat, snapshotInputs):

        from scipy import optimize
        import time

        print(' - %r dimensional model.' % self.dimr)
        print(' - %r controls detected.' % self.numControls)
        print(' - %r required components.' % self.dimDt)

        weights = np.ndarray([self.dimr, self.dimr, self.dimDt + 1])
        for t in range(0, self.dimDt + 1, 1):
            weights[:, :, t] = np.eye(self.dimr, self.dimr)

        print('')
        print('Linear model identification process:')
        print(' - Initializing components.')

        dynamics = np.zeros([self.dimr, self.dimr, self.dimDt])
        sensitivities = np.zeros([self.dimr, self.numControls, self.dimDt])

        for iStep in range(0, self.dimDt, 1):
            C = np.concatenate((snapshotRedMat[:, iStep, :], snapshotInputs),
                               axis=0)
            rhs = C.dot(C.transpose())
            lhs = snapshotRedMat[:, iStep + 1, :].dot(C.transpose())
            system = lhs.dot(np.linalg.pinv(rhs))

            dynamics[:, :, iStep] = system[:, :self.dimr].copy()
            sensitivities[:, :, iStep] = system[:, self.dimr:].copy()

        optimArg = np.concatenate(
            (np.reshape(dynamics, [self.dimr * self.dimr * self.dimDt, 1]),
             np.reshape(sensitivities, [self.dimr * self.numControls * self.dimDt, 1])),
            axis=0)

        cost = self.__componentEstimationCost(optimArg, snapshotRedMat, snapshotInputs,
                                            weights)

        print(' - Initial Cost: %r ' % cost)
        print(' - Minimization started (This process may take long time...)')
        tic = time.time()

        '''
         - factr (float): The iteration stops when
                    (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps,
            where eps is the machine precision, which is automatically generated by the
            code. Typical values for factr are:1e12 for low accuracy; 1e7 for moderate
            accuracy; 10.0 for extremely high accuracy.

         -  pgtol (float): The iteration will stop when
                max{|proj g_i | i = 1, ..., n} <= pgtol
            where pg_i is the i-th component of the projected gradient.

         - Callback (callable, optional): Called after each iteration, as callback(xk),
            where xk is the current parameter vector '''
        [optimArg, cost, spec] = optimize.fmin_l_bfgs_b(self.__componentEstimationCost,
                                   optimArg,
                                   fprime=self.__componentEstimationCostDerivative,
                                   args=(snapshotRedMat, snapshotInputs, weights),
                                   maxiter=60,
                                   disp=False)

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

        dynamics = np.reshape(optimArg[:(self.dimr * self.dimr * self.dimDt)],
                              [self.dimr, self.dimr, self.dimDt])
        sensitivities = np.reshape(optimArg[(self.dimr * self.dimr * self.dimDt):],
                                   [self.dimr, self.numControls, self.dimDt])

        return dynamics, sensitivities

    def __componentEstimationCost(self, optimArg, snapshots, inputVectors, weights):

        dynamics = np.reshape(optimArg[:(self.dimr*self.dimr*self.dimDt)].copy(),
                              [self.dimr,self.dimr,self.dimDt])

        sensitivities = np.reshape(optimArg[(self.dimr*self.dimr*self.dimDt):].copy(),
                                   [self.dimr,self.numControls,self.dimDt])

        cost = 0.0
        for imodel in range(0,snapshots.shape[2],1):
            simulation = self.__evaluateROM( snapshots[:,[0],imodel],
                                             inputVectors[:,[imodel]],
                                             dynamics, sensitivities, self.dimr )

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

    def __componentEstimationCostDerivative(self, optimArg, snapshotRedMat, inputVectors,
                                          weights):

        # 1.Intialize the cost:
        dj_dyn = np.zeros([self.dimr, self.dimr, self.dimDt])
        dj_sen = np.zeros([self.dimr, self.numControls, self.dimDt])

        dynamics = np.reshape(optimArg[:(self.dimr * self.dimr *
                                         self.dimDt)].copy(),
                              [self.dimr, self.dimr, self.dimDt])
        sensitivities = np.reshape(optimArg[(self.dimr * self.dimr *
                                             self.dimDt):].copy(),
                                   [self.dimr, self.numControls, self.dimDt])


        for imodel in range(0,snapshotRedMat.shape[2],1):

            # 2. Compute the innovation.
            simulation = self.__evaluateROM( snapshotRedMat[:,[0],imodel],
                                             inputVectors[:,[imodel]], dynamics,
                                             sensitivities, self.dimr)

            for it in range(0, self.dimDt+1, 1):

                innovation = weights[:,:,it].dot(simulation[:,[it]] - snapshotRedMat[:,[it],
                                                                   imodel])


                # Dynamic components:
                for idyn in range(0, dynamics.shape[2]):
                    dj_dyn[:,:,idyn] = dj_dyn[:,:,idyn] \
                                       + self.__diff_rom_dk( snapshotRedMat[:,[0],imodel],
                                                           dynamics,
                                                           sensitivities, it,
                                                           inputVectors[:,[imodel]],
                                                           idyn, innovation)

                # Sensitivity components:
                for isen in range(0, sensitivities.shape[2]):
                    dj_sen[:,:,isen] = dj_sen[:,:,isen] \
                                       + self.__diff_rom_sk( dynamics, it,
                                                           inputVectors[:,[imodel]],
                                                           isen, innovation)

        dJ =  np.concatenate( (np.reshape(dj_dyn,[np.size(dj_dyn)]), np.reshape(dj_sen,[np.size(dj_sen)])), axis=0)

        # print('Cost derivative: ' + str(np.sum(dJ)))
        return dJ

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

    def __diff_rom_dk(self, initialState, dynamics, sensitivities, index_end, alpha, k,
                    innovation):

        assert type(initialState) is np.ndarray, "InitialState is not an np.array: %r" % initialState
        assert type(innovation) is np.ndarray, "Innovation is not an np.array: %r" % innovation

        # initialState = np.reshape(initialState,[self.numStates,1])
        # alpha = np.reshape(alpha,[self.numControls,1])
        # innovation = np.reshape(innovation,[innovation.shape[0],1])

        # Dynamic part:
        dj_dvec = np.zeros([self.dimr, self.dimr])

        if (0 > k) or (index_end <= k):
            return dj_dvec

        # Dynamic part:
        kronleft = initialState
        for iindex in range(0,k,1):
            kronleft = dynamics[:,:,iindex].dot(kronleft)

        kronright = np.eye(self.dimr)
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

            kronright = np.eye(self.dimr)
            for iindex in range(k + 1, index_end, 1):
                kronright = dynamics[:,:,iindex].dot(kronright)

            # 3. Calculate the Jacobian
            dj_dvec = dj_dvec + \
                      kronright.transpose().dot(innovation).dot(kronleft.transpose())

        return dj_dvec

    def __diff_rom_sk(self, dynamics, index_end, alpha, k, innovation):
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
        dj_dvec = np.zeros([self.dimr, self.numControls])

        if (0 > k) or (index_end <= k): #I think this should be strictly less than
            return dj_dvec

        # Sensitivity part:

        kronleft = alpha

        kronright = np.eye(self.dimr)
        for iindex in range(k+1,index_end,1):
            kronright = dynamics[:,:,iindex].dot(kronright)

        dj_dvec = kronright.transpose().dot(innovation.dot(kronleft.transpose()))

        return dj_dvec

    def evaluateROM(self, initialState, inputVector):

        assert type(initialState) is np.ndarray, \
            "InitialState is not an np.array: %r" % initialState
        assert type(inputVector) is np.ndarray, \
            "InputVector is not an np.array: %r" % inputVector
        assert initialState.shape[0] == self.numStates, \
            " initialState.shape[0] does not match ROM definition: %r." % self.numStates
        assert inputVector.shape[0] == self.numControls, \
            "inputVector.shape[0] does not match ROM definition: %r." % self.numControls

        inpVec = inputVector - self.referenceInput

        simulation = np.zeros([self.dimr, self.dimDt + 1])
        simulation[:, [0]] = \
            self.projectionMatrix.transpose().dot(initialState -
                                                  self.referenceSimulation[:, [0]])

        for istep in range(0, self.dimDt, 1):
            simulation[:, [istep + 1]] = \
                self.dynamics[:, :, istep].dot(simulation[:, [istep]]) + \
                self.sensitivities[:, :, istep].dot(inpVec)

        simulation = self.projectionMatrix.dot(simulation) + self.referenceSimulation

        return simulation

    def evaluateROM_scikit(self, initialState, inputVector):

        assert type(initialState) is np.ndarray, \
            "InitialState is not an np.array: %r" % initialState

        assert type(inputVector) is np.ndarray, \
            "InputVector is not an np.array: %r" % inputVector

        assert initialState.shape[0] == self.numStates, \
            " initialState.shape[0] does not match ROM definition: %r." %self.numStates

        assert inputVector.shape[0] == self.numControls, \
            "inputVector.shape[0] does not match ROM definition: %r." %self.numControls

        u = inputVector - self.referenceInput
        xt = initialState - self.referenceSimulation[:,[0]]

        simulation = np.zeros([self.dimr, self.numt])
        simulation[:, [0]] = self.projectionMatrix.transpose().dot(xt)

        for istep in range(0, self.numt-1, 1):
            Xt = np.transpose(np.concatenate((simulation[:,[istep]],u), axis=0))
            simulation[:, [istep + 1]] = np.transpose(self.regr[istep].predict(Xt))

        simulation = self.projectionMatrix.dot(simulation) + self.referenceSimulation

        return simulation