__author__ = 'Ivan'

from types import *
import numpy as np


class container:

    def __init__(self, name, metadata, inputVectors, modelEnsemble, referenceInput,
                 referenceSimulation):

        assert type(name) is str, \
            "name is not a string: %r" % type(name)
        assert type(metadata) is str, \
            "metadata is not a string: %r" % type(metadata)
        assert type(modelEnsemble) is np.ndarray, \
            "modelEnsemble is not a np.ndarray: %r" % type(modelEnsemble)
        assert type(inputVectors) is np.ndarray, \
            "inputVectors is not a np.ndarray: %r" % type(inputVectors)
        assert type(referenceSimulation) is np.ndarray, \
            "referenceSimulation is not a np.ndarray: %r" % type(referenceSimulation)
        assert type(referenceInput) is np.ndarray, \
            "referenceInput is not a np.ndarray: %r" % type(referenceInput)

        assert modelEnsemble.ndim == 3, \
            "modelEnsemble should be 3D array: %r" % modelEnsemble.ndim
        assert inputVectors.ndim == 2, \
            "inputVectors should be a 2D array: %r" % inputVectors.ndim
        assert referenceSimulation.ndim == 2, \
            "referenceSimulation should be a 2D array: %r" % referenceSimulation.ndim
        assert referenceInput.ndim == 2, \
            "referenceInput should be a 2D array: %r" % referenceInput.ndim

        assert modelEnsemble.shape[2] == inputVectors.shape[1], \
            "modelEnsemble size does not match the number of inputVectors."
        assert modelEnsemble.shape[:2] == referenceSimulation.shape, \
            "referenceSimulation size doesn't match modelEnsemble members."


        # Define relevant members
        self._metadata = metadata
        self._name = name

        self.referenceSimulation = referenceSimulation.copy()
        self.referenceInput = referenceInput.copy()

        self.modelEnsemble = modelEnsemble.copy()
        self.inputVectors = inputVectors.copy()

        self.sizeEns = modelEnsemble.shape[2]
        self.dimDt = modelEnsemble.shape[1] - 1
        self.numStates = referenceSimulation.shape[0]
        self.numControls = referenceInput.shape[0]


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

    def __evaluateROM(self, initialState, inputVector, dynamics, sensitivities, dimr):

        assert type(initialState) is np.ndarray, \
            "InitialState is not an np.array: %r" % initialState
        assert type(inputVector) is np.ndarray, \
            "InputVector is not an np.array: %r" % inputVector
        assert initialState.shape[0] == dimr, \
            "initialState.shape[0] doesn't match ROM definition: %r." % dimr
        assert inputVector.shape[0] == self.numControls, \
            "inputVector.shape[0] doesn't match ROM definition: %r." % self.numControls

        simulation = np.zeros([dimr, self.dimDt + 1])
        simulation[:, [0]] = initialState.copy()

        for istep in range(0, self.dimDt, 1):
            simulation[:, [istep + 1]] = dynamics[:, :, istep].dot(simulation[:, [istep]]) \
                                         + sensitivities[:, :, istep].dot(inputVector)

        return simulation

