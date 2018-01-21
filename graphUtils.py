__author__ = 'Ivan'

from matplotlib import pyplot as plt
import numpy as np

def plot(x, y, xlabel='', ylabel='', title='', degrade=False, show=True):

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    if degrade:
        for iplot in range(0,max(x.shape[1],y.shape[2])):
            ax.plot(x[:,max(x.shape[1],iplot)], y[:,max(y.shape[1],iplot)], color=[0.4 + iplot*0.6/max(x.shape[1],y.shape[2]), 0.4 + iplot*0.6/max(x.shape[1],y.shape[2]), 0.4 + iplot*0.6/max(x.shape[1],y.shape[2])])
    else:
        ax.plot(x, y, c='r')


    if xlabel:
        ax.set_xlabel(xlabel)

    if ylabel:
        ax.set_ylabel(ylabel)

    if title:
        fig.suptitle(title)
    if show:
        plt.show()

def graphComponentEstimation():

    # ==========================================================
    # Graph Component Estimation Progress
    # ==========================================================
    iterations = np.array([[0,96.55,1061.78],[1,95.51,300.64],[2,95.28,187.08],[3,95.17,186.96],[4,94.94,200.93],
                           [5,94.46,435.94],[6,93.28,699.87],[7,89.73,1171.80],[8,80.78,1798.68],[9,64.00,2330.75],
                           [10,45.58,2654.99],[11,34.58,973.65],[12,31.71,174.56],[13,31.53,30.70],[14,31.52,39.70],
                           [15,31.48,88.86],[16,31.43,415.22],[17,31.34,157.73],[18,31.14,270.04],[19,30.84,521.18],
                           [20,29.87,961.35],[21,28.21,1288.23],[22,27.65,1660.46],[23,26.36,840.76],[24,26.01,283.49],
                           [25,25.93,90.98],[26,25.93,17.09],[27,25.93,17.02],[28,25.93,38.55],[29,25.93,69.36],
                           [30,25.92,121.18],[31,25.89,194.36],[32,25.85,278.36],[33,25.77,239.73],[34,25.67,202.92],
                           [35,25.65,496.90],[36,25.60,48.43],[37,25.59,61.19],[38,25.57,99.90],[39,25.55,97.43],
                           [40,25.53,70.93],[41,25.52,50.95],[42,25.50,65.78]])

    fig1 = plt.figure(1, figsize=(6, 5))

    ax0 = fig1.add_subplot(1, 1, 1)
    ax0.set_title('Component Estimation Progress')
    ax0.set_xlabel('Iteration')
    ax0.set_ylabel('Cost')
    ax0.plot(iterations[:,[0]], iterations[:,[1]]/max(iterations[:,1]), 'b.-')
    ax0.plot(iterations[:,[0]], iterations[:,[2]]/max(iterations[:,2]), 'm.--')
    ax0.set_xlabel('L-BFGS Iteration')
    plt.legend(['Cost Function','Gradient Norm'],loc='upper right')
    fig1.savefig('C:\\Users\\Ivan\\Dropbox\\Writings\\paper_enmor\\manuscript\\figures\\minimizationProgress.pdf')

def graphSpectralAnalysis(rom):

    # ==========================================================
    # Graph Spectral Analysis Outcome
    # ==========================================================
    fig2 = plt.figure(2, figsize=(20, 7))
    ax2 = fig2.add_subplot(1, 1, 1)

    # ax2.set_title('Principal Component Analysis', fontsize = 20)
    ax2.set_xlabel('Principal Components')
    ax2.set_ylabel('Depth [-]')
    for iplt in range(1,rom.projectionMatrix.shape[1]+1):
        ax2.plot((iplt*2-1)*np.ones([2,1]),np.array([0,1.05]),color=[0.8,0.8,0.8])
        ax2.plot((iplt*2-1)*np.ones([rom.projectionMatrix.shape[0],1]) + rom.projectionMatrix[:,[iplt-1]], \
                 np.array(range(1,rom.projectionMatrix.shape[0]+1),dtype=float)/rom.projectionMatrix.shape[0])


    ind = 2*np.array(range(1, rom.projectionMatrix.shape[1]+1))-1
    ax2.plot(ind, rom.spectralEnergy, 'k.--')
    ax2.set_xbound(0,36)
    ax2.set_ybound(0,1.05)
    start, end = ax2.get_xlim()
    ax2.xaxis.set_ticks(np.arange(start, end+1, 1))
    ax2.set_xticklabels(['-1','0','$\pm$1','0','$\pm$1','0','$\pm$1','0','$\pm$1','0', \
                         '$\pm$1','0','$\pm$1','0','$\pm$1','0','$\pm$1','0','$\pm$1', \
                         '0','$\pm$1','0','$\pm$1','0','$\pm$1','0','$\pm$1','0','$\pm$1', \
                         '0','$\pm$1','0','$\pm$1','0','$\pm$1','0','+1'], fontsize=16)

    ax2.xaxis.label.set_fontsize(20)
    ax2.yaxis.label.set_fontsize(20)
    ax2.set_yticklabels(ax2.get_yticks(), fontsize=16)

    fig2.show()
    fig2.savefig('C:\\Users\\Ivan\\Dropbox\\Writings\\paper_enmor\\manuscript\\figures\\spectralAnalysisOutcome.pdf')

def graph3(observationSet, rom):

    # ==========================================================
    # Graph Spectral Analysis Outcome
    # ==========================================================

    snapshot = (observationSet - rom.referenceSimulation)

    '''
    fig3 = plt.figure(3, figsize=(20,7))
    for iplt in range(1,rom.projectionMatrix.shape[1]+1):
        ax3 = fig3.add_subplot(3, 6, iplt)
        y, x = np.mgrid[slice(0, rom.referenceSimulation.shape[0], 1), slice(0, rom.referenceSimulation.shape[1] + 1, 1)]
        PPt = rom.projectionMatrix[:,:iplt].dot(rom.projectionMatrix[:,:iplt].transpose())
        z = PPt.dot(snapshot)
        im3 = ax3.pcolor(x,y,z, cmap='RdBu', vmin=-16, vmax=16)

        ax3.set_xlim([0,11])
        ax3.set_ylim([0,17])

        if iplt % 6 != 1:
            #ax3.set_yticks(ticks=[])
            ax3.tick_params(
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                labelleft='off'
            )
        else:
            ax3.xaxis.label.set_fontsize(16)


        if iplt < 13:
            #ax3.set_yticks(ticks=[])
            ax3.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                labelbottom='off'
            )
        else:
            ax3.yaxis.label.set_fontsize(16)

        fig3.subplots_adjust(right=0.8)
    cbar_ax = fig3.add_axes([0.81, 0.15, 0.02, 0.7])
    fig3.colorbar(im3, cax=cbar_ax)

    fig3.show()

    fig3.savefig('C:\\Users\\Ivan\\Dropbox\\Writings\\paper_enmor\\manuscript\\figures\\differentRankProjections.pdf')
    '''
    colors = ['b','g','r','c','m','y','k']
    fig = plt.figure(figsize=(10, 7))
    freq = np.zeros([10,1])
    ind = np.zeros([11,1])
    cont = 1
    legend = [''] * 9
    theticks = [-15,-10,-5,0,5]
    for iplt in [1,3,6,11]:

        ax4 = fig.add_subplot(2,2,cont)
        PPt = rom.projectionMatrix[:,:iplt].dot(rom.projectionMatrix[:,:iplt].transpose())
        ax4.plot(PPt.dot(snapshot[:,1:]),snapshot[:,1:],'k.',)

        if cont % 2 == 1:
            ax4.set_ylabel('Full rank', fontsize=20)

        ax4.set_yticks(theticks)
        ax4.set_yticklabels(ax4.get_yticks(),fontsize = 16)
        ax4.yaxis.grid(True)
        ax4.set_ylim([-20,10])

        if cont > 2:
            ax4.set_xlabel('Low rank', fontsize=20)

        ax4.set_xticks(theticks)
        ax4.set_xticklabels(ax4.get_xticks(), fontsize = 16)
        ax4.xaxis.grid(True)
        ax4.set_xlim([-20, 10])

        ax4.set_title('Rank: %r' %iplt, fontsize=20)
        fig.tight_layout()
        cont += 1

    fig.savefig('C:\\Users\\Ivan\\Dropbox\\Writings\\paper_enmor\\manuscript\\figures\\performance_different_rank_projections.pdf')

def graphicResultAnalysis(analyzedState, analyzedInput, observationSet, observationInput, referenceSimulation, referenceInput, rom):

    fig = plt.figure(figsize=(15, 10))

    ax0 = fig.add_subplot(2, 2, 1)
    ind = np.arange(observationSet[:, [0]].shape[0])
    plt.plot(ind, referenceSimulation[:, [0]], 'r.', ind, analyzedState, 'b.', ind, observationSet, 'k-')

    ax1 = fig.add_subplot(2, 2, 2)
    ind = np.arange(referenceInput.shape[0])
    width = 0.30  # the width of the bars
    ax1.bar(ind, (referenceInput/observationInput), width, color='b')
    ax1.bar(ind + width, (analyzedInput/observationInput), width, color='r')
    ax1.bar(ind + 2*width, (observationInput/observationInput), width, color='k')
    ax1.legend(['reference', 'analyzed', 'truth'], loc='best')


    ax2 = fig.add_subplot(2, 2, 3)
    y, x = np.mgrid[slice(0, observationSet.shape[0], 1), slice(0, observationSet.shape[1] + 1, 1)]
    z = (referenceSimulation - observationSet)
    im2 = ax2.pcolor(x, y, z, cmap='RdBu', vmin = -10, vmax=10)
    plt.colorbar(im2, ax=ax2)
    ax2.set_xlim([0, 11])
    ax2.set_ylim([0, 17])
    ax2.set_title('Reference minus observations. Max abs diff = ' + str(np.max(np.abs(referenceSimulation - observationSet))))

    ax3 = plt.subplot(2, 2, 4)
    y, x = np.mgrid[slice(0, observationSet.shape[0], 1), slice(0, observationSet.shape[1] + 1, 1)]
    analysisSimulation = rom.evaluateROM(analyzedState,analyzedInput)
    z = (analysisSimulation - observationSet)
    im3 = ax3.pcolor(x, y, z, cmap='RdBu', vmin = -10, vmax=10)
    ax3.set_title('Analysis minus observations. Max abs diff = ' + str(np.max(np.abs(analysisSimulation - observationSet))))
    plt.colorbar(im3, ax=ax3)
    ax3.set_xlim([0, 11])
    ax3.set_ylim([0, 17])

def graphicAnalysis(modelEnsemble, observationSet, observationInput, referenceSimulation, referenceInput):

    ind = np.arange(observationSet.shape[1])
    series = np.zeros([modelEnsemble.shape[2],modelEnsemble.shape[1]])

    # ===============================
    # Make pcolor plot of variances at location / time

    fig0 = plt.figure(1, figsize=(15, 15))

    ax1 = fig0.add_subplot(221)
    y, x = np.mgrid[slice(0, modelEnsemble.shape[0],1), slice(0, modelEnsemble.shape[1]+1,1)]
    z = np.var(modelEnsemble, axis=2)
    z_min, z_max = z.min(), z.max()
    im1 = ax1.pcolor(x, y, z, cmap='RdBu')
    ax1.set_title('Variances per location and time 1')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Location')
    plt.colorbar(im1, ax=ax1)
    ax1.set_xlim([0,11])
    ax1.set_ylim([0,17])

    # ===========================================
    # Make plot of mean absolute deviations from the reference run
    # series[:,:,imember] = abs(modelrun - reference)/reference
    ax2 = fig0.add_subplot(222)
    ax2.set_title('Mean absolute deviations from reference run')
    ax2.set_xlabel('Parameter')
    ax2.set_ylabel('Average Deviation [%]')

    for imem in range(0, modelEnsemble.shape[2]):
        series[imem,:] = 100*(1.0/modelEnsemble.shape[1])*np.sum(np.abs((modelEnsemble[:,:,imem] - referenceSimulation[:,:])/referenceSimulation[:,:]),axis=0)
        ax2.semilogy(ind, series[imem,:], 'k.')

    ax2.semilogy(ind, np.average(series,axis=0),'b-')
    ax2.semilogy(ind, np.average(series,axis=0) - np.std(series,axis=0),'m--')
    ax2.semilogy(ind, np.average(series,axis=0) + np.std(series,axis=0),'m--')
    ax2.set_xlim([-1,11])

    # ===========================================
    # Analysis of observations w.r.t. reference run
    # series[:,:,imember] = abs(modelrun - reference)/reference
    ax3 = fig0.add_subplot(223)
    y, x = np.mgrid[slice(0, modelEnsemble.shape[0],1), slice(0, modelEnsemble.shape[1]+1,1)]
    z = (referenceSimulation - observationSet)
    im3 = ax3.pcolor(x, y, z, cmap = 'RdBu')
    ax3.set_title('Reference minus observations')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Location')
    plt.colorbar(im3, ax=ax3)
    ax3.set_xlim([0, 11])
    ax3.set_ylim([0, 17])

    # ===========================================
    # Make plot of mean absolute deviations from the reference run
    # series[:,:,imember] = abs(modelrun - reference)/reference
    ax4 = fig0.add_subplot(224)
    ind = np.arange(referenceInput.shape[0])
    ax4.set_title('Observation vs reference input vector [% of reference]')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Deviation')
    width = 0.3
    im4 = ax4.bar(ind, (referenceInput), width, color='b')
    im4 = ax4.bar(ind + width, (observationInput), width, color='m')
    im5 = ax4.axhline(0, color='black')

def iterationAnalysis():
    range(1,251,5)
    [66574.14491124,57741.59684093,54565.45227143,54530.03039373,54235.65480511,54097.40377884,53146.128661,  50748.67467517,46733.34546352,44620.07045069,42756.33494622,37951.08836268,32470.70484261,22501.0438647, 21379.31834881,18255.75044594,18124.49654556,18001.2231593, 17858.77775182,17684.9884764,17524.5271592, 17425.58632728,17410.37800647,17402.23665395,17331.1185253, 17314.87559016,17265.96529384,17225.45135244,17193.72619447,17130.61747667,17003.53304185,16916.9443092,16896.83512423,16836.31333676,16784.91067781,16744.67470101,16720.04484062,16657.0838978, 16614.77164805,16602.73903971,16563.48466666,16533.14074016,16513.6083223, 16485.00612591,16477.22380993,16452.82568253,16444.43646543,16429.99305729,16411.50053635,16405.09882385]
    [ 1.19113548,1.24206451,1.24730712,1.20420758,1.17968832,1.19418358,1.23410925,1.09237736,1.02265946,0.89079839,0.67387381,0.46896408,0.48305507,0.31503588,0.27562592,0.23433683,0.25529474,0.30476354,0.24878064,0.24954329,0.27322789,0.2773784, 0.27478719,0.29701563,0.26495465,0.28291139,0.2279121, 0.26475654,0.27150017,0.27915828,0.2475799, 0.24628447,0.26344185,0.26411021,0.24364724,0.24371567,0.25247597,0.27487144,0.22959229,0.26617291,0.27147791,0.248605,0.26592893,0.29790964,0.25044693,0.25766458,0.25445569,0.27120637,0.28809579,0.26716619]
    [ 1.,0.993181,0.972909,0.939737,0.89457,0.838641,0.773474,0.700848,0.622743,0.54129,0.45871,0.377257,0.299152,0.226526,0.161359,0.10543,0.0602631, 0.0270914,0.00681935,0.0]

    icost = 0
    romCost = np.zeros(len(range(1,251,5)))
    analysisCost = np.zeros(len(range(1,251,5)))
    for iNumIter in range(1,251,5):
        rom = romClass('Profile model', 'Idealized model', inputVector, modelEnsemble, referenceInput, referenceSimulation,\
                       1, numIter=iNumIter)

        [analyzedState, analyzedInput, analysisCost[icost]] = rom.assimilateData( observationSet, observationOperator, observationCovariance,\
                                                             np.zeros(referenceInput.shape), referenceSimulation[:,[1]], inputCovariance,\
                                                             initialStateCov)
        romCost[icost] = rom.romCost
        icost = icost + 1

def romErrorStats(modelEnsemble, romError, numStates):

    fig = plt.figure(1, figsize=(10, 5))

    ax1 = plt.subplot(1, 2, 1)
    y, x = np.mgrid[slice(0, modelEnsemble.shape[0],1), slice(0, modelEnsemble.shape[1]+1,1)]
    z = np.mean(romError,axis=2)
    z_min, z_max = -50, 50
    im1 = ax1.pcolor(x, y, z, cmap='RdBu', vmin=-50, vmax=50)
    plt.title('E[ROM error] - Dim: ' + str(numStates))
    plt.xlabel('Time')
    plt.ylabel('Location')
    plt.colorbar(im1, ax=ax1)

    ax2 = plt.subplot(1, 2, 2)
    y, x = np.mgrid[slice(0, modelEnsemble.shape[0],1), slice(0, modelEnsemble.shape[1]+1,1)]
    z = np.var(romError,axis=2)
    im2 = ax2.pcolor(x, y, z, cmap='RdBu')
    plt.title('ROM error variance')
    plt.xlabel('Time')
    plt.ylabel('Location')
    plt.colorbar(im2, ax=ax2)

def romErrorAnalysis(rom, modelEnsemble, inputVector, xShoreCoord):

    romSimulation = np.zeros(modelEnsemble.shape)
    for imem in range(0, modelEnsemble.shape[2]):
        romSimulation[:, :, imem] = rom.evaluateROM(modelEnsemble[:, [0], imem], inputVector[:, [imem]])

    romError = romSimulation - modelEnsemble

    #ToDo LETS MAKE A BAR GRAPH AT EACH LOCATION WITH HORIZONTAL ERROR BANDS
    # |------.-------|
    #     |--.--|
    #   |----.----|

    fig1 = plt.figure(figsize=(25, 10))
    plt.suptitle("ROM error variance", size=16)
    for t in range(1, romSimulation.shape[1], 1):
        ax1 = fig1.add_subplot(2, np.ceil((romSimulation.shape[1]-1)/2), t)
        for imem in range(0, modelEnsemble.shape[2]):
            #im2 = ax2.plot(romError[:,[t],imem]/modelEnsemble[:,:,imem], xShoreCoord,'k.')
            ax1.semilogx(romSimulation[:,[t],imem], np.flipud(xShoreCoord),'r.')
            ax1.semilogx(modelEnsemble[:,[t],imem], np.flipud(xShoreCoord),'b-')

        plt.title('t = ' + str(t))
        if t > np.ceil(romSimulation.shape[1]/2):
            plt.xlabel('X [m]')

        if t == 1 or t == np.ceil(romSimulation.shape[1]/2) + 1:
            plt.ylabel('Depth [m]')

    # ===============================
    # Let's make some histograms to see how far away from gausiannity we are.
    # -------------------------------
    fig2 = plt.figure(figsize=(25, 20))
    plt.suptitle("Error histogram for grid cell at...", size=16)
    n = np.zeros([9])
    ind = np.zeros([9])
    for t in range(1, romError.shape[1], 1):
        ax2 = fig2.add_subplot(2, (np.ceil(romSimulation.shape[1]-1)/2), t)
        ax2.set_title("time %r" % t)
        for iloc in range(0, romError.shape[0]):
            n[1:-1], ind[1:] = np.histogram(romError[iloc,t,:], 7)
            ind[0] = 2*ind[1] - ind[2]
            ax2.plot(ind, n)

    return fig1,fig2

def romPerformacePcolor(modelEnsemble, inputVector, rom):

    cols = 2
    rows = 1

    # ===============================
    # Calculate basic inputs for the plots
    # -------------------------------
    y, x = np.mgrid[slice(0, modelEnsemble.shape[0], 1), slice(0, modelEnsemble.shape[1], 1)]
    romSimulation = np.zeros([modelEnsemble.shape[0],modelEnsemble.shape[1],(cols * rows)])
    members = [100,130]
    cont = 0
    for imem in members:
        romSimulation[:, :, cont] = rom.evaluateROM(modelEnsemble[:, [0], imem],
                                                 inputVector[:, [imem]])
        cont +=1

    # ===============================
    # Pcolor plots of model ensemble vs rom simulation, divided by standard deviations
    # -------------------------------

    fig = plt.figure(figsize=(cols*4, rows*4))
    cont = 1
    from matplotlib.colors import LogNorm
    for imem in members:
        ax = fig.add_subplot(rows, cols, cont)
        ax.set_title("Ensemble Member: %r" % imem)

        z = 100*(modelEnsemble[:,1:,imem] - romSimulation[:,1:,cont-1]) / \
            (modelEnsemble[:, 1:, imem] - modelEnsemble[:, :-1, imem])

        im = ax.pcolor(x, y, z, cmap='RdBu', vmax=15, vmin=-15)
        ax.set_xlim([0, modelEnsemble.shape[1]])
        ax.set_ylim([0, modelEnsemble.shape[0]-1])
        cbar = plt.colorbar(im, ax=ax)
        yticks = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels([tick.get_text() + '%' for tick in yticks],
                                ha='right')
        cbar.ax.yaxis.set_tick_params(pad=40)
        ax.set_xlabel('Time')
        ax.set_ylabel('Cross shore distance')
        ax.set_xlim([0,10])
        ax.set_xticklabels(range(1,12,2))

        cont += 1
    fig.tight_layout()
    return fig

def romPerformaceMAE(modelEnsemble, inputVector, other_members, other_inputs,  rom):

    fig = plt.figure(figsize=(8, 4))

    # ===============================
    # Calculate basic inputs for the plots
    # -------------------------------
    z = np.zeros([modelEnsemble.shape[2],1])
    cont = 0
    for imem in range(0,modelEnsemble.shape[2],1):
        romSimulation = rom.evaluateROM(modelEnsemble[:, [0], imem],
                                                 inputVector[:, [imem]])
        z[cont] = abs(modelEnsemble[:, :, imem] - romSimulation).mean()
        cont +=1
    ax = fig.add_subplot(121)
    ax.plot(range(0, cont, 1), z, '-b', [0, cont - 1], [z.mean(), z.mean()], '-m')
    ax.set_title('Mean absolute error: %.4f.\nStandard deviation: %.4f.' % (z.mean(),
                                                                              z.std()))
    ax.set_ylim([0,0.11])
    ax.set_ylabel('Mean absolute error')
    ax.set_xlabel('Training-ensemble member')
    z = np.zeros([other_members.shape[2], 1])
    cont = 0
    for imem in range(0,other_members.shape[2],1):
        romSimulation = rom.evaluateROM(other_members[:, [0], imem],
                                                 other_inputs[:, [imem]])
        z[cont] = abs(other_members[:, :, imem] - romSimulation).mean()
        cont +=1

    # ===============================
    # Pcolor plots of model ensemble vs rom simulation, divided by standard deviations
    # -------------------------------
    ax = fig.add_subplot(122)
    ax.plot(range(0, cont, 1), z,'-b',[0,cont-1],[z.mean(),z.mean()],'-m')
    ax.set_title('Mean absolute error: %.4f.\nStandard deviation: %.4f.' %(z.mean(),
                                                                            z.std()))
    ax.set_ylim([0, 0.11])
    ax.set_ylabel('Mean absolute error')
    ax.set_xlabel('Validation-ensemble member')

    fig.tight_layout()
    return fig

def dataAssimilation(observationSet, referenceSimulation, analyzedState, referenceInput, observationInput, analyzedInput):

    fig1 = plt.figure(2, figsize=(15, 10))
    plt.subplot(1, 2, 1)
    ind = np.arange(observationSet[:,[0]].shape[0])
    plt.plot(ind, referenceSimulation[:,[0]], 'r.', ind, analyzedState, 'b.', ind, observationSet[:,[0]], 'k-')

    ax6 = plt.subplot(1, 2, 2)
    ind = np.arange(referenceInput.shape[0])
    width = 0.30       # the width of the bars
    rects1 = ax6.bar(ind, (referenceInput - observationInput)/observationInput, width, color='m')
    rects2 = ax6.bar(ind + width, (analyzedInput - observationInput)/observationInput, width, color='b')
    plt.legend(['reference inputs deviations','analyzed input deviations'],loc='upper center')
    #width = 0.15       # the width of the bars
    #rects1 = ax6.bar(ind          , np.log10(referenceInput),   width, color='m')
    #rects2 = ax6.bar(ind +   width, np.log10(analyzedInput),    width, color='b')
    #rects3 = ax6.bar(ind + 2*width, np.log10(observationInput), width, color='c')
    #plt.legend(['reference run inputs','analyzed inputs','observation inputs'])

def graphDynamicSensitivityMatrices(mats,vmin=None,vmax=None):

    fig0 = plt.figure(figsize=(20, 7))
    cbaxes = fig0.add_axes([0.93, 0.1, 0.02, 0.8])
    y, x = np.mgrid[
        slice(0, mats.shape[0] + 1, 1), slice(0, mats.shape[1] +
                                                      1, 1)]
    for iplt in range(0, mats.shape[2], 1):
        ax0 = fig0.add_subplot(2, 5, iplt + 1, title=('T%i -> T%i' % (iplt, iplt + 1)))
        z = np.flipud(mats[:, :, iplt])
        im2 = ax0.pcolor(x, y, z, cmap='RdBu', vmin=vmin, vmax=vmax)
        ax0.xaxis.set_ticks(np.arange(0.5, 10.5, 2))
        ax0.set_xticklabels(['X0', 'X2', 'X4', 'X6', 'X8'])
        ax0.yaxis.set_ticks(np.arange(1.5, 10.5, 2))
        ax0.set_yticklabels(['X8', 'X6', 'X4', 'X2', 'X0'])
    plt.colorbar(im2, cax=cbaxes)
    plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05)
    fig0.tight_layout()

    return fig0