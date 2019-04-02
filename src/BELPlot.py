import numpy as np
import matplotlib.pyplot as plt

def PlotLowDimModels(D,H,DObs,Type='f',FontSize=12, ScatterSize=5, ObservedLineThickness=3, NumPlots=3, MaxPlotCol=3,
                     figwidth = 15, FigName=False):
    """

    :param D: simulated data variables
    :param H: predicted data variables
    :param DObs: observed data variables
    :param Type:
    :param FontSize: size of text on the plots
    :param ScatterSize: Size of the scatter points
    :param ObservedLineThickness: Thickness of the observed data line
    :param NumPlots: total number of plots
    :param MaxPlotCol: number for rows in subplot
    :param figwidth: width of figure
    :return: None
    """

    nPlotRow = int(NumPlots/MaxPlotCol)+1
    fig = plt.figure(figsize=(figwidth,nPlotRow*3))
    ax = []
    for _,ip in enumerate(range(NumPlots)):
        ax.append(fig.add_subplot(nPlotRow,MaxPlotCol,_+1))
        ax[_].plot(D[:,_],H[:,_],'bo', markersize=ScatterSize)
        ax[_].plot([DObs[_],DObs[_]],[np.min(H[:,_]),np.max(H[:,_])],linewidth=ObservedLineThickness,color='red')
        CorCof = np.corrcoef(D[:,_],H[:,_])
        ax[_].set_title(r'$\rho$ = {:.5f}'.format(CorCof[0,1]), fontsize=FontSize)

    if FigName:
        fig.savefig(FigName)


def PlotResponses(DataDict, FigName=False, FontSize = 12, figsize = [10, 5]):
    """

    :param DataDict:
    :param FontSize:
    :param FigName:
    :return:
    """
    types = ['Point', 'TimeSeries']

    if not any(itype in DataDict['type'] for itype in types):
        raise Exception('{} not incorporated'.format(DataDict['type']))

    if DataDict['type'] == types[1]:
        NumRealizations = np.shape(DataDict['data'])[0]
        NumResponses = np.shape(DataDict['data'])[2]

        ax = {}
        for ii in np.arange(NumResponses):
            fig = plt.figure(figsize=figsize)

            ax[ii] = fig.add_subplot(1, 1, 1)
            ax[ii].set_title('{} : {}'.format(DataDict['type'], DataDict['ObjNames'][ii]), FontSize=FontSize)

            for jj in np.arange(NumRealizations):
                lp, = ax[ii].plot(DataDict['time'], DataDict['data'][jj, :, ii], color='Grey', label='Prior')

            ax[ii].set_xlabel('Date', FontSize=FontSize)
            ax[ii].set_ylabel(DataDict['name'], FontSize=FontSize)

            obslist = ['dataTrue', 'dataObs']
            for io in obslist:
                if io in DataDict.keys():
                    ld, = ax[ii].plot(DataDict['time'], DataDict[io][:, ii], color='red', label='Obs')

            ax[ii].legend(handles=[lp, ld])

            if FigName:
                fig.savefig('{}_{:05d}'.format(FigName, ii))

    elif DataDict['type'] == types[0]:
        NumRealizations = np.shape(DataDict['data'])[0]
        NumObs = np.shape(DataDict['data'])[1]
        NumResponses = np.shape(DataDict['data'])[2]

        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('{} : {}'.format(DataDict['type'], DataDict['name']), FontSize=FontSize)

        for ii in np.arange(NumObs):
            lp, = ax.plot(np.zeros(NumRealizations, ) + ii, DataDict['data'][:, ii, 0], marker='*', color='Grey',
                           label='Prior')

        obslist = ['dataTrue', 'dataObs']
        for io in obslist:
            if io in DataDict.keys():
                ld, = ax.plot(np.arange(NumObs), DataDict[io], label='Obs', marker = 'o', markersize= 10,
                              color='red', linestyle='')

        ax.set_ylabel(DataDict['name'], FontSize=FontSize)
        ax.set_xticks(np.arange(NumObs))
        label = ax.set_xticklabels(DataDict['ObjNames'], rotation='vertical')
        ax.legend(handles=[lp, ld])

        if FigName:
            fig.savefig('{}_{:05d}'.format(FigName, ii))







