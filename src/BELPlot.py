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



