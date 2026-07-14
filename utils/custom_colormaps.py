##################################################
'''
Define a high contrast colormap based on matplotlib's 'tab20' for categorical visualization.
'''
##################################################

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl

def _create_high_contrast_tab20():
    '''Generates a high-contrast tab20 color map. '''
    # Load original tab20 cmap
    tab20_colors = list(plt.get_cmap('tab20').colors)

    # separating color pairs
    shuffeld_colors = [tab20_colors[(i * 9) % 20] for i in range(20)]

    # create and return the map
    return mcolors.ListedColormap(shuffeld_colors, name='high_contrast_tab20')

# expose colormap
high_contrast_tab20 = _create_high_contrast_tab20()

try:
    # register in Matplotlib
    mpl.colormaps.register(high_contrast_tab20)
except ValueError:
    pass
except AttributeError:
    plt.register_cmap(cmap=high_contrast_tab20)