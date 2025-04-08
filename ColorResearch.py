# ColorResearch  Creed Jones  VT ECE  Nov 20, 2024

import numpy as np
import matplotlib.pyplot as plt
import skimage as skim
import ImageUtils as Utils
from ColorConvertor import ColorConvertor, RGBtoGaussian

def main():
    doPlot = True
    conv = ColorConvertor(RGBtoGaussian(doAnalytic=True))
    imgdir = 'C:/Data/CompVisImages/'
    # imgfile = 'bananaplant.png'
    imgfile = 'fruit.png'
    # imgfile = 'SMPTE_Color_Bars.svg.png'
    colorimg = skim.io.imread(imgdir + imgfile)
    ROWS, COLS, PLANES = colorimg.shape
    pixdata = np.reshape(colorimg, (ROWS * COLS, PLANES))
    floatresimg = np.reshape(conv.convertPixels(pixdata), shape=(ROWS, COLS, PLANES))  ### DEBUG HERE
    maxvals = np.max(floatresimg, axis=(0, 1))
    minvals = np.min(floatresimg, axis=(0, 1))
    print("Maxvals: {},  minvals: {}".format(maxvals, minvals))
    Utils.dispimage(colorimg)
    for plane in (0, 1, 2):
        resimg = Utils.floatimagetobyte(floatresimg[:, :, plane], maxval=np.max(floatresimg[:, :, plane]),
                                        minval=np.min(floatresimg[:, :, plane]))
        Utils.dispimage(resimg, map='gray')
        hist = skim.exposure.histogram(resimg)
        if (doPlot):
            plt.plot(hist[1], hist[0])
            plt.show()
        skim.io.imsave(imgdir + imgfile.replace('.png', '_res{}.png'.format(plane)), resimg)

if (__name__ == "__main__"):
    np.set_printoptions(legacy='1.25')      # to stop the annoying print of arrays as "np.float64(whatever)
    main()

