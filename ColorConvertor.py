# ColorConvertor  Creed Jones  VT ECE  Mar 26, 2025

import numpy as np
import math
import matplotlib.pyplot as plt
import skimage as skim
import scipy
import ImageUtils as Utils
from tqdm import tqdm
import os.path

class Convertor():
    def convertPixels(self, pixels)-> None:
        pass
    def lastArea(self)-> None:
        pass

class ColorConvertor():
    def __init__(self, convertor: Convertor) -> None:
        self._convertor = convertor

    def convertor(self) -> Convertor:
        return self._convertor

    def convertor(self, convertor: Convertor) -> None:
        self._convertor = convertor

    def convertPixels(self, *args):
        return self._convertor.convertPixels(*args)

class RGBtoHSV(Convertor):
    def convertPixels(self, pixels):
        result = skim.rgb2hsv(pixels)
        return result

class RGBtoGaussian(Convertor):
    def __init__(self, doAnalytic=True):
        self.doAnalytic = doAnalytic
        self.lutfilename = "./__RGBtoGaussianLUT.npy"

        if os.path.isfile(self.lutfilename):
            self.lut = np.load(self.lutfilename)
        else:
            MINPIXVAL = 1
            index = np.arange(256 ** 3)
            r = np.maximum(np.bitwise_and(np.right_shift(index, 16), 255), MINPIXVAL)
            g = np.maximum(np.bitwise_and(np.right_shift(index, 8), 255), MINPIXVAL)
            b = np.maximum(np.bitwise_and(index, 255), MINPIXVAL)
            #r = np.maximum(np.divide(np.bitwise_and(np.right_shift(index, 16), 255).astype(float), 255), MINPIXVAL)
            #g = np.maximum(np.divide(np.bitwise_and(np.right_shift(index, 8), 255).astype(float), 255), MINPIXVAL)
            #b = np.maximum(np.divide(np.bitwise_and(index, 255).astype(float), 255), MINPIXVAL)
            rgblist = np.stack((r, g, b), axis=-1)
            self.lut = ConvRGBtoGaussianAnalytic(rgblist)
            np.save(self.lutfilename, self.lut)
        pass

    def convertPixels(self, rgbpix):
        lutval = np.left_shift(rgbpix[:, 0], 16) + np.left_shift(rgbpix[:, 1], 8) + rgbpix[:, 2]
        result = np.take(self.lut, lutval, axis=0)
        return result

def gaussianpdf(x, mu, sigma):
    floatx = x.astype(float)
    return ((1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-np.power(floatx - mu, 2) / (2 * sigma**2)).astype(np.float64))

def ConvRGBtoGaussianAnalytic(rgbinput):      # takes an 8-bit RGB pixel array
    rgbpix = np.divide(rgbinput.astype(float), 255)
    # TODO: need exception processing for overflow, at least
    ALTDENOM = 0.00001
    SMALLESTPIX = 1.0/255
    REDCENTER = 0.640
    GREENCENTER = 0.535
    BLUECENTER = 0.470
    HUGESIGMA = 255.0
    invertflag = np.zeros_like(rgbpix[:,0], dtype=bool)
    (x, y, z) = (REDCENTER, GREENCENTER, BLUECENTER)  # TODO - rethink this considering the centroid of the RGB spectral curves
    rgbpix = np.maximum(np.asarray(rgbpix).astype(float), SMALLESTPIX)
    if (rgbpix.ndim == 1):
        rgbpix = np.expand_dims(rgbpix, 0)
    r = rgbpix[:,0]
    g = rgbpix[:,1]
    b = rgbpix[:,2]
    # if the curve is actually bimodal, the best fit Gaussian will be negative. Determine if that's the case
    # find indices where the color levels are all the same
    indices = [i for i in range(len(rgbpix)) if (g[i]<b[i] and g[i]<=r[i]) or (g[i]<r[i] and g[i]<=b[i])]
    invertflag[indices] = True
    r[indices] = 1-r[indices]+SMALLESTPIX
    g[indices] = 1-g[indices]+SMALLESTPIX
    b[indices] = 1-b[indices]+SMALLESTPIX
    denom = np.log( np.multiply(np.multiply(np.float_power(r, np.subtract(z, y)), np.float_power(g, np.subtract(x, z))), np.float_power(b, np.subtract(y, x))) )
    denom = np.where(denom == 0, ALTDENOM, denom)
    # print("Denominator min = {},  max = {}".format(denom.min(), denom.max()))
    mean = np.multiply(0.5, np.divide(np.log( np.multiply(np.multiply(np.float_power(r, np.subtract(np.power(z,2), np.power(y,2))),
                                                    np.float_power(g, np.subtract(np.power(x,2), np.power(z,2)))),
                                                    np.float_power(b, np.subtract(np.power(y,2), np.power(x,2))))), denom))
    sigma = np.sqrt(np.fabs(np.divide(np.multiply(0.5, np.multiply(np.multiply(np.subtract(x, z), np.subtract(y, x)), np.subtract(z, y))), denom)))
    amplitude = np.minimum(np.multiply(np.multiply(np.multiply(np.sqrt(np.multiply(2, np.pi)), sigma),
                       np.cbrt( np.multiply(r, np.multiply(g, b)))),
                       np.exp(np.divide(np.add(np.add(np.power(np.subtract(x, mean), 2), np.power(np.subtract(y, mean), 2)), np.power(np.subtract(z, mean), 2)),
                                        np.multiply(6, np.power(sigma, 2)))) ), 255)
    mean[indices] = -mean[indices]
    # find indices where the color levels are all the same
    indices = [i for i in range(len(amplitude)) if np.min(rgbpix[i,:]) == np.max(rgbpix[i,:])]
    mean[indices] = GREENCENTER
    sigma[indices] = HUGESIGMA
    amplitude[indices] = r[indices]
    dat = np.stack( (amplitude, mean, sigma), axis=1)
    return dat

def ConvRGBtoGaussianCurveFit(self, rgbpix):
    self.lut = np.zeros((256**3, 3), dtype=float)
    for index in tqdm(range(len(self.lut))):
        rgbpix = ((index >> 16) & 255, (index >> 8) & 255, index & 255)
        (x, y, z) = (0.640, 0.535, 0.470)  # TODO - rethink this considering the centroid of the RGB spectral curves
        intensity = float(np.mean(rgbpix))
        rgbpix = np.maximum(np.asarray(rgbpix, dtype=np.float64), 0.001)
        (r, g, b) = np.divide(rgbpix, 255)
        if (self.doAnalytic):
            # the color representation is intensity, mean, sigma - IMS
            # See Li 1999, The modified three point Gaussian method for determining Gaussian peak parameters
            # https://www.sciencedirect.com/science/article/abs/pii/S0168900298011139
            denom = math.log(r ** (z - y) * g ** (x - z) * b ** (y - x))
            if (denom == 0):
                mean = 0
                sigma = 1
            else:
                mean = 0.5 * (math.log(r ** (z * z - y * y) * g ** (x * x - z * z) * b ** (y * y - x * x))) / denom
                sigma = math.sqrt(math.fabs(0.5 * ((x - z) * (y - x) * (z - y)) / denom))
            # area = math.sqrt(2 * math.pi) * sigma * np.power(r * g * b, 1 / 3.) * np.exp(
            #     ((x - mean) * (x - mean) + (y - mean) * (y - mean) + (z - mean) * (z - mean)) / (6 * sigma * sigma))
        else:
            xdata = np.array([x, y, z])
            ydata = np.array([r, g, b])
            (mean, sigma), cov, infodict, mesg, ier = scipy.optimize.curve_fit(gaussianpdf, xdata, ydata,full_output=True)
        self.lut[index] = (intensity, mean, sigma)

def main(doAnalytic = False):
    doPlot = True
    conv = ColorConvertor(RGBtoGaussian(doAnalytic))
    imgdir = 'C:/Data/CompVisImages/'
    # imgfile = 'bananaplant.png'
    imgfile = 'fruit.png'
    # imgfile = 'SMPTE_Color_Bars.svg.png'
    colorimg = skim.io.imread(imgdir + imgfile)
    ROWS, COLS, PLANES = colorimg.shape
    pixdata = np.reshape(colorimg, (ROWS*COLS, PLANES))
    floatresimg = np.reshape(conv.convertPixels(pixdata), shape=(ROWS, COLS, PLANES))  ### DEBUG HERE
    maxvals = np.max(floatresimg, axis=(0,1))
    minvals = np.min(floatresimg, axis=(0,1))
    print("Maxvals: {},  minvals: {}".format(maxvals, minvals))
    Utils.dispimage(colorimg)
    for plane in (0, 1, 2):
        resimg = Utils.floatimagetobyte(floatresimg[:, :, plane], maxval=np.max(floatresimg[:, :, plane]), minval=np.min(floatresimg[:, :, plane]))
        Utils.dispimage(resimg, map='gray')
        hist = skim.exposure.histogram(resimg)
        if (doPlot):
            plt.plot(hist[1], hist[0])
            plt.show()
        skim.io.imsave(imgdir + imgfile.replace('.png', '_res{}.png'.format(plane)), resimg)

def test(doAnalytic = False):
    doPlot = False
    area = 1.0
    conv = ColorConvertor(RGBtoGaussian(doAnalytic))
    result = gaussianpdf(np.array([400, 500, 600, 700]), 532, 100)
    print(result)
    # rgbpixels = ( [50, 100, 50], [0, 0, 0], [100, 200, 100], [100, 200, 120], [100, 90, 80], [128, 0, 0], [0, 128, 0], [0, 0, 128], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255] )
    # try the SMTPE color bar image
    rgbpixels = ( [192, 192, 192], [192, 192, 0], [0, 192, 192], [0, 192, 0], [192, 0, 192], [192, 0, 0], [0, 0, 192],
                  [0, 0, 192], [19, 19, 19], [192, 0, 192], [19, 19, 19], [0, 192, 192], [19, 19, 19], [192, 192, 192],
                  [0, 33, 76], [255, 255, 255], [50, 0, 106], [19, 19, 19], [9, 9, 9], [19, 19, 19], [29, 29, 29], [19, 19, 19] )
    colorspectrum = np.arange(.4, .8, .01)
    colormeans = np.asarray((0.640, 0.535, 0.470), dtype=np.float64)
    for rgbpix in rgbpixels:
        invertflag = False
        gausspix = np.ravel(conv.convertPixels(np.expand_dims(rgbpix, 0)))
        gausspix2 = np.ravel(ConvRGBtoGaussianAnalytic(np.expand_dims(rgbpix, 0)))
        print("RGB pixel {} converts to Gaussian pixel {} and Gaussian pixel 2 {}".format(rgbpix, gausspix, gausspix2))
        if (gausspix[1] < 0):
            gausspix[1] = -gausspix[1]
            invertflag = True
        print("Points are: {}, {};  {}, {};  {}, {}".format(
            (colormeans[0], rgbpix[0]), (colormeans[0], (1/area)*gaussianpdf(colormeans[0], gausspix[1], gausspix[2])),
                    (colormeans[1], rgbpix[1]), (colormeans[1], (1/area)*gaussianpdf(colormeans[1], gausspix[1], gausspix[2])),
                    (colormeans[2], rgbpix[2]), (colormeans[2], (1/area)*gaussianpdf(colormeans[2], gausspix[1], gausspix[2]))))
        if (invertflag):
            gaussdist = 1-gausspix[0]*255*gaussianpdf(colorspectrum, gausspix[1], gausspix[2])
        else:
            gaussdist = gausspix[0]*255*gaussianpdf(colorspectrum, gausspix[1], gausspix[2])
        gausssum = np.sum(gaussdist)
        print("Gaussian auc is {}".format(gausssum))
        if (doPlot):
            plt.plot(colorspectrum, gaussdist, color='blue')
            plt.scatter(colormeans, rgbpix, color='red')
            plt.show()

def printgaussian():
    colorspectrum = np.arange(.4, .8, .01)
    plt.plot(colorspectrum, gaussianpdf(colorspectrum, 0.550, 0.05), color='green')
    plt.show()


if (__name__ == "__main__"):
    np.set_printoptions(legacy='1.25')      # to stop the annoying print of arrays as "np.float64(whatever)
    for doAnalytic in (False, ):
        print("________________ doAnalytic = {} ______________".format(doAnalytic))
        main(doAnalytic)
        # printgaussian()
        # test(doAnalytic)

