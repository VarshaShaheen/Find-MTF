import matplotlib.pyplot as plt
import pylab as pylab
import numpy as np
import cv2 as cv2
import math as math

from PIL import Image, ImageOps
from scipy import interpolate
from scipy.fft import fft
from enum import Enum
from dataclasses import dataclass
import os


@dataclass
class cSet:
    x: np.ndarray
    y: np.ndarray


@dataclass
class cESF:
    rawESF: cSet
    interpESF: cSet
    threshold: float
    width: float
    angle: float
    edgePoly: np.ndarray


@dataclass
class cMTF:
    x: np.ndarray
    y: np.ndarray
    mtfAtNyquist: float
    width: float


class Verbosity(Enum):
    NONE = 0
    BRIEF = 1
    DETAIL = 2


class Helper:
    @staticmethod
    def LoadImage(filename):
        img = Image.open(filename)
        if img.mode in {'I;16', 'I;16L', 'I;16B', 'I;16N'}:
            gsimg = img
        else:
            gsimg = img.convert('L')
        return gsimg

    @staticmethod
    def LoadImageAsArray(filename):
        img = Helper.LoadImage(filename)
        if img.mode in {'I;16', 'I;16L', 'I;16B', 'I;16N'}:
            arr = np.asarray(img, dtype=np.double) / 65535
        else:
            arr = np.asarray(img, dtype=np.double) / 255
        return arr

    @staticmethod
    def ImageToArray(img):
        if img.mode in {'I;16', 'I;16L', 'I;16B', 'I;16N'}:
            arr = np.asarray(img, dtype=np.double) / 65535
        else:
            arr = np.asarray(img, dtype=np.double) / 255
        return arr

    @staticmethod
    def ArrayToImage(imgArr):
        img = Image.fromarray(imgArr * 255, mode='L')
        return img

    @staticmethod
    def CorrectImageOrientation(imgArr):
        tl = np.average(imgArr[0:2, 0:2])
        tr = np.average(imgArr[0:2, -3:-1])
        bl = np.average(imgArr[-3:-1, 0:2])
        br = np.average(imgArr[-3:-1, -3:-1])
        edges = [tl, tr, bl, br]
        edgeIndexes = np.argsort(edges)
        if (edgeIndexes[0] + edgeIndexes[1]) == 1:
            pass
        elif (edgeIndexes[0] + edgeIndexes[1]) == 5:
            imgArr = np.flip(imgArr, axis=0)
        elif (edgeIndexes[0] + edgeIndexes[1]) == 2:
            imgArr = np.transpose(imgArr)
        elif (edgeIndexes[0] + edgeIndexes[1]) == 4:
            imgArr = np.flip(np.transpose(imgArr), axis=0)

        return imgArr


class MTF:
    nyquistFrequency = 0.5

    @staticmethod
    def SafeCrop(values, distances, head, tail):
        isIncrementing = True
        if distances[0] > distances[-1]:
            isIncrementing = False
            distances = -distances
            dummy = -tail
            tail = -head
            head = dummy

        hindex = (np.where(distances < head)[0])
        tindex = (np.where(distances > tail)[0])

        if hindex.size < 2:
            h = 0
        else:
            h = np.amax(hindex)

        if tindex.size == 0:
            t = distances.size
        else:
            t = np.amin(tindex)

        if not isIncrementing:
            distances = -distances

        return cSet(distances[h:t], values[h:t])

    @staticmethod
    def GetEdgeSpreadFunction(imgArr, edgePoly, verbose=Verbosity.NONE):
        Y = imgArr.shape[0]
        X = imgArr.shape[1]

        values = np.reshape(imgArr, X * Y)

        distance = np.zeros((Y, X))
        column = np.arange(0, X) + 0.5
        for y in range(Y):
            distance[y, :] = (edgePoly[0] * column - (y + 0.5) + edgePoly[1]) / np.sqrt(edgePoly[0] * edgePoly[0] + 1)

        distances = np.reshape(distance, X * Y)
        indexes = np.argsort(distances)

        sign = 1
        if np.average(values[indexes[:10]]) > np.average(values[indexes[-10:]]):
            sign = -1

        values = values[indexes]
        distances = sign * distances[indexes]

        if distances[0] > distances[-1]:
            distances = np.flip(distances)
            values = np.flip(values)

        if verbose == Verbosity.BRIEF:
            print(
                "Raw ESF [done] (Distance from {0:2.2f} to {1:2.2f})".format(sign * distances[0], sign * distances[-1]))

        elif verbose == Verbosity.DETAIL:
            x = [0, np.size(imgArr, 1) - 1]
            y = np.polyval(edgePoly, x)

            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('Raw ESF')
            (ax1, ax2) = plt.subplots(2)
            ax1.imshow(imgArr, cmap='gray', vmin=0.0, vmax=1.0)
            ax1.plot(x, y, color='red')
            ax2.plot(distances, values)
            plt.show()
            plt.show(block=False)

        return cSet(distances, values)

    @staticmethod
    def GetEdgeSpreadFunctionCrop(imgArr, verbose=Verbosity.NONE):
        imgArr = Helper.CorrectImageOrientation(imgArr)
        edgeImg = cv2.Canny(np.uint8(imgArr * 255), 40, 90, L2gradient=True)

        line = np.argwhere(edgeImg == 255)
        edgePoly = np.polyfit(line[:, 1], line[:, 0], 1)
        angle = math.degrees(math.atan(-edgePoly[0]))

        finalEdgePoly = edgePoly.copy()
        if angle > 0:
            imgArr = np.flip(imgArr, axis=1)
            finalEdgePoly[1] = np.polyval(edgePoly, np.size(imgArr, 1) - 1)
            finalEdgePoly[0] = -edgePoly[0]

        esf = MTF.GetEdgeSpreadFunction(imgArr, finalEdgePoly, Verbosity.NONE)

        esfValues = esf.y
        esfDistances = esf.x

        maximum = np.amax(esfValues)
        minimum = np.amin(esfValues)

        threshold = (maximum - minimum) * 0.1

        head = np.amax(esfDistances[(np.where(esfValues < minimum + threshold))[0]])
        tail = np.amin(esfDistances[(np.where(esfValues > maximum - threshold))[0]])

        width = abs(head - tail)

        esfRaw = MTF.SafeCrop(esfValues, esfDistances, head - 1.2 * width, tail + 1.2 * width)

        qs = np.linspace(0, 1, 20)[1:-1]
        knots = np.quantile(esfRaw.x, qs)
        tck = interpolate.splrep(esfRaw.x, esfRaw.y, t=knots, k=3)
        ysmooth = interpolate.splev(esfRaw.x, tck)

        InterpDistances = np.linspace(esfRaw.x[0], esfRaw.x[-1], 500)
        InterpValues = np.interp(InterpDistances, esfRaw.x, ysmooth)

        esfInterp = cSet(InterpDistances, InterpValues)

        if verbose == Verbosity.BRIEF:
            print("ESF Crop [done] (Distance from {0:2.2f} to {1:2.2f})".format(esfRaw.x[0], esfRaw.x[-1]))

        elif verbose == Verbosity.DETAIL:
            x = [0, np.size(imgArr, 1) - 1]
            y = np.polyval(finalEdgePoly, x)

            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('ESF Crop')
            (ax1, ax2) = plt.subplots(2)
            ax1.imshow(imgArr, cmap='gray', vmin=0.0, vmax=1.0)
            ax1.plot(x, y, color='red')
            ax2.plot(esfRaw.x, esfRaw.y, InterpDistances, InterpValues)
            plt.show(block=False)
            plt.show()

        return cESF(esfRaw, esfInterp, threshold, width, angle, edgePoly)

    @staticmethod
    def SimplifyEdgeSpreadFunction(esf, verbose=Verbosity.NONE):
        res = np.unique(esf.x, return_index=True, return_counts=True)

        indexes = res[1]
        counts = res[2]
        sz = np.size(res[0])

        distances = esf.x[indexes]
        values = np.zeros(sz, dtype=np.float)

        for x in range(sz):
            values[x] = np.sum(esf.y[indexes[x]:indexes[x] + counts[x]]) / counts[x]

        if verbose == Verbosity.BRIEF:
            print("ESF Simplification [done] (Size from {0:d} to {1:d})".format(np.size(esf.x), np.size(distances)))

        elif verbose == Verbosity.DETAIL:
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title(
                "ESF Simplification (Size from {0:d} to {1:d})".format(np.size(esf.x), np.size(distances)))
            (ax1, ax2) = plt.subplots(2)
            ax1.plot(esf.x, esf.y)
            ax2.plot(distances, values)
            plt.show(block=False)
            plt.show()

        return cSet(distances, values)

    @staticmethod
    def GetLineSpreadFunction(esf, normalize=True, verbose=Verbosity.NONE):
        lsfDividend = np.diff(esf.y)
        lsfDivisor = np.diff(esf.x)

        lsfValues = np.divide(lsfDividend, lsfDivisor)
        lsfDistances = esf.x[0:-1]

        if normalize:
            lsfValues = lsfValues / (max(lsfValues))

        if verbose == Verbosity.BRIEF:
            print("MTF [done]")

        elif verbose == Verbosity.DETAIL:
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title("LSF")
            (ax1) = plt.subplots(1)
            ax1.plot(lsfDistances, lsfValues)
            plt.show(block=False)
            plt.show()

        return cSet(lsfDistances, lsfValues)

    @staticmethod
    def GetMTF(lsf, verbose=Verbosity.NONE):
        N = np.size(lsf.x)
        px = N / (lsf.x[-1] - lsf.x[0])
        values = 1 / np.sum(lsf.y) * abs(fft(lsf.y))
        distances = np.arange(0, N) / N * px

        interpDistances = np.linspace(0, 1, 50)
        interp = interpolate.interp1d(distances, values, kind='cubic')
        interpValues = interp(interpDistances)
        valueAtNyquist = interp(MTF.nyquistFrequency)

        if verbose == Verbosity.BRIEF:
            print("MTF [done]")

        elif verbose == Verbosity.DETAIL:
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title("MTF ({0:2.2f}% at Nyquist)".format(valueAtNyquist))
            (ax1) = plt.subplots(1)
            ax1.plot(interpDistances, interpValues)
            # ax1.plot( values)
            plt.show(block=False)
            plt.show()

        return cMTF(interpDistances, interpValues, valueAtNyquist, -1.0)

    @staticmethod
    def CalculateMtf(imgArr,plot_save_path,  verbose=Verbosity.NONE):
        global x, y
        imgArr = Helper.CorrectImageOrientation(imgArr)
        esf = MTF.GetEdgeSpreadFunctionCrop(imgArr, Verbosity.NONE)
        lsf = MTF.GetLineSpreadFunction(esf.interpESF, True, Verbosity.NONE)
        mtf = MTF.GetMTF(lsf, Verbosity.NONE)

        if verbose == Verbosity.BRIEF:
            print("MTF at Nyquist:{0:0.2f}%, Transition Width:{1:0.2f}".format(mtf.mtfAtNyquist, esf.width))

        elif verbose == Verbosity.DETAIL:
            x = [0, np.size(imgArr, 1) - 1]
            y = np.polyval(esf.edgePoly, x)

            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('MTF Analysis')
            gs = fig.add_gridspec(3, 2)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[2, 0])
            ax4 = fig.add_subplot(gs[:, 1])

            ax1.imshow(imgArr, cmap='gray', vmin=0.0, vmax=1.0)
            ax1.plot(x, y, color='red')
            ax1.axis('off')
            ax2.plot(esf.rawESF.x, esf.rawESF.y,
                     esf.interpESF.x, esf.interpESF.y)
            top = np.max(esf.rawESF.y) - esf.threshold
            bot = np.min(esf.rawESF.y) + esf.threshold
            ax2.plot([esf.rawESF.x[0], esf.rawESF.x[-1]], [top, top], color='red')
            ax2.plot([esf.rawESF.x[0], esf.rawESF.x[-1]], [bot, bot], color='red')
            ax2.xaxis.set_visible(False)
            ax2.yaxis.set_visible(False)
            ax3.plot(lsf.x, lsf.y)
            ax3.xaxis.set_visible(False)
            ax3.yaxis.set_visible(False)
            ax4.plot(mtf.x, mtf.y)
            ax4.set_title("MTF at Nyquist:{0:0.2f}\nTransition Width:{1:0.2f}".format(mtf.mtfAtNyquist, esf.width))
            print(mtf.mtfAtNyquist)
            ax4.grid(True)

            plt.savefig(plot_save_path)  # Save the plot before displaying it
            plt.show(block=False)
            plt.show()
            plt.close()

            return cMTF(x, y, mtf.mtfAtNyquist, esf.width)
