import numpy as np
import cupy as cp
import copy
from sklearn.decomposition import SparsePCA
from abc import *

class ConstructTool(metaClass=ABCMeta):

    def __init__(self, mode="numpy"):
        self.mode = mode

    def _centerizeCP(self, dataArray):

        mean = cp.mean(dataArray, axis=1)
        centerizedDataArray = copy.deepcopy(dataArray)
        for col in range(centerizedDataArray.shape[1]):
            centerizedDataArray[:,col] = centerizedDataArray[:,col] - mean
        return centerizedDataArray
    
    def _centerizeNP(self, dataArray):

        mean = np.mean(dataArray, axis=1)
        centerizedDataArray = copy.deepcopy(dataArray)
        for col in range(centerizedDataArray.shape[1]):
            centerizedDataArray[:,col] = centerizedDataArray[:,col] - mean
        return centerizedDataArray
    
    def centerize(self, dataArray):
        """
        Centerize the data given with the cupy or numpy.
        Each column vector of data array should be one datapoint.

        Args:
            dataArray: 'numpy.ndarray' or 'cupy.ndarray'
                The data array which is target of centerizing.
                The type of this array should be 'numpy.ndarray' or 'cupy.ndarray'.

        Raises:
            -

        Returns:
            centerizedDataArray: 'numpy.ndarray' or 'cupy.ndarray'
                Centerized data array.
        """
        if self.mode == "numpy": return self._centerizeNP(dataArray)
        elif self.mode == "cupy": return self._centerizeCP(dataArray)

    @abstractmethod
    def basicConst(self):
        pass

class QRTool(ConstructTool):
    def __init__(self, mode="numpy"):
        super().__init__(self, mode=mode)
    
    def basicConst(self, dataArray, centerize=False):
        X = None
        if centerize: X = self.centerize(dataArray)
        else: X = dataArray
        Q = None
        if self.mode == "numpy": Q, _ = np.linalg.qr(X)
        elif self.mode == "cupy": Q, _ = cp.linalg.qr(X)
        return Q

class PCATool(ConstructTool):
    def __init__(self, mode="numpy"):
        super().__init__(self, mode=mode)

    def printEigenvalues(self, values):

        values_sum = None
        if self.mode == "numpy": values_sum = np.sum(values)
        elif self.mode == "cupy": values_sum = cp.sum(values)

        print("<1. Top-10 eigenvalues>")
        print("lambda || cumulative sum || proportion of lambda || proportion of cumulative sum")
        sum_cumul = 0
        printCounter = 0
        for i, element in enumerate(values):
            printCounter += 1
            sum_cumul += element
            print("{:04d}. ".format(i), end="")
            print("{:.4f}".format(element), end=" || ")
            print("{:.4f}".format(sum_cumul), end=" || ")
            print("{:.4f}".format(element / values_sum), end=" || ")
            print("{:.4f}".format(sum_cumul / values_sum))
            if printCounter > 9:
                print()
                break

        print("<2. Overall distribution of eigenvalues>")
        print("lambda || cumulative sum || proportion of lambda || proportion of cumulative sum")
        sum_cumul = 0
        printCounter = 0
        period = int(len(values) / 20)
        for i, element in enumerate(values):
            sum_cumul += element
            if i % period == 0:
                print("{:04d}. ".format(i), end="")
                print("{:.4f}".format(element), end=" || ")
                print("{:.4f}".format(sum_cumul), end=" || ")
                print("{:.4f}".format(element / values_sum), end=" || ")
                print("{:.4f}".format(sum_cumul / values_sum))
            if element < 0.1:
                printCounter += 1
            if printCounter > 9:
                print()
                break

        print("<3. Intrinsic Dimension of subspace>")

        index = 0
        changeFlag = False
        for i, l in enumerate(values):
            if l < 1e-4:
                index = i
                changeFlag = True
                break
        if not changeFlag: index = values.shape[0]

        intrinsic_index_90 = 0
        intrinsic_index_95 = 0
        intrinsic_index_99 = 0
        foundFlag_90 = False
        foundFlag_95 = False
        sum_cumul = 0
        for i, element in enumerate(values):
            sum_cumul += element
            if sum_cumul / values_sum > 0.9 and not foundFlag_90:
                intrinsic_index_90 = i
                foundFlag_90 = True
            if sum_cumul / values_sum > 0.95 and not foundFlag_95:
                intrinsic_index_95 = i
                foundFlag_95 = True
            if sum_cumul / values_sum > 0.99:
                intrinsic_index_99 = i
                break
        print("90Percent-dimension : {:04d}".format(intrinsic_index_90))
        print("95Percent-dimension : {:04d}".format(intrinsic_index_95))
        print("99Percent-dimension : {:04d}".format(intrinsic_index_99))
        print("real-dimension : {:04d}".format(index))
        print("full-dimension : {:04d}".format(len(values)))
        print()
        
    def basicConst(self, dataArray, centerize=True, isVerbose=False): #  just doing PCA(doing eigen composition about auto-correlation matrix)

        dimension = dataArray.shape[0]
        dataNum = dataArray.shape[1]

        X = None
        if centerize: X = self.centerize(dataArray)
        else: X = dataArray

        values = None
        basis = None
        if self.mode == "numpy":
            r = X@np.transpose(X)
            values, basis = np.linalg.eigh(r)
        elif self.mode == "cupy":
            r = X@cp.transpose(X)
            values, basis = cp.linalg.eigh(r)
        
        values = values[::-1]
        basis = basis[:, ::-1]

        if dimension > dataNum:
            values = values[:dataNum]
            basis = basis[:,:dataNum]

        if isVerbose:
            print("="*20)
            self.printEigenvalues(values)
            print("="*20)

        return [values, basis]
    
    def lowcostConst(self, dataArray, centerize=True, isVerbose=False):

        dimension = dataArray.shape[0]
        dataNum = dataArray.shape[1]

        X = None
        if centerize: X = self.centerize(dataArray)
        else: X = dataArray

        values = None
        basis = None
        if self.mode == "numpy":
            r = np.transpose(X)@X
            values, projections = np.linalg.eigh(r)
            values = values[::-1]
            projections = projections[:, ::-1]
            basisContainer = []
            for i in range(dataNum):
                v = projections[:,i][:,None]
                basisContainer.append(np.squeeze(dataArray@v) / np.sqrt(values[i]))
            basis = np.transpose(np.array(basisContainer))
        elif self.mode == "cupy":
            r = cp.transpose(X)@X
            values, projections = cp.linalg.eigh(r)
            values = values[::-1]
            projections = projections[:, ::-1]
            basisContainer = []
            for i in range(dataNum):
                v = projections[:,i][:,None]
                basisContainer.append(cp.squeeze(dataArray@v) / cp.sqrt(values[i]))
            basis = cp.transpose(cp.array(basisContainer))

        if isVerbose:
            print("="*20)
            self.printEigenvalues(values)
            print("="*20)

        return [values, basis]
    
    def projection(self, dataArray, centerize=True, isVerbose=False): # after doing PCA, project each data to principal subspace
        
        X = None
        if centerize: X = self.centerize(dataArray)
        else: X = dataArray

        _, basis = self.basicConst(X, centerize=False, isVerbose=isVerbose)

        projections = None
        if self.mode == "numpy": projections = np.transpose(basis)@X
        elif self.mode == "cupy": projections = cp.transpose(basis)@X
        return projections

    def lowcostProjection(self, dataArray, centerize=True, isVerbose=False):

        X = None
        if centerize: X = self.centerize(dataArray)
        else: X = dataArray

        values = None
        projections = None

        if self.mode == "numpy":
            r = np.transpose(X)@X
            values, projections = np.linalg.eigh(r)
            values = values[::-1]
            projections = projections[:, ::-1]
        elif self.mode == "cupy":
            r = cp.transpose(X)@X
            values, projections = cp.linalg.eigh(r)
            values = values[::-1]
            projections = projections[:, ::-1]

        if isVerbose:
            print("="*20)
            self.printEigenvalues(values)
            print("="*20)
        
        return projections

class SPCATool(ConstructTool):
    def __init__(self, mode="numpy"):
        super().__init__(self, mode=mode)

    # it seems we cannot get eigenvalues(variance) with SparsePCA...
    # so I do not define functions about print dimension information.

    def basicConst(self, dataArray, centerize=True, alpha=0.25):
        dimension = dataArray.shape[0]
        dataNum = dataArray.shape[1]

        X = None
        if centerize: X = self.centerize(dataArray)
        else: X = dataArray

        transformer = SparsePCA(n_components=dataNum, alpha=alpha, random_state=0)

        if self.mode == "numpy": X = np.transpose(X)
        elif self.mode == "cupy": X = cp.transpose(X).get()

        transformer.fit(X)
        if self.mode == "numpy": basis = np.transpose(transformer.components_)
        elif self.mode == "cupy": basis = cp.transpose(cp.array(transformer.components_))
        basis = basis[:,::-1]
        
        return [np.zeros(dataNum,), basis]
    
    def projection(self, dataArray, alpha=0.25, centerize=True): # after doing PCA, project each data to principal subspace
        
        X = None
        if centerize: X = self.centerize(dataArray)
        else: X = dataArray

        _, basis = self.basicConst(X, centerize=False, alpha=alpha)

        projections = None
        if self.mode == "numpy": projections = np.transpose(basis)@X
        elif self.mode == "cupy": projections = cp.transpose(basis)@X
        return projections
        
class KPCATool(ConstructTool):
    def __init__(self, mode="numpy"):
        super().__init__(self, mode=mode)

    def centerize(self, K):
        N = K.shape[0]
        if self.mode == "numpy": one_n = np.ones((N, N)) / N
        elif self.mode == "cupy": one_n = cp.ones((N, N)) / N
        centerizedK = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        return centerizedK
    
    def printEigenvalues(self, values):

        values_sum = None
        if self.mode == "numpy": values_sum = np.sum(values)
        elif self.mode == "cupy": values_sum = cp.sum(values)

        print("<1. Top-10 eigenvalues>")
        print("lambda || cumulative sum || proportion of lambda || proportion of cumulative sum")
        sum_cumul = 0
        printCounter = 0
        for i, element in enumerate(values):
            printCounter += 1
            sum_cumul += element
            print("{:04d}. ".format(i), end="")
            print("{:.4f}".format(element), end=" || ")
            print("{:.4f}".format(sum_cumul), end=" || ")
            print("{:.4f}".format(element / values_sum), end=" || ")
            print("{:.4f}".format(sum_cumul / values_sum))
            if printCounter > 9:
                print()
                break

        print("<2. Overall distribution of eigenvalues>")
        print("lambda || cumulative sum || proportion of lambda || proportion of cumulative sum")
        sum_cumul = 0
        printCounter = 0
        period = int(len(values) / 20)
        for i, element in enumerate(values):
            sum_cumul += element
            if i % period == 0:
                print("{:04d}. ".format(i), end="")
                print("{:.4f}".format(element), end=" || ")
                print("{:.4f}".format(sum_cumul), end=" || ")
                print("{:.4f}".format(element / values_sum), end=" || ")
                print("{:.4f}".format(sum_cumul / values_sum))
            if element < 0.1:
                printCounter += 1
            if printCounter > 9:
                print()
                break

        print("<3. Intrinsic Dimension of subspace>")

        index = 0
        changeFlag = False
        for i, l in enumerate(values):
            if l < 1e-4:
                index = i
                changeFlag = True
                break
        if not changeFlag: index = values.shape[0]

        intrinsic_index_90 = 0
        intrinsic_index_95 = 0
        intrinsic_index_99 = 0
        foundFlag_90 = False
        foundFlag_95 = False
        sum_cumul = 0
        for i, element in enumerate(values):
            sum_cumul += element
            if sum_cumul / values_sum > 0.9 and not foundFlag_90:
                intrinsic_index_90 = i
                foundFlag_90 = True
            if sum_cumul / values_sum > 0.95 and not foundFlag_95:
                intrinsic_index_95 = i
                foundFlag_95 = True
            if sum_cumul / values_sum > 0.99:
                intrinsic_index_99 = i
                break
        print("90Percent-dimension : {:04d}".format(intrinsic_index_90))
        print("95Percent-dimension : {:04d}".format(intrinsic_index_95))
        print("99Percent-dimension : {:04d}".format(intrinsic_index_99))
        print("real-dimension : {:04d}".format(index))
        print("full-dimension : {:04d}".format(len(values)))
        print()

    def basicConst(self, K, centerize=True, isVerbose=False):

        _K = None
        if centerize: _K = self.centerize(K)
        else: _K = K

        values = None
        basis = None
        if self.mode == "numpy":
            values, basis = np.linalg.eigh(_K)
            b = np.array([1])
            values_abs = abs(values)
            basisLen = np.concatenate((b, np.sqrt(values_abs[1:])))
            basis = basis / basisLen
        elif self.mode == "cupy":
            values, basis = cp.linalg.eigh(_K)
            b = cp.array([1])
            values_abs = abs(values)
            basisLen = cp.concatenate((b, cp.sqrt(values_abs[1:])))
            basis = basis / basisLen
        
        values = values[::-1]
        basis = basis[:, ::-1]

        if isVerbose:
            print("="*20)
            self.printEigenvalues(values)
            print("="*20)

        return [values, basis]
    
    def projection(self, K, centerize=True, isVerbose=False): # after doing PCA, project each data to principal subspace
        
        _K = None
        if centerize: _K = self.centerize(K)
        else: _K = K

        _, basis = self.basicConst(_K, centerize=False, isVerbose=isVerbose)

        projections = None
        if self.mode == "numpy": projections = np.transpose(basis)@_K
        elif self.mode == "cupy": projections = cp.transpose(basis)@_K
        return projections