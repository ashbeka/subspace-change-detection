import cupy as cp
import numpy as np

class MagTool():

    def __init__(self, mode="numpy", etol=1e-6):
        self.mode = mode
        self.etol = etol

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

    def calcSumSubspace(self, basis1, basis2, isVerbose=False):

        values = None
        basis = None
        if self.mode == "numpy":
            G = basis1@np.transpose(basis1) + basis2@np.transpose(basis2)
            values, basis = np.linalg.eigh(G)
        elif self.mode == "cupy":
            G = basis1@cp.transpose(basis1) + basis2@cp.transpose(basis2)
            values, basis = cp.linalg.eigh(G)
        values = values[::-1]
        basis = basis[:, ::-1]
        values, basis = self.adjustEig(values, basis)

        index = 0
        for j, element in enumerate(values):
            if element < 0.0:
                index = j
                break

        if isVerbose: print("Dimesion of Sum Subspace : {:04d}".format(index))

        return [values[0:index], basis[:,0:index]]

    def calcOverlapSubspace(self, basis1, basis2, isVerbose=False):

        values = None
        basis = None
        if self.mode == "numpy":
            G = basis1@np.transpose(basis1) + basis2@np.transpose(basis2)
            values, basis = np.linalg.eigh(G)
        elif self.mode == "cupy":
            G = basis1@cp.transpose(basis1) + basis2@cp.transpose(basis2)
            values, basis = cp.linalg.eigh(G)
        values = values[::-1]
        basis = basis[:, ::-1]
        values, basis = self.adjustEig(values, basis)

        index = 0
        for i, a in enumerate(values):
            if a < 2.0:
                index = i
                break

        if isVerbose: print("Dimesion of Overlapped Subspace : {:04d}".format(index))

        return [values[0:index], basis[:,0:index]]
    
    def calcKarcherSubspace(self, basis1, basis2, isVerbose=False):

        values = None
        basis = None
        if self.mode == "numpy":
            G = basis1@np.transpose(basis1) + basis2@np.transpose(basis2)
            values, basis = np.linalg.eigh(G)
        elif self.mode == "cupy":
            G = basis1@cp.transpose(basis1) + basis2@cp.transpose(basis2)
            values, basis = cp.linalg.eigh(G)
        values = values[::-1]
        basis = basis[:, ::-1]
        values, basis = self.adjustEig(values, basis)

        index = 0
        for i, a in enumerate(values):
            if a < 1.0:
                index = i
                break
        
        if isVerbose: print("Dimesion of Karcher Subspace : {:04d}".format(index))

        return [values[0:index], basis[:,0:index]]
    
    def calcDiffSubspace(self, basis1, basis2, isVerbose=False):

        values = None
        basis = None
        if self.mode == "numpy":
            G = basis1@np.transpose(basis1) + basis2@np.transpose(basis2)
            values, basis = np.linalg.eigh(G)
        elif self.mode == "cupy":
            G = basis1@cp.transpose(basis1) + basis2@cp.transpose(basis2)
            values, basis = cp.linalg.eigh(G)
        values = values[::-1]
        basis = basis[:, ::-1]
        values, basis = self.adjustEig(values, basis)

        index_start = 0
        for i, a in enumerate(values):
            if a < 1.0:
                index_start = i
                break

        index_end = 0
        for i, a in enumerate(values):
            if a < 0:
                index_end = i
                break

        if isVerbose: print("Dimesion of Difference Subspace : {:04d}".format(index_end - index_start))

        return [values[index_start:index_end], basis[:,index_start:index_end]]
    
    def adjustEig(self, eigenvalues, eigenvectors):
        eig_num = eigenvalues.shape[0]

        eigenvalues_new = []
        eigenvectors_new = []

        for i in range(eig_num):
            value = eigenvalues[i]
            if value >= 2.0 or (value < 2.0 and value > 2.0 - self.etol): # dims overlapped
                eigenvalues_new.append(2.0)
                eigenvectors_new.append(eigenvectors[:,i])
            elif (value < self.etol and value > 0.0) or value <= 0.0: # dims cannot express
                eigenvalues_new.append(0.0)
                eigenvectors_new.append(eigenvectors[:,i])
            else:
                eigenvalues_new.append(float(value) if self.mode == "numpy" else float(value.get()))
                eigenvectors_new.append(eigenvectors[:,i])

        if self.mode == "numpy": return [np.array(eigenvalues_new), np.transpose(np.array(eigenvectors_new))]
        elif self.mode == "cupy": return [cp.array(eigenvalues_new), cp.transpose(cp.array(eigenvectors_new))]

    def calcMagnitude(self, basis1, basis2, isVerbose=False):

        if self.mode == "numpy": 
            G = np.transpose(basis1)@basis2
            _, s, _ = np.linalg.svd(G)
        elif self.mode == "cupy":
            G = cp.transpose(basis1)@basis2
            _, s, _ = cp.linalg.svd(G)

        if isVerbose:
            print("="*20)
            self.printEigenvalues(s)
            print("="*20)

        if self.mode == "numpy": return 2 * (len(s) - np.sum(s))
        elif self.mode == "cupy": return 2 * (len(s) - cp.sum(s))
    
    def calcSimilarity(self, basis1, basis2, isVerbose=False):
        
        if self.mode == "numpy": 
            G = np.transpose(basis1)@basis2
            _, s, _ = np.linalg.svd(G)
        elif self.mode == "cupy":
            G = cp.transpose(basis1)@basis2
            _, s, _ = cp.linalg.svd(G)

        if isVerbose:
            print("="*20)
            self.printEigenvalues(s)
            print("="*20)

        if self.mode == "numpy": return np.sum(s)
        elif self.mode == "cupy": return cp.sum(s)
    
    def calc1stMagDecomposed(self, basis1, basis2, basis3):

        mag_orth = None
        mag_along = None
        if self.mode == "numpy":
            X = np.concatenate((basis1, basis3), axis=1)
            values_W, W = np.linalg.eigh(X@np.transpose(X))
            values_W = values_W[::-1]
            W = W[:,::-1]
            idx = 0
            for i, value in enumerate(values_W):
                if value < self.etol:
                    idx = i
                    break
            W = W[:,:idx]
            _, s, _ = np.linalg.svd(np.transpose(W)@basis2)
            U, _ = np.linalg.qr(np.transpose(W)@basis2) # projection to geodesic
            basis2_projected = W@U

            mag_orth = 2 * (len(s) - np.sum(s))
            mag_along = self.calcMagnitude(basis2_projected, basis1)
        elif self.mode == "cupy":
            X = cp.concatenate((basis1, basis3), axis=1)
            values_W, W = cp.linalg.eigh(X@cp.transpose(X))
            values_W = values_W[::-1]
            W = W[:,::-1]
            idx = 0
            for i, value in enumerate(values_W):
                if value < self.etol:
                    idx = i
                    break
            W = W[:,:idx]
            _, s, _ = cp.linalg.svd(cp.transpose(W)@basis2)
            U, _ = cp.linalg.qr(cp.transpose(W)@basis2) # projection to geodesic
            basis2_projected = W@U

            mag_orth = 2 * (len(s) - cp.sum(s))
            mag_along = self.calcMagnitude(basis2_projected, basis1)

        return [mag_along, mag_orth]

    def calc2ndMagDecomposed(self, basis1, basis2, basis3):

        mag_orth = None
        mag_along = None
        if self.mode == "numpy":
            X = np.concatenate((basis1, basis3), axis=1)
            values_W, W = np.linalg.eigh(X@np.transpose(X))
            values_W = values_W[::-1]
            W = W[:,::-1]
            idx = 0
            for i, value in enumerate(values_W):
                if value < self.etol:
                    idx = i
                    break
            W = W[:,:idx]
            _, M = self.calcKarcherSubspace(basis1, basis3)
            _, s, _ = np.linalg.svd(np.transpose(W)@basis2)
            U, _ = np.linalg.qr(np.transpose(W)@basis2) # projection to geodesic
            basis2_projected = W@U

            mag_orth = 2 * (len(s) - np.sum(s))
            mag_along = self.calcMagnitude(basis2_projected, M)
        elif self.mode == "cupy":
            X = cp.concatenate((basis1, basis3), axis=1)
            values_W, W = cp.linalg.eigh(X@cp.transpose(X))
            values_W = values_W[::-1]
            W = W[:,::-1]
            idx = 0
            for i, value in enumerate(values_W):
                if value < self.etol:
                    idx = i
                    break
            W = W[:,:idx]
            _, M = self.calcKarcherSubspace(basis1, basis3)
            _, s, _ = cp.linalg.svd(cp.transpose(W)@basis2)
            U, _ = cp.linalg.qr(cp.transpose(W)@basis2) # projection to geodesic
            basis2_projected = W@U

            mag_orth = 2 * (len(s) - cp.sum(s))
            mag_along = self.calcMagnitude(basis2_projected, M)

        return [mag_along, mag_orth]
