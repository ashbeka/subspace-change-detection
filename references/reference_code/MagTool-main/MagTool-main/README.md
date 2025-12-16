# MagTool
Tools for constructing subspace and calculating magnitude between them.

Anyone can easily construct subspace from the data, and calculate magnitude between constructed subspaces!

This code is written based on the two papers, <br> [Difference Subspace and Its Generalization for Subspace-Based Methods](https://ieeexplore.ieee.org/document/7053916) and [Second-order difference subspace](https://arxiv.org/abs/2409.08563).

## Usage
### `construct.py`

This python code contains 4 classes, which have methods that can be used for constructing subspace, or projecting datapoints onto constructed subspace.

Please note that
1. You can give additional argument to constructors, `mode='numpy'` or `mode='cupy'`, to specify which math calculation module you want to use.
2. Your data matrix should contain datapoints as column vectors. <br> Data matrix $X \in \mathbb{R}^{d \times n}$, $d$ as dimension of vector space that data exists, and $n$ as number of datapoints.

Description of each class is like below :
* `QRTool` :
  Construct subspace with QR Decomposition.
  |Method|Args|Detail|
  |-----|-----|-----|
  |`QRTool.basicConst(self, dataArray, centerize=False)`|`dataArray`: 'numpy.ndarray' or 'cupy.ndarray' - The data array which is target of QR decomposition. <br> `centerize`: boolean - Flag for whether applying centerization or not.|execute QR decomposition of given array and return Q matrix, which is orthonormal basis of subspace that data spans|
* `PCATool` :
  Construct subspace with PCA(Principal Component Analysis).
  |Method|Args|Detail|
  |-----|-----|-----|
  |`PCATool.basicConst(self, dataArray, centerize=True, isVerbose=False)`|`dataArray`: 'numpy.ndarray' or 'cupy.ndarray' - The data array which is target of PCA. <br> `centerize`: boolean - Flag for whether applying centerization or not. <br> `isVerbose`: boolean - Flag for whether printing detail of eigenvalues or not.|execute PCA of given array. <br> return \[eigenvalues, orthonormal basis of subspace that data spans\]|
  |`PCATool.lowcostConst(self, dataArray, centerize=True, isVerbose=False)`|`dataArray`: 'numpy.ndarray' or 'cupy.ndarray' - The data array which is target of PCA. <br> `centerize`: boolean - Flag for whether applying centerization or not. <br> `isVerbose`: boolean - Flag for whether printing detail of eigenvalues or not.|execute PCA of given array, especially with low cost by utilizing dual covariance matrix. <br> return \[eigenvalues, orthonormal basis of subspace that data spans\].|
  |`PCATool.projection(self, dataArray, centerize=True, isVerbose=False)`|`dataArray`: 'numpy.ndarray' or 'cupy.ndarray' - The data array which is target of PCA. <br> `centerize`: boolean - Flag for whether applying centerization or not. <br> `isVerbose`: boolean - Flag for whether printing detail of eigenvalues or not.|execute PCA of given array and project datapoints onto the eigenspace calculated just before. <br> return coordination matrix of datapoints after projection.|
  |`PCATool.lowcostProjection(self, dataArray, centerize=True, isVerbose=False)`|`dataArray`: 'numpy.ndarray' or 'cupy.ndarray' - The data array which is target of PCA. <br> `centerize`: boolean - Flag for whether applying centerization or not. <br> `isVerbose`: boolean - Flag for whether printing detail of eigenvalues or not.|execute PCA of given array, especially with low cost by utilizing dual covariance matrix, and project datapoints onto the eigenspace calculated just before. <br> return coordination matrix of datapoints after projection.|
* and so on...

### `magnitude.py`

This python code contains only 1 classes, `MagTool`, which have methods that can be used for calculating Difference/Karcher subspace or magnitude between two subspaces.

Please note that
1. You can give additional argument to constructors, `mode='numpy'` or `mode='cupy'`, to specify which math calculation module you want to use.
2. You can also give additional argument to constructors, `etol`, to specify threshold for which eigenvalues(and corresponding eigenvectors) you want to use.
3. Your orthonormal basis matrix should contain vectors as column vectors. <br> Data matrix $X \in \mathbb{R}^{d \times n}$, $d$ as dimension of vector space that subspace exists, and $n$ as dimension of subspace itself.

|Method|Args|Detail|
|-----|-----|-----|
|`MagTool.calcSumSubspace(self, basis1, basis2, isVerbose=False)`|`basis1`, `basis2`: 'numpy.ndarray' or 'cupy.ndarray' - The two orthonormal basis matrix which is target for calculating sum subspace. <br> `isVerbose`: boolean - Flag for whether printing detail of eigenvalues or not.|calculate sum subspace of two given subspace. <br> return new orthonormal basis matrix that spans sum subspace.|
|`MagTool.calcOverlapSubspace(self, basis1, basis2, isVerbose=False)`|`basis1`, `basis2`: 'numpy.ndarray' or 'cupy.ndarray' - The two orthonormal basis matrix which is target for calculating overlap subspace. <br> `isVerbose`: boolean - Flag for whether printing detail of eigenvalues or not.|calculate overlap subspace of two given subspace. <br> return new orthonormal basis matrix that spans overlap subspace.|
|`MagTool.calcKarcherSubspace(self, basis1, basis2, isVerbose=False)`|`basis1`, `basis2`: 'numpy.ndarray' or 'cupy.ndarray' - The two orthonormal basis matrix which is target for calculating karcher subspace. <br> `isVerbose`: boolean - Flag for whether printing detail of eigenvalues or not.|calculate karcher(mean) subspace of two given subspace. <br> return new orthonormal basis matrix that spans karcher subspace.|
|`MagTool.calcMagnitude(self, basis1, basis2, isVerbose=False)`|`basis1`, `basis2`: 'numpy.ndarray' or 'cupy.ndarray' - The two orthonormal basis matrix which is target for calculating magnitude of difference subspace between them. <br> `isVerbose`: boolean - Flag for whether printing detail of eigenvalues or not.|calculate magnitude of difference subspace of two given subspace.|
|`MagTool.calcSimilarity(self, basis1, basis2, isVerbose=False)`|`basis1`, `basis2`: 'numpy.ndarray' or 'cupy.ndarray' - The two orthonormal basis matrix which is target for calculating similarity between them. <br> `isVerbose`: boolean - Flag for whether printing detail of eigenvalues or not.|calculate similarity(sum of Canonical Correlation Analysis coefficient) between two given subspace.|
|`MagTool.calc1stMagDecomposed(self, basis1, basis2, basis3)`|`basis1`, `basis2`, `basis3`: 'numpy.ndarray' or 'cupy.ndarray' - The three orthonormal basis matrix which is target for calculating magnitude of difference subspace between them.|calculate magnitude of difference subspace, especially as decomposed form($\textnormal{magnitude} = \textnormal{comp along geodesic} + \textnormal{comp orth to geodesic}$) <br> of two given subspace. Please note that you need information of one additional subspace(`basis3`), as we use geodesic between `basis1` and `basis3`. <br> return \[component along geodesic, component orthogonal to geodesic\].|
|`MagTool.calc2ndMagDecomposed(self, basis1, basis2, basis3)`|`basis1`, `basis2`, `basis3`: 'numpy.ndarray' or 'cupy.ndarray' - The three orthonormal basis matrix which is target for calculating magnitude of difference subspace between them.|calculate magnitude of 2nd difference subspace, especially as decomposed form($\textnormal{magnitude} = \textnormal{comp along geodesic} + \textnormal{comp orth to geodesic}$) of three given subspace. <br> return \[component along geodesic, component orthogonal to geodesic\].|
