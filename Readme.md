Programs of farPCA Algorithm [1]
---
### 1.Main Algorithms

1.matlab/exp2/farPCA.m ---- fast adaptive randomized PCA algorithm (Alg. 5 in our paper).

2.matlab/exp1/faster_randQB_EI.m ---- faster fixed-precision QB factorization(Alg. 4 in our paper).

3.mkl/farPCA.c ---- farPCA and randQB_EI algorithm which are implemented in C with MKL and OpenMP.

### 2.Experiments for Testing

(1)The program for testing farPCA with MKL is in "mkl/". The MKL library needs the support of Intel MKL [2]. When all the libraries have been prepared, firstly modified the path of MKL in makefile, and secondly use "make" to produce the executable program "./farpcatest". Before execute "./farpcatest", you should perform "GenImage.m" in Matlab to generate the data for Image. In farpcatest.c, "dense_test" is used for testing Image dataset, while "sparse_test" is used for testing SNAP dataset. You should ensure the space is enough for testing SNAP. Besides, you can use "MKL_NUM_THREADS=X ./farpcatest" to limit the the number of threads to X in MKL.  

(2)matlab/exp1/AccuracyTest.m is used to test the effectiveness of shifted power iteration. The comparison is between Alg. 2 (randQB_EI_auto.m)  [3], Alg. 4  (faster_randQB_EI.m), Alg. 5  (farPCA.m) on Dense1 or Dense2 in size 1000 x 1000.

(3)matlab/exp2/TimeTest.m is used to validate farPCA with svds,  randQB_EI [3] and randUBV [4] on Image dataset.

(4)The singular values computed by eigSVD are in ascending order.

### Reference

[1] Xu Feng, Wenjian Yu. A Fast Adaptive Randomized PCA Algorithm. In Proc. 32nd International Joint Conference on Artificial Intelligence (IJCAI), 2023. (accepted)

[1]  Intel oneAPI Math Kernel Library. https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html, 2021. 

[2] Wenjian Yu, Yu Gu, and Yaohang Li. Efficient randomized algorithms for the fixed-precision low-rank matrix approximation. SIAM Journal on Matrix Analysis and Applications, 39(3):1339–1359, 2018.

[3] Eric Hallman. A block bidiagonalization method for fixed-accuracy low-rank matrix approximation. SIAM Journal on Matrix Analysis and Applications, 43(2):661–680, 2022.