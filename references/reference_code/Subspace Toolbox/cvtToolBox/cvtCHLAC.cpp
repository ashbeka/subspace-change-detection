#include <stdio.h>
#include "mex.h" // Required for the use of MEX files

/* CHLAC特徴抽出用mexファイル
 CHLAC=cvtCHLAC(img,r,rt);
/* input    img-> 画像のセット(double型)
              r-> 格子平面内での変位
             rt-> 時間方向の変位
 
   output CHLAC-> CHLAC特徴(251次元の列ベクトル)(double型)
/* operand[0][0]の正の平方根を returned[0][0]に格納する． */
/* ・mxArrayはMatlab中でのスカラデータ型に行と列の情報を加えたもの(たぶん)
     行列はmxArrayの一次元配列として扱われる．
   ・引数となる行列や戻り値となる行列がそれぞれ一つならば，使用されるのは
     returned[0]，operand[0]だけであり，returned[1]などは使用されない．
   ・NreturnedとNoperandはreturnedなどの何番目までが使用されるかを表す．    */

void SetCHLAC(double *mat, int height, int width, int depth, int r, int rt, double* extract){
    
   #define mat(y,x,z) (mat[(z)*width*height+(x)*height+(y)])
   double tmp[14];
	for(int k=rt;k<depth-rt;k++)
		for(int j=r; j<height-r; j++)
			for(int i=r; i<width-r; i++){
				if(mat(j,i,k)==0) continue;
				tmp[0]=(double)mat(j,i,k);
				extract[217]+=tmp[0]*(double)mat(j,i,k+rt)*(double)mat(j,i+r,k-rt);
				extract[218]+=tmp[0]*(double)mat(j,i,k+rt)*(double)mat(j+r,i-r,k-rt);
				extract[219]+=tmp[0]*(double)mat(j,i,k+rt)*(double)mat(j+r,i,k-rt);
				extract[220]+=tmp[0]*(double)mat(j,i,k+rt)*(double)mat(j+r,i+r,k-rt);
				extract[221]+=tmp[0]*(double)mat(j,i+r,k-rt)*(double)mat(j,i+r,k+rt);
				extract[222]+=tmp[0]*(double)mat(j,i+r,k-rt)*(double)mat(j+r,i-r,k-rt);
				extract[223]+=tmp[0]*(double)mat(j,i+r,k-rt)*(double)mat(j+r,i-r,k);
				extract[224]+=tmp[0]*(double)mat(j,i+r,k-rt)*(double)mat(j+r,i-r,k+rt);
				extract[225]+=tmp[0]*(double)mat(j,i+r,k-rt)*(double)mat(j+r,i,k+rt);
				extract[226]+=tmp[0]*(double)mat(j,i+r,k-rt)*(double)mat(j+r,i+r,k+rt);
				extract[227]+=tmp[0]*(double)mat(j,i+r,k)*(double)mat(j+r,i-r,k-rt);
				extract[228]+=tmp[0]*(double)mat(j,i+r,k)*(double)mat(j+r,i-r,k);
				extract[229]+=tmp[0]*(double)mat(j,i+r,k)*(double)mat(j+r,i-r,k+rt);
				extract[230]+=tmp[0]*(double)mat(j,i+r,k+rt)*(double)mat(j+r,i-r,k-rt);
				extract[231]+=tmp[0]*(double)mat(j,i+r,k+rt)*(double)mat(j+r,i-r,k);
				extract[232]+=tmp[0]*(double)mat(j,i+r,k+rt)*(double)mat(j+r,i-r,k+rt);
				extract[233]+=tmp[0]*(double)mat(j,i+r,k+rt)*(double)mat(j+r,i,k-rt);
				extract[234]+=tmp[0]*(double)mat(j,i+r,k+rt)*(double)mat(j+r,i+r,k-rt);
				extract[235]+=tmp[0]*(double)mat(j+r,i-r,k-rt)*(double)mat(j+r,i-r,k+rt);
				extract[236]+=tmp[0]*(double)mat(j+r,i-r,k-rt)*(double)mat(j+r,i,k+rt);
				extract[237]+=tmp[0]*(double)mat(j+r,i-r,k-rt)*(double)mat(j+r,i+r,k-rt);
				extract[238]+=tmp[0]*(double)mat(j+r,i-r,k-rt)*(double)mat(j+r,i+r,k);
				extract[239]+=tmp[0]*(double)mat(j+r,i-r,k-rt)*(double)mat(j+r,i+r,k+rt);
				extract[240]+=tmp[0]*(double)mat(j+r,i-r,k)*(double)mat(j+r,i+r,k-rt);
				extract[241]+=tmp[0]*(double)mat(j+r,i-r,k)*(double)mat(j+r,i+r,k);
				extract[242]+=tmp[0]*(double)mat(j+r,i-r,k)*(double)mat(j+r,i+r,k+rt);
				extract[243]+=tmp[0]*(double)mat(j+r,i-r,k+rt)*(double)mat(j+r,i,k-rt);
				extract[244]+=tmp[0]*(double)mat(j+r,i-r,k+rt)*(double)mat(j+r,i+r,k-rt);
				extract[245]+=tmp[0]*(double)mat(j+r,i-r,k+rt)*(double)mat(j+r,i+r,k);
				extract[246]+=tmp[0]*(double)mat(j+r,i-r,k+rt)*(double)mat(j+r,i+r,k+rt);
				extract[247]+=tmp[0]*(double)mat(j+r,i,k-rt)*(double)mat(j+r,i,k+rt);
				extract[248]+=tmp[0]*(double)mat(j+r,i,k-rt)*(double)mat(j+r,i+r,k+rt);
				extract[249]+=tmp[0]*(double)mat(j+r,i,k+rt)*(double)mat(j+r,i+r,k-rt);
				extract[250]+=tmp[0]*(double)mat(j+r,i+r,k-rt)*(double)mat(j+r,i+r,k+rt);
				extract[0]+=tmp[0];

				tmp[1]=(double)mat(j-r,i-r,k-rt)*(double)mat(j,i,k);
				if(tmp[1]>1e-9){
					extract[26]+=tmp[1]*(double)mat(j,i,k+rt);
					extract[27]+=tmp[1]*(double)mat(j,i+r,k-rt);
					extract[28]+=tmp[1]*(double)mat(j,i+r,k);
					extract[29]+=tmp[1]*(double)mat(j,i+r,k+rt);
					extract[30]+=tmp[1]*(double)mat(j+r,i-r,k-rt);
					extract[31]+=tmp[1]*(double)mat(j+r,i-r,k);
					extract[32]+=tmp[1]*(double)mat(j+r,i-r,k+rt);
					extract[33]+=tmp[1]*(double)mat(j+r,i,k-rt);
					extract[34]+=tmp[1]*(double)mat(j+r,i,k);
					extract[35]+=tmp[1]*(double)mat(j+r,i,k+rt);
					extract[36]+=tmp[1]*(double)mat(j+r,i+r,k-rt);
					extract[37]+=tmp[1]*(double)mat(j+r,i+r,k);
					extract[38]+=tmp[1]*(double)mat(j+r,i+r,k+rt);
					extract[1]+=tmp[1];
				}
				tmp[2]=(double)mat(j-r,i-r,k)*(double)mat(j,i,k);
				if(tmp[2]>1e-9){
					extract[50]+=tmp[2]*(double)mat(j,i+r,k-rt);
					extract[51]+=tmp[2]*(double)mat(j,i+r,k);
					extract[52]+=tmp[2]*(double)mat(j,i+r,k+rt);
					extract[53]+=tmp[2]*(double)mat(j+r,i-r,k-rt);
					extract[54]+=tmp[2]*(double)mat(j+r,i-r,k);
					extract[55]+=tmp[2]*(double)mat(j+r,i-r,k+rt);
					extract[56]+=tmp[2]*(double)mat(j+r,i,k-rt);
					extract[57]+=tmp[2]*(double)mat(j+r,i,k);
					extract[58]+=tmp[2]*(double)mat(j+r,i,k+rt);
					extract[59]+=tmp[2]*(double)mat(j+r,i+r,k-rt);
					extract[60]+=tmp[2]*(double)mat(j+r,i+r,k);
					extract[61]+=tmp[2]*(double)mat(j+r,i+r,k+rt);
					extract[2]+=tmp[2];
					extract[14]+=(double)mat(j-r,i-r,k-rt)*tmp[2];
				}
				tmp[3]=(double)mat(j-r,i-r,k+rt)*(double)mat(j,i,k);
				if(tmp[3]>1e-9){
					extract[72]+=tmp[3]*(double)mat(j,i+r,k-rt);
					extract[73]+=tmp[3]*(double)mat(j,i+r,k);
					extract[74]+=tmp[3]*(double)mat(j,i+r,k+rt);
					extract[75]+=tmp[3]*(double)mat(j+r,i-r,k-rt);
					extract[76]+=tmp[3]*(double)mat(j+r,i-r,k);
					extract[77]+=tmp[3]*(double)mat(j+r,i-r,k+rt);
					extract[78]+=tmp[3]*(double)mat(j+r,i,k-rt);
					extract[79]+=tmp[3]*(double)mat(j+r,i,k);
					extract[80]+=tmp[3]*(double)mat(j+r,i,k+rt);
					extract[81]+=tmp[3]*(double)mat(j+r,i+r,k-rt);
					extract[82]+=tmp[3]*(double)mat(j+r,i+r,k);
					extract[83]+=tmp[3]*(double)mat(j+r,i+r,k+rt);
					extract[3]+=tmp[3];
					extract[15]+=(double)mat(j-r,i-r,k-rt)*tmp[3];
					extract[39]+=(double)mat(j-r,i-r,k)*tmp[3];
				}
				tmp[4]=(double)mat(j-r,i,k-rt)*(double)mat(j,i,k);
				if(tmp[4]>1e-9){
					extract[93]+=tmp[4]*(double)mat(j,i,k+rt);
					extract[94]+=tmp[4]*(double)mat(j,i+r,k+rt);
					extract[95]+=tmp[4]*(double)mat(j+r,i-r,k-rt);
					extract[96]+=tmp[4]*(double)mat(j+r,i-r,k);
					extract[97]+=tmp[4]*(double)mat(j+r,i-r,k+rt);
					extract[98]+=tmp[4]*(double)mat(j+r,i,k-rt);
					extract[99]+=tmp[4]*(double)mat(j+r,i,k);
					extract[100]+=tmp[4]*(double)mat(j+r,i,k+rt);
					extract[101]+=tmp[4]*(double)mat(j+r,i+r,k-rt);
					extract[102]+=tmp[4]*(double)mat(j+r,i+r,k);
					extract[103]+=tmp[4]*(double)mat(j+r,i+r,k+rt);
					extract[4]+=tmp[4];
					extract[16]+=(double)mat(j-r,i-r,k-rt)*tmp[4];
					extract[40]+=(double)mat(j-r,i-r,k)*tmp[4];
					extract[62]+=(double)mat(j-r,i-r,k+rt)*tmp[4];
				}
				tmp[5]=(double)mat(j-r,i,k)*(double)mat(j,i,k);
				if(tmp[5]>1e-9){
					extract[112]+=tmp[5]*(double)mat(j+r,i-r,k-rt);
					extract[113]+=tmp[5]*(double)mat(j+r,i-r,k);
					extract[114]+=tmp[5]*(double)mat(j+r,i-r,k+rt);
					extract[115]+=tmp[5]*(double)mat(j+r,i,k-rt);
					extract[116]+=tmp[5]*(double)mat(j+r,i,k);
					extract[117]+=tmp[5]*(double)mat(j+r,i,k+rt);
					extract[118]+=tmp[5]*(double)mat(j+r,i+r,k-rt);
					extract[119]+=tmp[5]*(double)mat(j+r,i+r,k);
					extract[120]+=tmp[5]*(double)mat(j+r,i+r,k+rt);
					extract[5]+=tmp[5];
					extract[17]+=(double)mat(j-r,i-r,k-rt)*tmp[5];
					extract[41]+=(double)mat(j-r,i-r,k)*tmp[5];
					extract[63]+=(double)mat(j-r,i-r,k+rt)*tmp[5];
					extract[84]+=(double)mat(j-r,i,k-rt)*tmp[5];
				}
				tmp[6]=(double)mat(j-r,i,k+rt)*(double)mat(j,i,k);
				if(tmp[6]>1e-9){
					extract[128]+=tmp[6]*(double)mat(j,i+r,k-rt);
					extract[129]+=tmp[6]*(double)mat(j+r,i-r,k-rt);
					extract[130]+=tmp[6]*(double)mat(j+r,i-r,k);
					extract[131]+=tmp[6]*(double)mat(j+r,i-r,k+rt);
					extract[132]+=tmp[6]*(double)mat(j+r,i,k-rt);
					extract[133]+=tmp[6]*(double)mat(j+r,i,k);
					extract[134]+=tmp[6]*(double)mat(j+r,i,k+rt);
					extract[135]+=tmp[6]*(double)mat(j+r,i+r,k-rt);
					extract[136]+=tmp[6]*(double)mat(j+r,i+r,k);
					extract[137]+=tmp[6]*(double)mat(j+r,i+r,k+rt);
					extract[6]+=tmp[6];
					extract[18]+=(double)mat(j-r,i-r,k-rt)*tmp[6];
					extract[42]+=(double)mat(j-r,i-r,k)*tmp[6];
					extract[64]+=(double)mat(j-r,i-r,k+rt)*tmp[6];
					extract[85]+=(double)mat(j-r,i,k-rt)*tmp[6];
					extract[104]+=(double)mat(j-r,i,k)*tmp[6];
				}
				tmp[7]=(double)mat(j-r,i+r,k-rt)*(double)mat(j,i,k);
				if(tmp[7]>1e-9){
					extract[144]+=tmp[7]*(double)mat(j,i,k+rt);
					extract[145]+=tmp[7]*(double)mat(j,i+r,k+rt);
					extract[146]+=tmp[7]*(double)mat(j+r,i-r,k-rt);
					extract[147]+=tmp[7]*(double)mat(j+r,i-r,k);
					extract[148]+=tmp[7]*(double)mat(j+r,i-r,k+rt);
					extract[149]+=tmp[7]*(double)mat(j+r,i,k-rt);
					extract[150]+=tmp[7]*(double)mat(j+r,i,k);
					extract[151]+=tmp[7]*(double)mat(j+r,i,k+rt);
					extract[152]+=tmp[7]*(double)mat(j+r,i+r,k-rt);
					extract[153]+=tmp[7]*(double)mat(j+r,i+r,k);
					extract[154]+=tmp[7]*(double)mat(j+r,i+r,k+rt);
					extract[7]+=tmp[7];
					extract[19]+=(double)mat(j-r,i-r,k-rt)*tmp[7];
					extract[43]+=(double)mat(j-r,i-r,k)*tmp[7];
					extract[65]+=(double)mat(j-r,i-r,k+rt)*tmp[7];
					extract[86]+=(double)mat(j-r,i,k-rt)*tmp[7];
					extract[105]+=(double)mat(j-r,i,k)*tmp[7];
					extract[121]+=(double)mat(j-r,i,k+rt)*tmp[7];
				}
				tmp[8]=(double)mat(j-r,i+r,k)*(double)mat(j,i,k);
				if(tmp[8]>1e-9){
					extract[160]+=tmp[8]*(double)mat(j+r,i-r,k-rt);
					extract[161]+=tmp[8]*(double)mat(j+r,i-r,k);
					extract[162]+=tmp[8]*(double)mat(j+r,i-r,k+rt);
					extract[163]+=tmp[8]*(double)mat(j+r,i,k-rt);
					extract[164]+=tmp[8]*(double)mat(j+r,i,k);
					extract[165]+=tmp[8]*(double)mat(j+r,i,k+rt);
					extract[166]+=tmp[8]*(double)mat(j+r,i+r,k-rt);
					extract[167]+=tmp[8]*(double)mat(j+r,i+r,k);
					extract[168]+=tmp[8]*(double)mat(j+r,i+r,k+rt);
					extract[8]+=tmp[8];
					extract[20]+=(double)mat(j-r,i-r,k-rt)*tmp[8];
					extract[44]+=(double)mat(j-r,i-r,k)*tmp[8];
					extract[66]+=(double)mat(j-r,i-r,k+rt)*tmp[8];
					extract[87]+=(double)mat(j-r,i,k-rt)*tmp[8];
					extract[106]+=(double)mat(j-r,i,k)*tmp[8];
					extract[122]+=(double)mat(j-r,i,k+rt)*tmp[8];
					extract[138]+=(double)mat(j-r,i+r,k-rt)*tmp[8];
				}
				tmp[9]=(double)mat(j-r,i+r,k+rt)*(double)mat(j,i,k);
				if(tmp[9]>1e-9){
					extract[173]+=tmp[9]*(double)mat(j,i+r,k-rt);
					extract[174]+=tmp[9]*(double)mat(j+r,i-r,k-rt);
					extract[175]+=tmp[9]*(double)mat(j+r,i-r,k);
					extract[176]+=tmp[9]*(double)mat(j+r,i-r,k+rt);
					extract[177]+=tmp[9]*(double)mat(j+r,i,k-rt);
					extract[178]+=tmp[9]*(double)mat(j+r,i,k);
					extract[179]+=tmp[9]*(double)mat(j+r,i,k+rt);
					extract[180]+=tmp[9]*(double)mat(j+r,i+r,k-rt);
					extract[181]+=tmp[9]*(double)mat(j+r,i+r,k);
					extract[182]+=tmp[9]*(double)mat(j+r,i+r,k+rt);
					extract[9]+=tmp[9];
					extract[21]+=(double)mat(j-r,i-r,k-rt)*tmp[9];
					extract[45]+=(double)mat(j-r,i-r,k)*tmp[9];
					extract[67]+=(double)mat(j-r,i-r,k+rt)*tmp[9];
					extract[88]+=(double)mat(j-r,i,k-rt)*tmp[9];
					extract[107]+=(double)mat(j-r,i,k)*tmp[9];
					extract[123]+=(double)mat(j-r,i,k+rt)*tmp[9];
					extract[139]+=(double)mat(j-r,i+r,k-rt)*tmp[9];
					extract[155]+=(double)mat(j-r,i+r,k)*tmp[9];
				}
				tmp[10]=(double)mat(j,i-r,k-rt)*(double)mat(j,i,k);
				if(tmp[10]>1e-9){
					extract[186]+=tmp[10]*(double)mat(j,i,k+rt);
					extract[187]+=tmp[10]*(double)mat(j,i+r,k-rt);
					extract[188]+=tmp[10]*(double)mat(j,i+r,k);
					extract[189]+=tmp[10]*(double)mat(j,i+r,k+rt);
					extract[190]+=tmp[10]*(double)mat(j+r,i-r,k+rt);
					extract[191]+=tmp[10]*(double)mat(j+r,i,k+rt);
					extract[192]+=tmp[10]*(double)mat(j+r,i+r,k-rt);
					extract[193]+=tmp[10]*(double)mat(j+r,i+r,k);
					extract[194]+=tmp[10]*(double)mat(j+r,i+r,k+rt);
					extract[10]+=tmp[10];
					extract[22]+=(double)mat(j-r,i-r,k-rt)*tmp[10];
					extract[46]+=(double)mat(j-r,i-r,k)*tmp[10];
					extract[68]+=(double)mat(j-r,i-r,k+rt)*tmp[10];
					extract[89]+=(double)mat(j-r,i,k-rt)*tmp[10];
					extract[108]+=(double)mat(j-r,i,k)*tmp[10];
					extract[124]+=(double)mat(j-r,i,k+rt)*tmp[10];
					extract[140]+=(double)mat(j-r,i+r,k-rt)*tmp[10];
					extract[156]+=(double)mat(j-r,i+r,k)*tmp[10];
					extract[169]+=(double)mat(j-r,i+r,k+rt)*tmp[10];
				}
				tmp[11]=(double)mat(j,i-r,k)*(double)mat(j,i,k);
				if(tmp[11]>1e-9){
					extract[197]+=tmp[11]*(double)mat(j,i+r,k-rt);
					extract[198]+=tmp[11]*(double)mat(j,i+r,k);
					extract[199]+=tmp[11]*(double)mat(j,i+r,k+rt);
					extract[200]+=tmp[11]*(double)mat(j+r,i+r,k-rt);
					extract[201]+=tmp[11]*(double)mat(j+r,i+r,k);
					extract[202]+=tmp[11]*(double)mat(j+r,i+r,k+rt);
					extract[11]+=tmp[11];
					extract[23]+=(double)mat(j-r,i-r,k-rt)*tmp[11];
					extract[47]+=(double)mat(j-r,i-r,k)*tmp[11];
					extract[69]+=(double)mat(j-r,i-r,k+rt)*tmp[11];
					extract[90]+=(double)mat(j-r,i,k-rt)*tmp[11];
					extract[109]+=(double)mat(j-r,i,k)*tmp[11];
					extract[125]+=(double)mat(j-r,i,k+rt)*tmp[11];
					extract[141]+=(double)mat(j-r,i+r,k-rt)*tmp[11];
					extract[157]+=(double)mat(j-r,i+r,k)*tmp[11];
					extract[170]+=(double)mat(j-r,i+r,k+rt)*tmp[11];
					extract[183]+=(double)mat(j,i-r,k-rt)*tmp[11];
				}
				tmp[12]=(double)mat(j,i-r,k+rt)*(double)mat(j,i,k);
				if(tmp[12]>1e-9){
					extract[204]+=tmp[12]*(double)mat(j,i+r,k-rt);
					extract[205]+=tmp[12]*(double)mat(j,i+r,k);
					extract[206]+=tmp[12]*(double)mat(j,i+r,k+rt);
					extract[207]+=tmp[12]*(double)mat(j+r,i-r,k-rt);
					extract[208]+=tmp[12]*(double)mat(j+r,i,k-rt);
					extract[209]+=tmp[12]*(double)mat(j+r,i+r,k-rt);
					extract[210]+=tmp[12]*(double)mat(j+r,i+r,k);
					extract[211]+=tmp[12]*(double)mat(j+r,i+r,k+rt);
					extract[12]+=tmp[12];
					extract[24]+=(double)mat(j-r,i-r,k-rt)*tmp[12];
					extract[48]+=(double)mat(j-r,i-r,k)*tmp[12];
					extract[70]+=(double)mat(j-r,i-r,k+rt)*tmp[12];
					extract[91]+=(double)mat(j-r,i,k-rt)*tmp[12];
					extract[110]+=(double)mat(j-r,i,k)*tmp[12];
					extract[126]+=(double)mat(j-r,i,k+rt)*tmp[12];
					extract[142]+=(double)mat(j-r,i+r,k-rt)*tmp[12];
					extract[158]+=(double)mat(j-r,i+r,k)*tmp[12];
					extract[171]+=(double)mat(j-r,i+r,k+rt)*tmp[12];
					extract[184]+=(double)mat(j,i-r,k-rt)*tmp[12];
					extract[195]+=(double)mat(j,i-r,k)*tmp[12];
				}
				tmp[13]=(double)mat(j,i,k-rt)*(double)mat(j,i,k);
				if(tmp[13]>1e-9){
					extract[212]+=tmp[13]*(double)mat(j,i,k+rt);
					extract[213]+=tmp[13]*(double)mat(j,i+r,k+rt);
					extract[214]+=tmp[13]*(double)mat(j+r,i-r,k+rt);
					extract[215]+=tmp[13]*(double)mat(j+r,i,k+rt);
					extract[216]+=tmp[13]*(double)mat(j+r,i+r,k+rt);
					extract[13]+=tmp[13];
					extract[25]+=(double)mat(j-r,i-r,k-rt)*tmp[13];
					extract[49]+=(double)mat(j-r,i-r,k)*tmp[13];
					extract[71]+=(double)mat(j-r,i-r,k+rt)*tmp[13];
					extract[92]+=(double)mat(j-r,i,k-rt)*tmp[13];
					extract[111]+=(double)mat(j-r,i,k)*tmp[13];
					extract[127]+=(double)mat(j-r,i,k+rt)*tmp[13];
					extract[143]+=(double)mat(j-r,i+r,k-rt)*tmp[13];
					extract[159]+=(double)mat(j-r,i+r,k)*tmp[13];
					extract[172]+=(double)mat(j-r,i+r,k+rt)*tmp[13];
					extract[185]+=(double)mat(j,i-r,k-rt)*tmp[13];
					extract[196]+=(double)mat(j,i-r,k)*tmp[13];
					extract[203]+=(double)mat(j,i-r,k+rt)*tmp[13];
				}
			}

#undef mat(y,x,z)
}



void mexFunction( int Nreturned, mxArray *returned[], int Noperand, const mxArray *operand[] ){
  double *x,*y;
  double *mat;
  double *extract;
  int rows,cols;
  int ChlacDim = 251;
  int r,rt;

  const size_t *Dim = mxGetDimensions(*operand);
  
   /* Matlab側に渡されるデータ(戻り値みたいなもの)を作る．C言語のmalloc()みたいなもの．        */
  /* Matlabの基本データ型は行列なので，行の数(rows)，列の数(cols)，要素が何か(mxREAL)を伝える */
  *returned = mxCreateDoubleMatrix(ChlacDim,1, mxREAL);
  
  
  /* Matlab側の変数のアドレスをC側のポインタにコピーする */
  mat = mxGetPr(operand[0]);
  extract = mxGetPr(*returned); 
  r = (int)(*mxGetPr(operand[1]));
  rt= (int)(*mxGetPr(operand[2]));

  
  //エラー処理
  if (mxGetNumberOfDimensions(*operand) != 3){
      printf("Error : The size of Matrix is incorrect\n");
     return;
  }
  if(r  < 1 ){
     printf("Error : r is incorrect\n");
     return;
  }
  if(rt < 1 ){
     printf("Error : rt is incorrect\n"); 
     return;
  }
  
  if (mxIsUint8(operand[0])){
      printf("Error : img type must be double\n");
      return;
  }
  
  //printf("%d\n%d\n%d\n",Dim[0],Dim[1],Dim[2]);
  //printf("\n%d\n%d\n",r,rt);
  
  /* Matlabから呼び出したい関数 mysqrt() を呼び出す */
   SetCHLAC(mat, Dim[0], Dim[1], Dim[2], r,rt, extract);
   
}
