#include "mex.h"
#include <cstdio>
#include "boost.h"
#include "singlefeaturesigmoidclassifier.h"
#include "singlefeaturemultithresholdedsigmoidlearner.h"

#include "additiveclassifier.h"
#include "utils.h"
#include "weaklearner.h"
#include "classifier.h"

#include <limits>
#include <list>
#include <cstdlib>
#include <cstdio>
#include <list>
#include <vector>

// input:
// parameters from sigboost_mex, test data in single precision
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
   if(!mxIsSingle(prhs[0])) {
	  mexErrMsgTxt("parameters must be single!");
   }
   if(!mxIsSingle(prhs[1])) {
	  mexErrMsgTxt("test data must be single!");
   }
   // reconstruct the classifier
   int M = mxGetM(prhs[0]);
   int N = mxGetN(prhs[0]);
   float* params = static_cast<float*>(mxGetData(prhs[0]));
   float* data_tmp = static_cast<float*>(mxGetData(prhs[1]));
   int dataM = mxGetM(prhs[1]);
   int dataN = mxGetN(prhs[1]);
   float** data = new float*[dataM];
   for(int i = 0; i < dataM; ++i) {
	  data[i] = new float[dataN];
	  for(int j = 0; j < dataN; ++j){
		 data[i][j] = data_tmp[i+j*dataM];
	  }
   }

   std::list< Classifier* > classifiers;
   for(int i = 0; i < M; ++i) {


	  int featColumn = (int)params[i+0*M];
	  float threshold = params[i+1*M];
	  float smoothW = params[i+2*M];
	  float lessConf = params[i+3*M];
	  float grtrConf = params[i+4*M];
	  float dcBias = params[i+5*M];
	  classifiers.push_back(new SingleFeatureSigmoidClassifier(featColumn,
															   threshold,
															   smoothW,
															   lessConf,
															   grtrConf,
															   dcBias));

   }
   AdditiveClassifier *final = new AdditiveClassifier(classifiers);
   mxArray* res_out = mxCreateNumericMatrix(dataM, 1, mxSINGLE_CLASS, mxREAL);
   float* res = static_cast<float*>(mxGetData(res_out));
   final->classify(res, data, dataM, dataN);
   plhs[0] = res_out;
   delete final;
   for(int i =  0; i < dataM; ++i) {
	  delete[] data[i];
   }
   delete[] data;

}
