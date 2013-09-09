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
#include <cstdlib>
#include <cstdio>
#include <list>
#include <vector>

using std::list;
using std::vector;

Classifier* run_boost(float** data, int* labels, int M, int N, vector< vector<float> > threshes, float* smooth_wts, int numIter) {
   vector< WeakLearner * > learners;

   for(int i = 0; i < N; ++i) {
	  learners.push_back(new SingleFeatureMultiThresholdedSigmoidLearner(i, threshes[i], smooth_wts[i]));
   }

   Classifier* c = boosting::train(data, labels, M, N, learners, numIter);


   for(std::vector<WeakLearner* >::iterator it = learners.begin(); it != learners.end(); ++it) {
	  delete (*it);
   }
   return c;
}


// nlhs: num left hand side
// plhs: array of left hand side arrays.. etc.
// INPUT: data, labels, thresholds-for-each-column, smoothing weight for each column, maxIterations
// data: one example per row (M by N)
// labels: {1, -1}, length M

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   if (nrhs != 5) {
	  mexErrMsgTxt("Five input arguments reqiured!");
   } else if (nlhs > 1 ) {
	  mexErrMsgTxt("Too many output arguments!");
   }

   if (!mxIsSingle(prhs[0])) {
	  mexErrMsgTxt("Training data must be Single!");
   }
   int M = mxGetM(prhs[0]); // num rows
   int N = mxGetN(prhs[0]); // num cols
   printf("Examples: %d, Features: %d\n", M, N);
   if (!mxIsInt32(prhs[1])) {
	  mexErrMsgTxt("Training labels must be int32!");
   }

   if (!mxIsCell(prhs[2])) {
	  mexErrMsgTxt("Threshold points must be passed in cell array!");
   }
   int numThreshSets = mxGetNumberOfElements(prhs[2]);
   if( numThreshSets != N) {
	  mexErrMsgTxt("Number of Threshold sets must match number of features!");
   }
   if(mxGetNumberOfElements(prhs[3]) != numThreshSets) {
	  mexErrMsgTxt("Number of smoothing weights must match number of thresh sets!");
   }
   if (!mxIsSingle(prhs[3])) {
	  mexErrMsgTxt("Smoothing weights must be single!");
   }

   if(mxGetM(prhs[1]) != M) {
	  mexErrMsgTxt("Labels length not matching data length!");
   }

   void* data_vd = mxGetData(prhs[0]);
   float* data_tmp = static_cast<float*>(data_vd);
   void* labels_vd = mxGetData(prhs[1]);
   int* labels = static_cast<int*>(labels_vd);
   vector< vector< float > > threshSets;

   for(int i = 0; i < numThreshSets; ++i ) {
	  if(!mxIsSingle(mxGetCell(prhs[2],i))) {
		 mexErrMsgTxt("Thresh value not single!");
	  }
	  float* ts = static_cast<float*>(mxGetData(mxGetCell(prhs[2],i)));
	  int numElts = mxGetNumberOfElements(mxGetCell(prhs[2],i));
	  vector< float > threshes;
	  for(int j = 0; j < numElts; ++j) {
		 threshes.push_back (ts[j]);
	  }
	  threshSets.push_back(threshes);
   }

   void* smooth_wts_vd = mxGetData(prhs[3]);
   float* smooth_wts = static_cast<float*>(smooth_wts_vd);
   int numIters = (int)mxGetScalar(prhs[4]);

   // reshaping the data matrix
   float** data = new float*[M];
   for(int i = 0; i < M; ++i) {
	  data[i] = new float[N];
	  for(int j = 0; j < N; ++j) {
		 data[i][j] = data_tmp[i+j*M];
	  }
   }


   Classifier* out = run_boost(data, labels, M, N, threshSets, smooth_wts, numIters);

   vector< vector< float> > all_params = out->getParams();
   int num_learners = all_params.size();
   int num_params; // assuming all learners are of the same type
   if(num_learners > 0) {
	  num_params = all_params[0].size();
   }
   else {
	  num_params = -1;
   }
   if (num_params != -1) {
	  plhs[0] = mxCreateNumericMatrix(num_learners, num_params,
									  mxSINGLE_CLASS, 0);

	  float* output = static_cast<float*>(mxGetData(plhs[0]));
	  // now fill in the data
	  for(int i = 0; i < num_learners; ++i) {
		 for(int j = 0; j < num_params; ++j) {
			output[j*num_learners + i] = all_params[i][j];
		 }
		 printf("\n");
	  }


   }

   // free data
   for(int i = 0; i < M; ++i) {
	  delete[] data[i];
   }
   delete[] data;
   delete out;

}
