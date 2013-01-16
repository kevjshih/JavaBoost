package javaboost.boosting;

import java.util.List;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Arrays;


import javaboost.*;
import javaboost.weaklearning.*;
import javaboost.util.*;

import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Callable;

public class SharedFeatureMCBoost {
    // data is formatted as M by N where we have M examples with N features
    // labels is formatted as C by M, where each row corresponds to a class c in C and
    // is of length M for number of examples (Note: essentially transposed of data)
    public static AdditiveClassifierMC train(final float[][] data,
					     final int[][] labels,
					     Set<Integer> classes,
					     List<WeakLearnerMC> allLearners,
					     int maxIterations) {
	float[][] weights = initalizeWeights(labels);
	numClasses = classes.size();
	List<WeakClassifierMC> classifiers = new ArrayList<WeakClassifierMC>();
	for(int t = 0; t < maxIterations; ++t) {
	    Set<Integer>[] psets = new Set<Integer>[numClasses];
	    Set<Integer>[] nsets = new Set<Integer>[numClasses];
	    float bestLossSubsetLoop = Float.MAX_VALUE;
	    int bestSubsetIdx = -1;
	    for(int c = 0; c < numClasses; ++c) {
		Set<Integer> currPset;
		Set<Integer> currNset;
		Set<Integer> currNsetFixed;
		if(c > 0) {
		    currPset = new Set<Integer>(psets[c-1]);
		    currNsetFixed = new Set<Integer>(nsets[c-1]);

		} else {
		    currPset = new HashSet<Integer>();
		    currNsetFixed = new Set<Integer>(classes);
		}
		float bestLossClassLoop  = Float.MAX_VALUE;
		Integer nextClassCandidate = null;
		for(Integer classId : currNsetFixed) { // pick next best class
		    // set class classId to new pset element and compute lowest loss
		    currNset = new HashSet<Integer>(currNsetFixed);
		    currNset.remove(classId);
		    currPset.add(classId);
		    float bestLoss = Float.MAX_VALUE;
		    for(WeakLearnerMC wl: allLearners) {
			float currLoss = wl.train(data, labels, weights, currPset, currNset);
			if(currLoss < bestLoss) {
			    bestLoss = currLoss;
			}
		    }
		    if(bestLoss < bestLossClassLoop) {
			bestLossClassLoop = bestLoss;
			nextClassCandidate = classId;
		    }
		} // end of class loop
		// update greedy set
		nsets[c] = new HashSet<Integer>(nsets[c-1]);
		nsets[c].remove(nextClassCandidate);
		psets[c] = new HashSet<Integer>(psets[c-1]);
		psets[c].add(nextClassCandidate);
		if(bestLossClassLoop < bestLossSubsetLoop) {
		    bestLossSubsetLoop = bestLossClassLoop;
		    bestSubsetIdx = c;
		}
	    } // end of subset loop
	    // we should know the best subset here
	    // retrain because too lazy to cache...
	    float bestLoss = Float.MAX_VALUE;
	    WeakLearnerMC bestWl = null;
	    for(WeakLearnerMC wl: allLearners) {
		float currLoss = wl.train(data, labels, weights, currPset, currNset);
		if(currLoss < bestLoss) {
		    bestLoss = currLoss;
		    bestWl = wl;
		}
	    }
	    classifiers.add(wl.buildLearnedClassifier());
	} // end of iter loop
	return new AdditiveClassifierMC(classifiers, classes);
    } // end of function

    private static int[] combineSubsetLabels(final int[][] labels,
					     Set<Integer> pset,
					     Set<Integer> nset) {
	numClasses = labels.length;
	numEx = labels[0].length;
	int[] subLabels = new int[numEx];
	for(int i = 0; i < numClasses; ++i) {
	    boolean inPset = pset.contains(i);
	    for(int j = 0; j < numEx; ++j) {
		if(labels[i][j] == 1) {
		    if(inPset) {
			subLabels[j] = 1;
		    } else {
			subLabels[j] = -1;
		    }
		}
	    }

	}
	// now double check to make sure all labels were assigned
	for(int i = 0; i < subLabels.length; ++i) {
	    assert(subLabels[i] != 0);
	}
	return subLabels;
    }


}
