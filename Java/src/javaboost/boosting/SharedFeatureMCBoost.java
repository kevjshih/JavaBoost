package javaboost.boosting;

import java.util.List;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Arrays;
import java.util.Set;
import java.util.HashSet;


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
	int numClasses = classes.size();
	List<WeakClassifierMC> classifiers = new ArrayList<WeakClassifierMC>();
	for(int t = 0; t < maxIterations; ++t) {
	    System.out.println("Iteration: " + t);
	    List<Set<Integer>> psets = new ArrayList<Set<Integer>>();
	    List<Set<Integer>> nsets = new ArrayList<Set<Integer>>();
	    // start with everything in negative set, slowly add classes into pset
	    float bestLossSubsetLoop = Float.MAX_VALUE;
	    int bestSubsetIdx = -1;
	    for(int c = 0; c < numClasses; ++c) {
		Set<Integer> currPset;
		Set<Integer> currNset;
		Set<Integer> currNsetFixed;
		if(c > 0) {
		    currPset = new HashSet<Integer>(psets.get(c-1));
		    currNsetFixed = new HashSet<Integer>(nsets.get(c-1));

		} else {
		    currPset = new HashSet<Integer>();
		    currNsetFixed = new HashSet<Integer>(classes);
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
		if(c > 0) {
		    nsets.add(new HashSet<Integer>(nsets.get(c-1)));
		    psets.add( new HashSet<Integer>(psets.get(c-1)));
		} else {
		    nsets.add(new HashSet<Integer>(classes));
		    psets.add(new HashSet<Integer>());
		}
		nsets.get(c).remove(nextClassCandidate);

		psets.get(c).add(nextClassCandidate);
		if(bestLossClassLoop < bestLossSubsetLoop) {
		    bestLossSubsetLoop = bestLossClassLoop;
		    bestSubsetIdx = c;
		}
	    } // end of subset loop
	    // we should know the best subset here
	    // retrain because too lazy to cache..
	    for(Integer classId : psets.get(bestSubsetIdx)) {
		System.out.print(classId + " ");
	    }
	    System.out.println();
	    float bestLoss = Float.MAX_VALUE;
	    WeakLearnerMC bestWl = null;
	    for(WeakLearnerMC wl: allLearners) {
		float currLoss = wl.train(data, labels, weights, psets.get(bestSubsetIdx), nsets.get(bestSubsetIdx));
		if(currLoss < bestLoss) {
		    bestLoss = currLoss;
		    bestWl = wl;
		}
	    }
	    WeakClassifierMC bestClassifier = bestWl.buildLearnedClassifier();
	    classifiers.add(bestClassifier);
	    // update weights by reference
	    updateWeights(data, weights, labels, bestClassifier, classes);
	} // end of iter loop
	return new AdditiveClassifierMC(classifiers, classes);
    } // end of function

    public static void updateWeights(final float[][] data, float[][] weights, final int[][] labels,
				     WeakClassifierMC bestClassifier, Set<Integer> classes) {
	for(Integer c : classes) {
	    float[] outputs = bestClassifier.classify(data, c);
	    for(int i = 0; i < data.length; ++i) {
		weights[c][i] = weights[c][i]*((float)Math.exp(-1*labels[c][i]*outputs[i]));
	    }
	}
    }

    public static float[][] initalizeWeights(int[][] labels) {
	float[][] out = new float[labels.length][labels[0].length];
	for(int i =0; i < out.length; ++i) {
	    for(int j = 0; j < out[i].length; ++j) {
		out[i][j] = 1.0f;
	    }
	}
	return out;
    }

    private static int[] combineSubsetLabels(final int[][] labels,
					     Set<Integer> pset,
					     Set<Integer> nset) {
	int numClasses = labels.length;
	int numEx = labels[0].length;
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
