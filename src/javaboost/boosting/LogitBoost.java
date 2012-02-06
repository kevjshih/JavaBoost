package javaboost.boosting;

import java.util.List;
import java.util.ArrayList;

import weaklearning.*;
import util.*;

public class LogitBoost{

    public static AdditiveClassifier train(float[][] data,
					   int[] labels,
					   List<WeakLearner> learners,
					   int maxIterations){
	List<WeakClassifier> output = new ArrayList<WeakClassifier>();

        int numExamples = labels.length;

	double[] weights = new double[numExamples];
	double[] confs = new double[numExamples];
	// initialize to uniform weights
	double initVal = 1.0/numExamples;
	for(int i = 0; i < numExamples; ++i) {
	    weights[i] = initVal;
	    confs[i] = 0; // just in case
	}

	for(int t = 0; t < maxIterations; ++t) {
	    WeakLearner bestLearner = null;
	    double bestLoss = 1000; // anything greater than 1 should work
	    for(WeakLearner wl: learners) {
		double currLoss = wl.train(data, labels, weights);
		if(currLoss < bestLoss) {
		    bestLoss = currLoss;
		    bestLearner = wl;
		}
	    }
	    // pick the best learner and construct corresponding classifier
	    WeakClassifier bestClassifier = bestLearner.buildLearnedClassifier();
	    output.add(bestClassifier);
	    System.out.println("Iteration: " + t  + " weighted loss: " + bestLoss);

	    // update weights
	    confs = Utils.addVectors(bestClassifier.classify(data), confs);

	    double[] labelDotConfs = Utils.dotMultiplyVectors(confs, labels);

	    for(int i = 0; i < labelDotConfs.length; ++i) {
		weights[i] = 1/(1+Math.exp(labelDotConfs[i]));
	    }

	    Utils.normalizeVector(weights);

	}
	return new AdditiveClassifier(output);
    }
}