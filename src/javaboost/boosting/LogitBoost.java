package javaboost.boosting;

import java.util.List;
import java.util.ArrayList;
import java.util.Collection;

import javaboost.weaklearning.*;
import javaboost.util.*;

import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Callable;

public class LogitBoost{

    public static AdditiveClassifier train(float[][] data,
					   int[] labels,
					   List<WeakLearner> learners,
					   int maxIterations,
					   double[] weights){
	List<WeakClassifier> output = new ArrayList<WeakClassifier>();

        int numExamples = labels.length;
	double initVal = 1.0/numExamples;
	if(weights == null){
	    weights = new double[numExamples];
	    for(int i = 0; i < numExamples; ++i) {
		weights[i] = initVal;
	    }
	}
	double[] confs = new double[numExamples];
	// initialize to uniform weights

	for(int i = 0; i < numExamples; ++i) {
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


	    // update weights
	    confs = Utils.addVectors(bestClassifier.classify(data), confs);

	    double[] labelDotConfs = Utils.dotMultiplyVectors(confs, labels);

	    for(int i = 0; i < labelDotConfs.length; ++i) {
		weights[i] = 1/(1+Math.exp(labelDotConfs[i]));
	    }

	    System.out.println("Iteration: " + t
			       + " weighted loss: " + bestLoss
			       + " mean weight: "+ Utils.mean(weights));


	    Utils.normalizeVector(weights);

	}
	return new AdditiveClassifier(output);
    }

    public static AdditiveClassifier trainConcurrent(final float[][] data,
					   final int[] labels,
					   List<WeakLearner> learners,
						     int maxIterations,
						     int numThreads, double[] weights){
	ExecutorService threadpool = ThreadPool.getThreadpoolInstance(numThreads);
	List<WeakClassifier> output = new ArrayList<WeakClassifier>();

        int numExamples = labels.length;


	double initVal = 1.0/numExamples;
	if(weights == null){
	    weights = new double[numExamples];
	    for(int i = 0; i < numExamples; ++i) {
		weights[i] = initVal;
	    }
	}
	double[] confs = new double[numExamples];
	// initialize to uniform weights

	for(int i = 0; i < numExamples; ++i) {
	    confs[i] = 0; // just in case
	}


	final Collection<Callable<Object>> tasks
	    = new ArrayList<Callable<Object>>(learners.size());

	for(int t = 0; t < maxIterations; ++t) {
	    //WeakLearner bestLearner = null;
	    //double bestLoss = 1000; // anything greater than 1 should work
	    final double[] weightsCpy = weights;
	    tasks.clear();

	    for(final WeakLearner wl: learners) {
		tasks.add(Executors.callable(new Runnable() {
			public void run() {
			    try{
				wl.train(data, labels, weightsCpy);
			    }catch(Exception ex) {
				ex.printStackTrace();
			    }
			}
		    }));
	    }
	    try{
		threadpool.invokeAll(tasks);
	    }catch(Exception ex) {}

	    WeakLearner bestLearner = null;
	    double bestLoss = 1000;
	    for(WeakLearner wl: learners) {
		if(wl.getLearnedLoss() < bestLoss) {
		    bestLoss = wl.getLearnedLoss();
		    bestLearner = wl;
		}
	    }
		    // pick the best learner and construct corresponding classifier
	    WeakClassifier bestClassifier = bestLearner.buildLearnedClassifier();
	    output.add(bestClassifier);


	    // update weights
	    confs = Utils.addVectors(bestClassifier.classify(data), confs);


	    double[] labelDotConfs = Utils.dotMultiplyVectors(confs, labels);

	    for(int i = 0; i < labelDotConfs.length; ++i) {
		weights[i] = 1/(1+Math.exp(labelDotConfs[i]));
	    }

	    System.out.println("Iteration: " + t
			       + " weighted loss: " + bestLoss
			       + " mean weight: "+ Utils.mean(weights));


	    Utils.normalizeVector(weights);

	}

	return new AdditiveClassifier(output);
    }

}