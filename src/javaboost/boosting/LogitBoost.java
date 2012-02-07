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

    public static AdditiveClassifier trainConcurrent(final float[][] data,
					   final int[] labels,
					   List<WeakLearner> learners,
						     int maxIterations,
						     int numThreads){
	ExecutorService threadpool = ThreadPool.getThreadpoolInstance(numThreads);
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
	    //WeakLearner bestLearner = null;
	    //double bestLoss = 1000; // anything greater than 1 should work
	    final double[] weightsCpy = weights;
	    final Collection<Callable<Object>> tasks
		= new ArrayList<Callable<Object>>();
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
	    System.out.println("Iteration: " + t  + " weighted loss: " + bestLoss);

	    // update weights
	    confs = Utils.addVectors(bestClassifier.classify(data), confs);

	    double[] labelDotConfs = Utils.dotMultiplyVectors(confs, labels);

	    for(int i = 0; i < labelDotConfs.length; ++i) {
		weights[i] = 1/(1+Math.exp(labelDotConfs[i]));
	    }

	    Utils.normalizeVector(weights);

	}
	threadpool.shutdown();
	return new AdditiveClassifier(output);
    }

}