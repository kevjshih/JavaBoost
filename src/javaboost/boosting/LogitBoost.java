package javaboost.boosting;

import java.util.List;
import java.util.ArrayList;
import java.util.Collection;

import javaboost.*;
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

	    double[] labelDotConfs = Utils.ebeMultiplyVectors(confs, labels);

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

    public static List<Classifier> trainMultitaskClassifiers(final float[][] data,
							     final int[][] labels,
							     List<List<WeakLearner>> allLearners,
							     int maxIterations) {
	int num_tasks = labels[0].length;
	List<double[]> all_weights = new ArrayList<double[]>();
	int[][] transposedLabels = Utils.transposeMatrix(labels);
	for(int i = 0; i < num_tasks; ++i) {
	    all_weights.add(Utils.getBalancedWeights(transposedLabels[i]));

	}

	return null;
    }

    public static LayeredClassifier trainLayeredClassifier(final float[][] positives,
							   final float[][] negatives,
							   final int[][] labels,
							   List<List<WeakLearner> > allBaseLearners,
							   int maxIterations){
	// figure out how many base level classifiers there will be
	int num_bases = labels[0].length;

	List<AdditiveClassifier> baseClassifiers = new ArrayList<AdditiveClassifier>();
	for(int i = 0; i < num_bases; ++i) { // for each base
	    int num_negs = 0;
	    // figure out how many negatives are in this base
	    for(int j = 0; j < labels.length; ++j) {
		if(labels[j][i] == -1) {
		    ++num_negs;
		}
	    }
	    int data_length = num_negs + positives.length;
	    float[][] data = new float[data_length][positives[0].length];
	    int[] data_labels = new int[data_length];

	    // load the positives
	    for(int j = 0; j < positives.length; ++j) {
		for(int k = 0; k < positives[0].length; ++k) {
		    data[j][k] = positives[j][k];
		}
		data_labels[j] = 1;
	    }
	    int data_j = positives.length;
	    for(int j = 0; j < negatives.length; ++j) {
		if(labels[j][i] == -1) {
		    for(int k = 0; k < negatives[0].length; ++k) {
			data[data_j][k] = negatives[j][k];
		    }
		    data_labels[data_j] = -1;
		    ++data_j;
		}
	    }
	    assert(data_j == data_length +1);
	    double[] weights = Utils.getBalancedWeights(data_labels);
	    assert(weights.length == data_length);

	    AdditiveClassifier baseC = trainConcurrent(data, data_labels,
						       allBaseLearners.get(i),
						       maxIterations, 8,
						       weights);
	    baseClassifiers.add(baseC);
	}

	// create the aggregate dataset
	int num_data_all = positives.length+negatives.length;
	float[][] data_all = new float[num_data_all][positives[0].length];
	int[] labels_all = new int[num_data_all];
	for(int i = 0; i < num_data_all; ++i) {
	    if(i < positives.length) {
		for(int j = 0; j < positives[0].length; ++j) {
		    data_all[i][j] = positives[i][j];
		}
	        labels_all[i] = 1;
	    }else{
		labels_all[i] = -1;
		for(int j = 0; j < positives[0].length; ++j) {
		    data_all[i][j] = negatives[i-positives.length][j];
		}
	    }
	}
	float[][] data_final = new float[num_data_all][num_bases];
	// find the new thresholds
	int num_threshes = 100;
	List<WeakLearner> learners2 = new ArrayList<WeakLearner>();

	for(int c = 0; c < num_bases; ++c) {
	    double[] output = baseClassifiers.get(c).classify(data_all);
	    float smallestPos = 10000;
	    float greatestNeg = -10000;
	    for(int i = 0; i < num_data_all; ++i) {
		data_final[i][c] = (float)output[i];
		if(i < positives.length && output[i] < smallestPos) {
		    smallestPos = (float)output[i];
		}else if(i >= positives.length && output[i] > greatestNeg) {
		    greatestNeg = (float)output[i];
		}
	    }
	    float diff = Math.abs(smallestPos - greatestNeg);
	    float lower = Math.min(smallestPos, greatestNeg);
	    float upper = Math.max(smallestPos, greatestNeg);
	    float stepSize = diff/num_threshes;
	    float[] threshes = new float[num_threshes];
	    for(int i = 0; i < num_threshes; ++i) {
		threshes[i] = lower + stepSize*i;
	    }
	    double sw = 1/(3*Utils.std(output));
	    learners2.add(new SingleFeatureMultiThresholdedToSigmoidLearner(c, threshes, true, sw));
	}
	double[] weights_final = Utils.getBalancedWeights(labels_all);

	AdditiveClassifier finalClassifier = trainConcurrent(data_final,
							     labels_all,
							     learners2,
							     maxIterations,
							     8,
							     weights_final);
	return new LayeredClassifier(baseClassifiers, finalClassifier);
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


	    double[] labelDotConfs = Utils.ebeMultiplyVectors(confs, labels);

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