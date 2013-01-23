package javaboost.tests;

import java.util.List;
import java.util.ArrayList;
import java.util.Random;
import java.util.Set;
import java.util.HashSet;

import javaboost.weaklearning.*;
import javaboost.boosting.*;

public class SharedTests {
    public static void main(String[] args) {
	int numcols = 10;
	int numClasses = 20;
	Set<Integer> classes = new HashSet<Integer>();
	for(int i = 0; i < numClasses; ++i) {
	    classes.add(i);
	}
	Dataset data = gen(200, numcols, numClasses);
	float[] arr = new float[100];
	List<WeakLearnerMC> wl = new ArrayList<WeakLearnerMC>();
	for(int i = 0; i < 20; ++i) {
	    arr[i] = (float)i/20;
	}

	for(int i = 0; i < numcols; ++i) {
	    wl.add(new WeakLearnerSharedMC(i, arr, classes));
	}

	float[][] weights = javaboost.boosting.SharedFeatureMCBoost.initalizeWeights(data.m_labels);

	for(int i = 0; i < weights.length; ++i) {
	    for(int j = 0; j < weights[i].length; ++j) {
		System.out.print(weights[i][j]+ " ");
	    }
	    System.out.println();
	}
	Set<Integer> pset = new HashSet<Integer>();
	Set<Integer> nset = new HashSet<Integer>();
	for(int i = 0; i < 3; ++i) {
	    pset.add(i);
	}
	for(int i = 3; i < classes.size(); ++i) {
	    nset.add(i);
	}
	for(int i = 0; i < wl.size(); ++i) {
	    float loss = wl.get(i).train(data.m_data, data.m_labels, weights, pset, nset);
	    System.out.println(loss);
	}
	WeakClassifierMC wc = wl.get(0).buildLearnedClassifier();
	float[] out = wc.classify(data.m_data, 0);
	for(int i = 0; i < out.length; ++i) {
	    System.out.println(out[i]);
	}

	SharedFeatureMCBoost.updateWeights(data.m_data, weights, data.m_labels, wc, classes);
	for(int i = 0; i < weights.length; ++i) {
	    for(int j = 0; j < weights[i].length; ++j) {
		System.out.print(weights[i][j]+ " ");
	    }
	    System.out.println();
	}

	AdditiveClassifierMC result = SharedFeatureMCBoost.train(data.m_data, data.m_labels, classes, wl, 20);

    }

    private static Dataset gen(int rows, int cols, int numClasses) {
	Random rand = new Random(1);
	float[][] data = new float[rows][cols];
	int[][] labels = new int[numClasses][rows];
	int pos = rows/2;
	int negs = rows - pos;
	for(int i = 0; i < pos; ++i) {
	    for(int j = 0; j < cols; ++j) {
		if(rand.nextDouble() >0.9)
		    data[i][j] = Float.NEGATIVE_INFINITY;
		else {
		    data[i][j] = (rand.nextFloat() + 0.5f);
		}
	    }
	    for(int k = 0; k < numClasses; ++k) {
		if(rand.nextDouble() > 0.2) {
		    labels[k][i] = 1;
		} else {
		    labels[k][i] = -1;
		}
	    }
	}
	for(int i = pos; i < rows; ++i) {
	    for(int j = 0; j < cols; ++j) {
		data[i][j] = rand.nextFloat();
	    }
	    for(int k = 0; k < numClasses; ++k) {
		if(rand.nextDouble() > 0.2) {
		    labels[k][i] = -1;
		} else {
		    labels[k][i] = 1;
		}
	    }
	}
	return new Dataset(data, labels);

    }

    private  static class Dataset {
	public float[][] m_data;
	public int[][] m_labels;
	public Dataset(float[][] data, int[][] labels) {
	    m_data = data;
	    m_labels = labels;
	}
    }


}
