package javaboost.tests;

import java.util.ArrayList;
import java.util.Random;

import javaboost.weaklearning.*;
import javaboost.boosting.*;

public class LRTest {
    public static void main(String[] args) {
	Dataset data = gen(600, 10, 1);
	Dataset test = gen(300, 10, 2);
	ArrayList<WeakLearner> wl = new ArrayList<WeakLearner>();
	float[] arr = new float[200];
	for(int i = 0; i < 200; ++i)
	    arr[i] = (float)i/100;
	for(int i = 0; i < 10; ++i) {
	    wl.add(new SingleFeatureMultiThresholdedToSigmoidLearner(i, arr, false, 1.5, true));
	    //wl.add(new SingleFeatureThresholdedLearner(i, (float)0.5));
	}
	//LogitBoost.trainConcurrent(data.m_data, data.m_labels, wl, 100, 4, null);
	AdditiveClassifier ac = LogitBoost.train(data.m_data, data.m_labels, wl, 100, null);
	ac.listClassifiers();
	/*
	double[][] contributions = ac.getContributions(data.m_data);
	/*	for(int i = 0; i < contributions.length; ++i) {
	    for(int j = 0; j < contributions[0].length; ++j) {
		System.out.print(contributions[i][j]+" ");
	    }
	    System.out.println();

	    }


	int[] rel = ac.getColumnsUsed();
	/*for(int i = 0; i < rel.length; ++i) {
	    System.out.print(rel[i] + " ");
	}
	System.out.println();
	*/

	double[] output = ac.classify(test.m_data);
	int correct = 0;
	for(int i = 0; i < test.m_data.length; ++i) {
	    if(test.m_labels[i] > 0 && output[i] > 0 || test.m_labels[i] <= 0 && output[i] <= 0)
		correct++;


	}
	System.out.println(correct);

	System.exit(0);
    }

    private static Dataset gen(int rows, int cols, int seed) {
	Random rand = new Random(seed);
	float[][] data = new float[rows][cols];
	int[] labels = new int[rows];
	int pos = rows/2;
	int negs = rows - pos;
	for(int i = 0; i < pos; ++i) {
	    for(int j = 0; j < cols; ++j) {
		if(rand.nextDouble() >0.25)
		    data[i][j] = Float.NEGATIVE_INFINITY;
		else {
		    data[i][j] = (rand.nextFloat() + 0.3f);
		}
	    }
	    labels[i] = 1;
	}
	for(int i = pos; i < rows; ++i) {
	    for(int j = 0; j < cols; ++j) {
		if(rand.nextDouble() >0.5)
		    data[i][j] = Float.NEGATIVE_INFINITY;
		else
		    data[i][j] = rand.nextFloat();
	    }
	    labels[i] = -1;
	}
	return new Dataset(data, labels);
    }

    private  static class Dataset {
	public float[][] m_data;
	public int[] m_labels;
	public Dataset(float[][] data, int[] labels) {
	    m_data = data;
	    m_labels = labels;
	}
    }


}
