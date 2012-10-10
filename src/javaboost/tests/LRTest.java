package javaboost.tests;

import java.util.ArrayList;

import javaboost.weaklearning.*;
import javaboost.boosting.*;
public class LRTest {
    public static void main(String[] args) {
	Dataset data = gen(500, 20);
	ArrayList<WeakLearner> wl = new ArrayList<WeakLearner>();
	for(int i = 0; i < 10; ++i) {
	    int[] arr = {i*2, i*2+1};
	    wl.add(new MultiFeatureLRToSigmoidLearner(arr, 1.5));
	    //wl.add(new SingleFeatureThresholdedLearner(i, (float)0.5));
	}
	LogitBoost.train(data.m_data, data.m_labels, wl, 900, null);
    }

    private static Dataset gen(int rows, int cols) {
	float[][] data = new float[rows][cols];
	int[] labels = new int[rows];
	int pos = rows/2;
	int negs = rows - pos;
	for(int i = 0; i < pos; ++i) {
	    for(int j = 0; j < cols; ++j) {
		data[i][j] = (float)(Math.random() + 0.2);
	    }
	    labels[i] = 1;
	}
	for(int i = pos; i < rows; ++i) {
	    for(int j = 0; j < cols; ++j) {
		data[i][j] = (float)Math.random();
	    }
	    labels[i] = -1;
	}
	return new Dataset(data, labels);
    }

    private static class Dataset {
	public float[][] m_data;
	public int[] m_labels;
	public Dataset(float[][] data, int[] labels) {
	    m_data = data;
	    m_labels = labels;
	}
    }


}
