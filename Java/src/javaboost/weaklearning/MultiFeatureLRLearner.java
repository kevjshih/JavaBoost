package javaboost.weaklearning;
import java.util.Arrays;
import java.util.Comparator;

import javaboost.util.Utils;
import javaboost.regression.*;

public class MultiFeatureLRLearner implements WeakLearner {
    /*
       This classifier learns a logistic regressor and separates at S(x) = 0.a5.
       Missing data( -inf) causes the entire example to be ignored in training and a 0
       is returned in test.
    */
    private int[] m_featColumns = null;

    private double m_negConf = 0;
    private double m_posConf = 0;
    private double m_storedLoss = 0;
    private double[]  m_lrSolution = null;
    private int m_numToDrop = 0;

    public MultiFeatureLRLearner(final int[] featColumns) {
	m_featColumns = featColumns;
    }

    private char[] findMissingDataRows(final float[][] data) {
	char[] badRows = new char[data.length];
	int numCols = m_featColumns.length;
	m_numToDrop = 0;
	for(int i = 0; i < data.length; ++i) {
	    badRows[i] = 0;
	    for(int j = 0; j < numCols; ++j) {
		// count number of examples with -inf where we care
		if(Float.isInfinite(data[i][m_featColumns[j]])) {
		    badRows[i] = 1;
		    m_numToDrop++;
		    break;
		}
	    }
	}
	return badRows;
    }


    private double[] pruneExampleWeights(final double[] weights, char[] badRows) {
	double[] prunedWeights = new double[weights.length - m_numToDrop];
	int cnt = 0;
	for(int i = 0; i < weights.length; ++i) {
	    if(badRows[i] == 1)
		continue;

	    prunedWeights[cnt] = weights[i];
	    ++cnt;
	}
	return prunedWeights;
    }

    private double[] pruneExampleLabels(final int[] labels, char[] badRows) {
        double[] prunedLabels = new double[labels.length - m_numToDrop];
	int cnt = 0;
	for(int i = 0; i < labels.length; ++i) {
	    if(badRows[i] == 1)
		continue;

	    prunedLabels[cnt] = (double)labels[i];
	    ++cnt;
	}
	return prunedLabels;
    }

    // removes bad rows and adds bias
    private double[][] pruneExampleFeatures(final float[][] data, char[] badRows) {
	int numCols = m_featColumns.length;
	int numToDrop = m_numToDrop;


	int numRows = data.length - numToDrop;
	double[][] prunedData =  new double[numRows][numCols+1];
	int rowcnt = 0;
	for(int i = 0; i < data.length; ++i) {
	    if(badRows[i] == 1)
		continue;
	    for(int j = 0; j < numCols+1; ++j) {
		if(j == numCols) {
		    prunedData[rowcnt][j] = 1; // bias term
		} else {
		    prunedData[rowcnt][j] = data[i][m_featColumns[j]];
		}
	    }
	    ++rowcnt;
	}
	return prunedData;
    }

    public final double train(final float[][] data, final int labels[], final double[] weights) {
	char[] badRows =  findMissingDataRows(data);
	double[][] prunedData = pruneExampleFeatures(data, badRows);
	double[] prunedWeights = pruneExampleWeights(weights, badRows);
	double[] prunedLabels = pruneExampleLabels(labels, badRows);
	double regularizer = 1.0/data.length;
	m_lrSolution = IRLS.solve(prunedData, prunedLabels, prunedWeights, 0.00001);

	double[] output = Utils.operate(prunedData, m_lrSolution);
	double weightedPos_l = 0;
	double weightedNeg_l = 0;
	double weightedPos_r = 0;
	double weightedNeg_r = 0;
	double dcWeights = 0;
	for(int i = 0; i < output.length; ++i) {
	    if(output[i] < 0) {
		if( prunedLabels[i] >= 0)
		    weightedPos_l += prunedWeights[i];
		else
		    weightedNeg_l += prunedWeights[i];
	    } else {
		if( prunedLabels[i] >= 0)
		    weightedPos_r += prunedWeights[i];
		else
		    weightedNeg_r += prunedWeights[i];

	    }
	}
	for(int i = 0; i < badRows.length; ++i) {
	    if(badRows[i] == 1)
		dcWeights += weights[i];
	}

	m_negConf = 0.5*Math.log((regularizer + weightedPos_l)/(regularizer + weightedNeg_l));
	m_posConf = 0.5*Math.log((regularizer + weightedPos_r)/(regularizer + weightedNeg_r));
		//return the weighted loss and cache it for parallel runs
	m_storedLoss = Math.log(1+Math.exp(-m_negConf))*weightedPos_l +
	    Math.log(1+Math.exp(m_negConf))*weightedNeg_l +
	    Math.log(1+Math.exp(-m_posConf))*weightedPos_r +
	    Math.log(1+Math.exp(m_posConf))*weightedNeg_r+
	    Math.log(2)*dcWeights;
	return m_storedLoss;

    }


    public WeakClassifier buildLearnedClassifier() {
	return new MultiFeatureLRClassifier(m_featColumns,
					    m_lrSolution,
					    m_negConf,
					    m_posConf);
    }



    public double getLearnedLoss() {
	return m_storedLoss;
    }

    public int[] getTargetColumns() {
	return m_featColumns;
    }

}
