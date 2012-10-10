package javaboost.weaklearning;
import java.util.Arrays;
import java.util.Comparator;

import javaboost.util.Utils;
import javaboost.regression.*;


public class MultiFeatureLRToSigmoidLearner implements WeakLearner{

    private int[] m_featColumns = null;
    private double m_leftConf = 0;
    private double m_rightConf = 0;
    private double m_storedLoss = 0;
    private double[]  m_lrSolution = null;
    private double m_smoothingW;
    private int m_numToDrop = 0;
    private static float[] m_thresholds = {0.3f, 0.4f,
					   0.45f, 0.5f, 0.55f,
					   0.6f, 0.7f};
    private int m_chosenThreshold = -1;

    public MultiFeatureLRToSigmoidLearner(final int[] featColumns, double smoothingParam) {
	m_featColumns = featColumns;
	m_smoothingW = smoothingParam;
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
	// force output to be between 0 and 1
	for(int i = 0; i < output.length; ++i) {
	    output[i] = 1/(Math.exp(-output[i]) + 1.0);
	}

	// sort the data by output
	// aggregate the data
	double [][] dataLabelsSorted = new double[prunedData.length][3];
	for(int i = 0; i < data.length; ++i) {
	    dataLabelsSorted[i][0] = output[i];
	    dataLabelsSorted[i][1] = prunedLabels[i];
	    dataLabelsSorted[i][2] = prunedWeights[i];
	}
	//sort it
	Arrays.sort(dataLabelsSorted, new Comparator<double[]>(){
		public int compare(final double[] row1, final double[] row2){
		    return Double.compare(row1[0], row2[0]);
		}
	    });




	int numBins = m_thresholds.length+1;
	double[] cumPosBins = new double[numBins];
	double[] cumNegBins = new double[numBins];
	double dcWeights = 0;
	int binIdx = 0;

	for(int i = 0; i < dataLabelsSorted.length; ++i) {
	    if(Double.isInfinite(dataLabelsSorted[i][0])){
		dcWeights += dataLabelsSorted[i][2];
		continue;
	    }
	    while(binIdx < m_thresholds.length &&
		  dataLabelsSorted[i][0] >= m_thresholds[binIdx]) {
		++binIdx;
	    }
	    if(dataLabelsSorted[i][1] >= 0) {
		cumPosBins[binIdx]+= dataLabelsSorted[i][2];
	    }else{
		cumNegBins[binIdx]+= dataLabelsSorted[i][2];
	    }
	}

	// compute sum total positive and negative weights
	// and the cummulative sums

	for(int i = 1; i < numBins; ++i) {
		cumPosBins[i] += cumPosBins[i-1];
		cumNegBins[i] += cumNegBins[i-1];
	}
	double posSum = cumPosBins[numBins-1];
	double negSum = cumNegBins[numBins-1];

	double[] leftConfs = new double[m_thresholds.length];
	double[] rightConfs = new double[m_thresholds.length];
	//double[] losses = new double[m_thresholds.length];
	double loss = 0;
	double bestLoss = 1000;
	int bestThresh = -1;
	for(int t = 0; t < m_thresholds.length; ++t) {
	    double rightPos = posSum - cumPosBins[t];
	    double rightNeg = negSum - cumNegBins[t];
	    leftConfs[t] = 0.5*Math.log((regularizer+cumPosBins[t])/(regularizer+cumNegBins[t]));
	    rightConfs[t] = 0.5*Math.log((regularizer+rightPos)/(regularizer+rightNeg));
	    // compute the sigmoid
	    double alpha = rightConfs[t] - leftConfs[t];
	    double bias = leftConfs[t];
	    loss = 0;
	    for(int i = 0; i < dataLabelsSorted.length; ++i) {
		double out = 0;
		if(!Double.isInfinite(dataLabelsSorted[i][0])) {
		    out = bias +alpha/(1+Math.exp(-m_smoothingW*(dataLabelsSorted[i][0]-m_thresholds[t])));
		}

		loss += dataLabelsSorted[i][2]*Math.log(1+Math.exp(-dataLabelsSorted[i][1]*out));

	    }

	    if(loss < bestLoss) {
		bestThresh = t;
		bestLoss = loss;
	    }
	}
	m_storedLoss = bestLoss;
	m_chosenThreshold = bestThresh;
	m_leftConf = leftConfs[bestThresh];
	m_rightConf = rightConfs[bestThresh];

	return m_storedLoss;
    }

    public WeakClassifier buildLearnedClassifier() {
	return new MultiFeatureLRSigmoidClassifier(m_featColumns,
						   m_thresholds[m_chosenThreshold],
						   m_smoothingW,
						   m_lrSolution,
						   m_leftConf,
						   m_rightConf);
    }

    public double getLearnedLoss() {
	return m_storedLoss;
    }

    public int[] getTargetColumns() {
	return m_featColumns;
    }
}
