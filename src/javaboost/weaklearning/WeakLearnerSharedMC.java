package javaboost.weaklearning;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Set;
import java.util.HashSet;
import java.util.Map;
import java.util.HashMap;

import javaboost.util.Utils;

public class WeakLearnerSharedMC implements WeakLearnerMC{
    private float[] m_thresholds;
    final private Set<Integer> m_classes = null;

    private float m_a_s = -1;
    private float m_b_s = -1;
    private int m_thresholdIdx = -1;

    private float m_storedLoss = -1;
    private int m_featColumn = -1;
    public WeakLearnerSharedMC(final int featColumn, final float[] thresholds, final Set<Integer> classes) {
	m_featColumn = featColumn;
	m_thresholds = thresholds;
	m_classes = classes;
    }

    public final float train(final float[][] data,
			     final int[][] labels, final float[][] weights, Set<Integer> pset, Set<Integer> nset) {




     	float regularizer = 1.0/data.length;
	int numClasses = labels.length;

	// first column is data, second is the original position
	float [][] dataIdxSorted = new float[data.length][2];
	for(int i = 0; i < data.length; ++i) {
	    dataIdxSorted[i][0] = data[i][m_featColumn];
	    dataIdxSorted[i][1] = (float)i;
	}
	// sort it
	Arrays.sort(dataIdxSorted, new Comparator<float[]>(){
		public int compare(final float[] row1, final float[] row2){
		    return Float.compare(row1[0], row2[0]);
		}
	    });
	// results should be in ascending order

	reorderedWeights = new float[weights.length][weights[0].length];
	reorderedWLproducts = new float[weights.length][weights[0].length];
	for(int c = 0; c < weights.length; ++c) {
	    for(int i = 0; i < dataIdxSorted.length; ++i) {
		reorderedWeights[c][i] = weights[c][dataIdxSorted[i][1]];
		reorderedWLproducts[c][i] = reorderedWeights[c][i]*labels[c][dataIdxSorted[i][1]];
	    }
	}
	int numBins = m_thresholds.length+1;
	float[] cumPosBins = new float[numBins];
	float[] cumNegBins = new float[numBins];
	float[] cumDenomBins = new float[numBins];

	int binIdx = 0;
	for(int c = 0; c < labels.length; ++c) {
	    if(pset.contains(c)){
		for(int i = 0; i < dataIdxSorted.length; ++i) {
		    // skip don't care values
		    if(Float.isInfinite(dataIdxSorted[i][0])) {
			continue;
		    }
		    while(binIdx < m_thresholds.length &&
			  dataIdxSorted[i][0] >= m_thresholds[binIdx]) {
			++binIdx;
		    }

		    if(labels[c][dataIdxSorted[i][1]] >= 0) {
			cumPosBins[binIdx] += reorderedWLproducts[c][i];
		    }else{
			cumNegBins[binIdx] += reorderedWLproducts[c][i];
		    }
		    cumDenomBins[binIdx] += reorderedWeights[c][i];
		}
	    }
	}

	for(int i = 1; i < numBins; ++i) {
	    cumPosBins[i] += cumPosBins[i-1];
	    cumNegBins[i] += cumNegBins[i-1];
	    cumDenomBins[i] += cumDenomBins[i-1];

	}
	float posSum = cumPosBins[numBins-1];
	float negSum = cumNegBins[numBins-1];
	float denomSum = cumDenomBins[numBins-1];

	Map<Integer, Float> biasMap = new HashMap<Integer, Float>();

	// compute bias for non-subset classes
	for(Integer c : nset) {
	    float k_c_num = (float)Utils.vectorSum(reorderedWLproducts[c]);
	    float k_c_denom = Utils.vectorSum(weights[c]);
	    biasMap.put(c, new Float(k_c_num/k_c_denom));
	}

	// go through thresholds to pick out one that minimizes loss for this feature
	float bestLoss = Float.MAX_VALUE;
	int bestThresh = -1;
	float besta_s = -1;
	float bestb_s = -1;
	for(int t = 0; t< m_thresholds.length; ++t) {
	    // comput a_s
	    float pos_val_a = posSum - cumPosBins[t];
	    float neg_val_a = negSum - cumNegBins[t];
	    float denom_a = denomSum - cumDenomBins[t];
	    a_s = (pos_val_a - neg_val_a) / denom_a;

	    float pos_val_b = cumPosBins[t];
	    float neg_val_b =cumNegBins[t];
	    float denom_b = cumDenomBins[t];
	    b_s = (pos_val_b - neg_val_b) / denom_b;

	    float loss = 0;
	    WeakClassifierMC currHyp = new SingleFeatureThresholdedSharedClassifierMC(m_featColumn,  m_thresholds[t], a_s, b_s, m_classes, biasMap);
	    loss = computeSquaredLossMC(data, m_featColumn, weights, labels, currHyp);
	    if(loss < bestLoss) {
		bestLoss = loss;
		besta_s = a_s;
		bestb_s = b_s;
		bestThresh = t;
	    }
	}
	m_a_s = besta_s;
	m_b_s = bestb_s;
	m_thresholdIdx = bestThresh;
	m_storedLoss = bestLoss;
	return bestLoss;

    }



    public WeakClassifierMC buildLearnedClassifier() {
	return new SingleFeatureThresholdedSharedClassifierMC(m_featColumn, m_thresholds[m_thresholdIdx],
							      m_a_s, m_b_s, m_classes, m_biasMap);
    }

    public static float computeSquaredLossMC(final float[][] data,
					     final float[] thresholds, final float[][] weights,
					     final int[][] labels, WeakClassifierMC hyp) {
	float loss = 0;
	int numClasses = labels.length;
	for(int c = 0; c < numClasses; ++c) {
	    scores = hyp.classify(data, c);
	    for(int i = 0; i < scores.length; ++i) {
		float diff = labels[c][i] -  scores[i];
		loss += weights[c][i]*diff*diff;
	    }
	}

    }

    public float getLearnedLoss() {
	return m_storedLoss;
    }

    public int[] getTargetColumns() {
	int[] out = {m_featColumn};
	return out;
    }

}
