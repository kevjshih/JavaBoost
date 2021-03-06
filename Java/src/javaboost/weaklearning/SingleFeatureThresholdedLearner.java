package javaboost.weaklearning;

public class SingleFeatureThresholdedLearner implements WeakLearner{
    private int m_featColumn;
    private float m_threshold;
    private double m_leftConf = 0;
    private double m_rightConf = 0;

    private double m_storedLoss = 0;

    public SingleFeatureThresholdedLearner(int featColumn, float threshold) {
	m_featColumn = featColumn;
	m_threshold = threshold;
    }

    public final double train(final float[][] data,final int labels[],final double[] weights){
	m_leftConf = 0;
	m_rightConf = 0;
	double weightedPos_l = 0;
	double weightedNeg_l = 0;
	double weightedPos_r = 0;
	double weightedNeg_r = 0;
	double dcWeights = 0;
	double regularizer = 1.0/data.length;

	for(int i = 0; i < data.length; ++i) {
	    if(Float.isInfinite(data[i][m_featColumn])) {
		dcWeights+=weights[i];
		continue;
	    }
	    if(data[i][m_featColumn] < m_threshold) {
		if(labels[i] >= 0){
		    weightedPos_l+=weights[i];
		}else{
		    weightedNeg_l+=weights[i];
		}
	    }else {
		if(labels[i] >= 0){
		    weightedPos_r+=weights[i];
		}else {
		    weightedNeg_r+=weights[i];
		}
	    }
	}

	m_leftConf = 0.5*Math.log((regularizer+weightedPos_l)/(regularizer+weightedNeg_l));
	m_rightConf = 0.5*Math.log((regularizer+weightedPos_r)/(regularizer+weightedNeg_r));

	//return the weighted loss and cache it for parallel runs
	m_storedLoss = Math.log(1+Math.exp(-m_leftConf))*weightedPos_l +
	    Math.log(1+Math.exp(m_leftConf))*weightedNeg_l +
	    Math.log(1+Math.exp(-m_rightConf))*weightedPos_r +
	    Math.log(1+Math.exp(m_rightConf))*weightedNeg_r+
	    Math.log(2)*dcWeights;

	return m_storedLoss;
    }

    public WeakClassifier buildLearnedClassifier() {
	return new SingleFeatureThresholdedClassifier(m_featColumn,
						      m_threshold,
						      m_leftConf,
						      m_rightConf);
    }

    public double getLearnedLoss() {
	return m_storedLoss;
    }

    public int[] getTargetColumns() {
	int[] out = {m_featColumn};
	return out;
    }

}