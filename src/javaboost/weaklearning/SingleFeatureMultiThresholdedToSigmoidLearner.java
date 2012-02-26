package javaboost.weaklearning;
import java.util.Arrays;
import java.util.Comparator;

public class SingleFeatureMultiThresholdedToSigmoidLearner implements WeakLearner{
    private int m_featColumn = -1;
    private float[] m_thresholds = null;
    private int m_chosenThreshold = -1;
    private double m_leftConf = 0;
    private double m_rightConf = 0;
    private double m_storedLoss = 0;
    private boolean m_isMonotonic = false;
    private double m_smoothingW = 10;
    private MonotonicityManager m_manager = null;


    private class MonotonicityManager{
	// construct bins based on the thresholds
	private double[] confs = null;
	public MonotonicityManager() {

	    confs = new double[m_thresholds.length +1];
	    for(int i = 0; i < confs.length; ++i) {
		confs[i] = 0;
	    }
	}

	public double[] adjustConfidences(double leftConf, double rightConf, int thresholdIdx) {
	    if(rightConf < leftConf) {

		// tune down leftConf
		double rightResult = rightConf + confs[thresholdIdx+1];
		double leftResult = leftConf + confs[thresholdIdx];
		if(rightResult < leftResult) {
		    leftConf = rightResult - confs[thresholdIdx];
		}
	    }
	    double[] outConf = {leftConf, rightConf};
	    return outConf;
	}

	public void addConfidences(double leftConf, double rightConf, int thresholdIdx) {
	    for(int i = 0; i < confs.length; ++i) {
		if(i <= thresholdIdx) {
		    confs[i] += leftConf;
		}else {
		    confs[i] += rightConf;
		}
	    }
	}

	public double[] getConfidenceFunction() {
	    return confs;
	}

    }


    public SingleFeatureMultiThresholdedToSigmoidLearner(final int featColumn, final float[] thresholds, final boolean isMonotonic) {
	m_featColumn = featColumn;
	m_thresholds = thresholds;
	Arrays.sort(m_thresholds);
	m_isMonotonic = isMonotonic;
	if(isMonotonic) {
	    m_manager = new MonotonicityManager();
	}
    }

    public double[] getConfidenceFunction() {
	if(m_isMonotonic) {
	    return m_manager.getConfidenceFunction();
	}else{
	    return null;
	}
    }

    public float[] getThresholds() {
	return m_thresholds;
    }

    public final double train(final float[][] data, final int labels[], final double[] weights) {
	m_leftConf = 0;
	m_rightConf = 0;
     	double regularizer = 1.0/data.length;

	// aggregate the data
	double [][] dataLabelsSorted = new double[data.length][3];
	for(int i = 0; i < data.length; ++i) {
	    dataLabelsSorted[i][0] = data[i][m_featColumn];
	    dataLabelsSorted[i][1] = labels[i];
	    dataLabelsSorted[i][2] = weights[i];
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
	    if(m_isMonotonic) {
		double[] adjusted = m_manager.adjustConfidences(leftConfs[t], rightConfs[t], t);
		leftConfs[t] = adjusted[0];
		rightConfs[t] = adjusted[1];
	    }


	    // compute the sigmoid
	    double alpha = rightConfs[t] - leftConfs[t];
	    double bias = leftConfs[t];
	    loss = 0;
	    for(int i = 0; i < dataLabelsSorted.length; ++i) {
		double output = 0;
		if(!Double.isInfinite(dataLabelsSorted[i][0])) {
		    output = bias +alpha/(1+Math.exp(-m_smoothingW*(dataLabelsSorted[i][0]-m_thresholds[t])));
		}

		loss += dataLabelsSorted[i][2]*Math.log(1+Math.exp(-dataLabelsSorted[i][1]*output));
		/*		if(Double.isInfinite(dataLabelsSorted[i][0])) {
		    loss += dataLabelsSorted[i][2];
		}else{
		    loss += dataLabelsSorted[i][2]*Math.exp(-1*dataLabelsSorted[i][1]*
				     (bias +
				      alpha/(1+Math.exp(-m_smoothingW*(dataLabelsSorted[i][0]-m_thresholds[t])))));
				      }*/
	    }

	    //	    loss = loss/(1+loss);
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
	if(m_isMonotonic) {
	    m_manager.addConfidences(m_leftConf, m_rightConf, m_chosenThreshold);
	}
	return new SingleFeatureSigmoidClassifier(m_featColumn,
						  m_thresholds[m_chosenThreshold],
						  m_smoothingW,
						  m_leftConf,
						  m_rightConf);
    }

    public double getLearnedLoss() {
	return m_storedLoss;
    }

}

