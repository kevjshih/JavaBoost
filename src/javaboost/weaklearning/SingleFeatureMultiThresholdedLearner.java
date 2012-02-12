package javaboost.weaklearning;
import java.util.Arrays;
import java.util.Comparator;

public class SingleFeatureMultiThresholdedLearner implements WeakLearner{
    private int m_featColumn = -1;
    private float[] m_thresholds = null;
    private int m_chosenThreshold = -1;
    private double m_leftConf = 0;
    private double m_rightConf = 0;
    private double m_storedLoss = 0;
    private boolean m_isMonotonic = false;
    private MonotonicityManager m_manager = null;


    private class MonotonicityManager{
	// construct bins based on the thresholds
	private double[] confs = null;
	public MonotonicityManager() {
	    Arrays.sort(m_thresholds);
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

    }


    public SingleFeatureMultiThresholdedLearner(int featColumn, float[] thresholds, boolean isMonotonic) {
	m_featColumn = featColumn;
	m_thresholds = thresholds;
	m_isMonotonic = isMonotonic;
	if(isMonotonic) {
	    m_manager = new MonotonicityManager();
	}
    }

    public final double train(final float[][] data, final int labels[], final double[] weights) {
	m_leftConf = 0;
	m_rightConf = 0;
	double regularizer = 1/data.length;

	// aggregate the data
	double [][] dataLabelsSorted = new double[data.length][3];
	for(int i = 0; i < data.length; ++i) {
	    dataLabelsSorted[i][0] = data[i][m_featColumn];
	    dataLabelsSorted[i][1] = labels[i];
	    dataLabelsSorted[i][2] = weights[i];
	}
	//sort it
	Arrays.sort(dataLabelsSorted, new Comparator<double[]>(){
		public int compare(final double[] row, final double[] row2){
		    return Double.compare(row[0], row2[0]);
		}
	    });
	int numBins = m_thresholds.length +1;
	double[] cumPosBins = new double[numBins];
	double[] cumNegBins = new double[numBins];
	int[] threshIdx = new int[m_thresholds.length];
	int binIdx = 0;
	for(int i = 0; i < data.length; ++i) {
	    while(binIdx < m_thresholds.length &&
		  data[i][0] > m_thresholds[binIdx]) {
		threshIdx[binIdx] = i;
		binIdx++;
	    }
	    if(data[i][1] >= 0) {
		cumPosBins[binIdx]+= data[i][2];
	    }else{
		cumNegBins[binIdx]+= data[i][2];
	    }
	}

	double posSum = 0;
	double negSum = 0;
	for(int i = 0; i < numBins; ++i) {
	    posSum += cumPosBins[i];
	    negSum += cumNegBins[i];
	    if(i > 0) {
		cumPosBins[i] += cumPosBins[i-1];
		cumNegBins[i] += cumNegBins[i-1];
	    }
	}

	double[] leftConfs = new double[m_thresholds.length];
	double[] rightConfs = new double[m_thresholds.length];
	double[] losses = new double[m_thresholds.length];

	double bestLoss = 1000;
	int bestThresh = -1;
	for(int t = 0; t < m_thresholds.length; ++t) {
	    double rightPos = posSum - cumPosBins[t];
	    double rightNeg = negSum - cumNegBins[t];
	    leftConfs[t] = 0.5*Math.log((regularizer+cumPosBins[t])/(regularizer+cumNegBins[t]));
	    rightConfs[t] = 0.5*Math.log((regularizer+rightPos)/(regularizer+negSum));
	    double[] adjusted = m_manager.adjustConfidences(leftConfs[t], rightConfs[t], t);
	    leftConfs[t] = adjusted[0];
	    rightConfs[t] = adjusted[1];
	    losses[t] = Math.exp(-leftConfs[t])*cumPosBins[t] +
		Math.exp(leftConfs[t])*cumNegBins[t] +
		Math.exp(-rightConfs[t])*rightPos +
		Math.exp(rightConfs[t])*rightNeg;
	    losses[t] = losses[t]/(1+losses[t]);
	    if(losses[t] < bestLoss) {
		bestThresh = t;
		bestLoss = losses[t];
	    }
	}

	m_storedLoss = bestLoss;
	m_chosenThreshold = bestThresh;
	m_leftConf = leftConfs[bestThresh];
	m_rightConf = rightConfs[bestThresh];
	return m_storedLoss;
    }

    public WeakClassifier buildLearnedClassifier() {
	m_manager.addConfidences(m_leftConf, m_rightConf, m_chosenThreshold);
	return new SingleFeatureThresholdedClassifier(m_featColumn,
						      m_thresholds[m_chosenThreshold],
						      m_leftConf,
						      m_rightConf);
    }

    public double getLearnedLoss() {
	return m_storedLoss;
    }

}