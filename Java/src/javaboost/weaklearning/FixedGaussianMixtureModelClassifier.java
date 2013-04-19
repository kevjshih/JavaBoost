package javaboost.weaklearning;

import javaboost.mixturemodels.*;

public final class FixedGaussianMixtureModelClassifier implements WeakClassifier{
    private GaussianMixtureModel m_positiveModel = null;
    private GaussianMixtureModel m_negativeModel = null;

    private double m_posConf;
    private double m_negConf;

    public FixedGaussianMixtureModelClassifier(double posConf,
					       double negConf,
					       GaussianMixtureModel pos,
					       GaussianMixtureModel neg) {
	m_positiveModel = pos;
	m_negativeModel = neg;
	m_posConf = posConf;
	m_negConf = negConf;
    }

    public double[] classify(float[][] data) {
	double[] output = new double[data.length];
	for(int i = 0; i < data.length; ++i) {
	    double posProb = m_positiveModel.getPDF(data[i]);
	    double negProb = m_negativeModel.getPDF(data[i]);
	    if(posProb > negProb) {
		output[i] = m_posConf;
	    }else { // equal or neg
		output[i] = m_negConf;
	    }
	}
	return output;
    }

    public String toString() {
	String description = "FixedGMM Positive:" + m_posConf +
	    "Negative: " + m_negConf;
	return description;
    }

    public int[] getColumns() {
	return null;
    }

}