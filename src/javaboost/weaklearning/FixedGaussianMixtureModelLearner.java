package javaboost.weaklearning;
import javaboost.mixturemodels.*;

public final class FixedGaussianMixtureModelLearner implements WeakLearner{
    private GaussianMixtureModel m_positiveModel = null;
    private GaussianMixtureModel m_negativeModel = null;

    private double m_posConf;
    private double m_negConf;

    private double m_storedLoss;

    public FixedGaussianMixtureModelLearner(GaussianMixtureModel positives,
					    GaussianMixtureModel negatives) {
	m_positiveModel = positives;
	m_negativeModel = negatives;
    }

    public double train(final float[][] data, final int[] labels, final double[] weights) {
	double regularizer = 1.0/data.length;

	// positive model
	double positiveTrueWeights = 0;
	double positiveFalseWeights = 0;

	// negative model
	double negativeTrueWeights = 0;
	double negativeFalseWeights = 0;

	for(int i = 0; i < data.length; ++i) {
	    double posProb = m_positiveModel.getPDF(data[i]);
	    double negProb = m_negativeModel.getPDF(data[i]);
	    if(posProb > negProb) {
		if(labels[i] >= 0) {
		    positiveTrueWeights += weights[i];
		}else{
		    positiveFalseWeights += weights[i];
		}
	    }else { // equal or neg
		if(labels[i] >= 0) {
		    negativeTrueWeights += weights[i];
		}else{
		    negativeFalseWeights += weights[i];
		}
	    }
	}

	m_posConf = 0.5*Math.log((regularizer + positiveTrueWeights)/(regularizer + positiveFalseWeights));
	m_negConf = 0.5*Math.log((regularizer + negativeTrueWeights)/(regularizer + negativeFalseWeights));

	m_storedLoss = Math.exp(-m_posConf)*positiveTrueWeights +
	    Math.exp(m_posConf)*positiveFalseWeights +
	    Math.exp(-m_negConf)*negativeTrueWeights +
	    Math.exp(m_negConf)*negativeFalseWeights;
	m_storedLoss = m_storedLoss/(1+m_storedLoss);
	return m_storedLoss;
    }

    public WeakClassifier buildLearnedClassifier() {
	return new FixedGaussianMixtureModelClassifier(m_posConf, m_negConf,
						  m_positiveModel,
						  m_negativeModel);
    }

    public double getLearnedLoss() {
	return m_storedLoss;
    }
}