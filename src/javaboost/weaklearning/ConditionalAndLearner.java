package javaboost.weaklearning;

import java.util.List;

import javaboost.conditioning.Conditional;
import javaboost.util.Utils;

public class ConditionalAndLearner implements WeakLearner{


    private List<Conditional> m_cond = null;
    private double m_trueConf = 0;

    private double m_storedLoss = 0;

    public ConditionalAndLearner(List<Conditional> conditions) {
	m_cond = conditions;
    }

    public double train(float[][] data, int labels[], double[] weights) {
	m_trueConf = 0;
	double weightedTruePos = 0;
	double weightedTrueNeg = 0;

	for(int i = 0; i < data.length; ++i) {
	    if(Utils.isValid(data[i], m_cond)) {
		if(labels[i] >= 0){
		    weightedTruePos += weights[i];
		}else{
		    weightedTrueNeg += weights[i];
		}

	    }
	}
	m_trueConf = 0.5*Math.log((1+weightedTruePos)/(1+weightedTrueNeg));

	m_storedLoss = Math.exp(-m_trueConf)*weightedTruePos +
	    Math.exp(m_trueConf)*weightedTrueNeg +
	    (1.0-weightedTruePos-weightedTrueNeg);
	return m_storedLoss;
    }

    public WeakClassifier buildLearnedClassifier() {
	return new ConditionalAndClassifier(m_cond, m_trueConf);
    }

    public double getLearnedLoss() {
	return m_storedLoss;
    }
}