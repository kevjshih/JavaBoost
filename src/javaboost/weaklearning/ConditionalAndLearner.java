package javaboost.weaklearning;

import java.util.List;

import javaboost.conditioning.Conditional;
import javaboost.util.Utils;

public class ConditionalAndLearner implements WeakLearner{


    private List<Conditional> m_cond = null;
    private double m_trueConf = 0;
    private double m_falseConf = 0;

    private double m_storedLoss = 0;

    public ConditionalAndLearner(List<Conditional> conditions) {
	m_cond = conditions;
    }

    public double train(final float[][] data,final int labels[],final double[] weights) {
	m_trueConf = 0;
	m_falseConf = 0;
	// if true
	double weightedTruePos = 0;
	double weightedTrueNeg = 0;
	// if false;
	double weightedFalsePos = 0;
	double weightedFalseNeg = 0;

	double regularizer = 1/data.length;

	for(int i = 0; i < data.length; ++i) {
	    if(Utils.isValidAnd(data[i], m_cond)) {
		if(labels[i] >= 0){
		    weightedTruePos += weights[i];
		}else{
		    weightedTrueNeg += weights[i];
		}

	    }
	    else {
		if(labels[i] >= 0){
		    weightedFalsePos += weights[i];
		}else{
		    weightedFalseNeg += weights[i];
		}
	    }
	}
	m_trueConf = 0.5*Math.log((regularizer+weightedTruePos)/(regularizer+weightedTrueNeg));
	m_falseConf = 0.5*Math.log((regularizer+weightedFalsePos)/(regularizer+weightedFalseNeg));

	m_storedLoss = Math.exp(-m_trueConf)*weightedTruePos +
	    Math.exp(m_trueConf)*weightedTrueNeg +
	    Math.exp(-m_falseConf)*weightedFalsePos +
	    Math.exp(m_falseConf)*weightedFalseNeg;
	return m_storedLoss;
    }

    public WeakClassifier buildLearnedClassifier() {
	return new ConditionalAndClassifier(m_cond, m_trueConf, m_falseConf);
    }

    public double getLearnedLoss() {
	return m_storedLoss;
    }
}