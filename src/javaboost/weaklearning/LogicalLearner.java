package javaboost.weaklearning;

import java.util.List;

import javaboost.conditioning.Conditional;
import javaboost.util.Utils;
import javaboost.conditioning.LogicOps;

public class LogicalLearner implements WeakLearner{


    private List<Conditional> m_cond = null;
    private double m_trueConf = 0;
    private double m_falseConf = 0;

    private double m_storedLoss = 0;

    private byte m_logicOp;

    public LogicalLearner(List<Conditional> conditions, byte logicOp) {
	m_cond = conditions;
	m_logicOp = logicOp;
    }

    public double train(final float[][] data,final int[] labels,final double[] weights) {
	m_trueConf = 0;
	m_falseConf = 0;
	// if true
	double weightedTruePos = 0;
	double weightedTrueNeg = 0;
	// if false;
	double weightedFalsePos = 0;
	double weightedFalseNeg = 0;

	double regularizer = 1.0/data.length;

	for(int i = 0; i < data.length; ++i) {
	    if(Utils.isValid(data[i], m_cond, m_logicOp)) {
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

	m_storedLoss = m_storedLoss/(1+m_storedLoss);

	return m_storedLoss;
    }

    public WeakClassifier buildLearnedClassifier() {
	return new LogicalClassifier(m_cond, m_trueConf, m_falseConf, m_logicOp);
    }

    public double getLearnedLoss() {
	return m_storedLoss;
    }
}