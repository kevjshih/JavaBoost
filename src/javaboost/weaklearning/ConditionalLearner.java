package javaboost.weaklearning;

import java.util.List;

import javaboost.conditioning.Conditional;
import javaboost.util.Utils;
import javaboost.conditioning.LogicOps;

public class ConditionalLearner implements WeakLearner{
    private List<Conditional> m_cond = null;
    private Conditional m_decisionBound = null;

    private double m_trueConf = 0;
    private double m_falseConf = 0;

    private double m_storedLoss = 0;

    public ConditionalLearner(List<Conditional> conditions, Conditional decisionBound) {
	m_cond = conditions;
	m_decisionBound = decisionBound;
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

       // everything else that fails the initial conditions
       double weightedOthers = 0;

       double regularizer = 1.0/data.length;

       for(int i = 0; i < data.length; ++i) {
	   if(Utils.isValid(data[i], m_cond, LogicOps.AND)) {
	       if(m_decisionBound.isValid(data[i])) {
		   if(labels[i] >= 0){
		       weightedTruePos += weights[i];
		   }else{
		       weightedTrueNeg += weights[i];
		   }
	       }else{
		   if(labels[i] >= 0){
		       weightedFalsePos += weights[i];
		   }else{
		       weightedFalseNeg += weights[i];
		   }
	       }

	   } else {
	       weightedOthers += weights[i];
	   }
       }
       m_trueConf = 0.5*Math.log((regularizer+weightedTruePos)/(regularizer+weightedTrueNeg));
       m_falseConf = 0.5*Math.log((regularizer+weightedFalsePos)/(regularizer+weightedFalseNeg));

       m_storedLoss = Math.exp(-m_trueConf)*weightedTruePos +
	   Math.exp(m_trueConf)*weightedTrueNeg +
	   Math.exp(-m_falseConf)*weightedFalsePos +
	   Math.exp(m_falseConf)*weightedFalseNeg +
	   weightedOthers;

       m_storedLoss = m_storedLoss/(1+m_storedLoss);

       return m_storedLoss;
   }

    public WeakClassifier buildLearnedClassifier() {
	return new ConditionalClassifier(m_cond, m_decisionBound, m_trueConf, m_falseConf);
    }

    public double getLearnedLoss() {
	return m_storedLoss;
    }

}