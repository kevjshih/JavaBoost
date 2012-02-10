package javaboost.weaklearning;

import java.util.List;

import javaboost.conditioning.Conditional;
import javaboost.util.Utils;
import javaboost.conditioning.LogicOps;

public class ConditionalClassifier implements WeakClassifier{
    private List<Conditional> m_cond = null;
    private Conditional m_decisionBound = null;

    private double m_trueConf = 0;
    private double m_falseConf = 0;

    public ConditionalClassifier(List<Conditional> conditions,
				 Conditional decisionBound,
				 double trueConf,
				 double falseConf) {
	m_cond = conditions;
	m_decisionBound = decisionBound;
	m_trueConf = trueConf;
	m_falseConf = falseConf;
    }

    public double[] classify(float[][] data) {
	double[] output = new double[data.length];
	for(int i = 0; i < data.length; ++i) {
	    if(Utils.isValid(data[i], m_cond, LogicOps.AND)) {
		output[i] = m_decisionBound.isValid(data[i]) ? m_trueConf : m_falseConf;
	    } else {
		output[i] = 0;
	    }
	}
	return output;

    }

    public String toString() {
	String description = "Conditional" +
	    " conf T:"+m_trueConf+
	    " F:"+m_falseConf+
	    ": primary " + m_decisionBound.toString() + " initial ";
	for(Conditional cond : m_cond) {
	    description = description.concat(cond.toString() + "&");
	}
	description = description.substring(0, description.length()-1);
	return description;
    }
}