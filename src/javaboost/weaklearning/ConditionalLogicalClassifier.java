package javaboost.weaklearning;

import java.util.List;

import javaboost.conditioning.Conditional;
import javaboost.util.Utils;

public class ConditionalAndClassifier implements WeakClassifier{
    private List<Conditional> m_cond = null;
    private double m_trueConf = 0;
    private double m_falseConf = 0;

    public ConditionalAndClassifier(List<Conditional> conditions, double trueConf, double falseConf) {
	m_trueConf = trueConf;
	m_falseConf = falseConf;
	m_cond = conditions;
    }

    public double[] classify(float[][] data) {
	double[] output = new double[data.length];
	for(int i = 0; i < data.length; ++i) {
	    output[i] = Utils.isValidAnd(data[i], m_cond) ? m_trueConf : 0;
	}
	return output;
    }

    public String toString() {
	String description = "And conf T:"+m_trueConf+" F:"+m_falseConf+":";
	for(Conditional cond : m_cond) {
	    description = description.concat(cond.toString() + "&");
	}
	description = description.substring(0, description.length()-1);
	return description;
    }
}