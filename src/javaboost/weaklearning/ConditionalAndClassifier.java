package javaboost.weaklearning;

import java.util.List;

import javaboost.conditioning.Conditional;
import javaboost.util.Utils;

public class ConditionalAndClassifier implements WeakClassifier{
    private List<Conditional> m_cond = null;
    private double m_trueConf = 0;

    public ConditionalAndClassifier(List<Conditional> conditions, double conf) {
	m_trueConf = conf;
	m_cond = conditions;
    }

    public double[] classify(float[][] data) {
	double[] output = new double[data.length];
	for(int i = 0; i < data.length; ++i) {
	    output[i] = Utils.isValid(data[i], m_cond) ? m_trueConf : 0;
	}
	return output;
    }

    public String toString() {
	String description = "And conf("+m_trueConf+"):";
	for(Conditional cond : m_cond) {
	    description.concat(m_cond.toString() + "&");
	}
	description = description.substring(0, description.length()-1);
	return description;
    }
}