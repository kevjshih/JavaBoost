package javaboost.weaklearning;

import java.util.List;

import javaboost.conditioning.*;
import javaboost.util.Utils;

public class LogicalClassifier implements WeakClassifier{
    private List<Conditional> m_cond = null;
    private double m_trueConf = 0;
    private double m_falseConf = 0;
    private byte m_logicOp;

    public LogicalClassifier(List<Conditional> conditions, double trueConf, double falseConf, byte logicOp) {
	m_trueConf = trueConf;
	m_falseConf = falseConf;
	m_cond = conditions;
	m_logicOp = logicOp;
    }

    public double[] classify(float[][] data) {
	double[] output = new double[data.length];
	for(int i = 0; i < data.length; ++i) {
	    output[i] = Utils.isValid(data[i], m_cond, m_logicOp) ? m_trueConf : m_falseConf;
	}
	return output;
    }

    public String toString() {
	String description = LogicOps.getName(m_logicOp) + " conf T:"+m_trueConf+" F:"+m_falseConf+":";
	for(Conditional cond : m_cond) {
	    description = description.concat(cond.toString() + "&");
	}
	description = description.substring(0, description.length()-1);
	return description;
    }

    public int[] getColumns(){
	return null;
    }
}