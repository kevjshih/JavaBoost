package javaboost.boosting;



import java.util.List;
import java.util.HashSet;
import java.util.Set;

import javaboost.*;
import javaboost.weaklearning.WeakClassifier;
import javaboost.util.Utils;
import java.io.Serializable;

public class AdditiveClassifier implements Serializable, Classifier{
    private List<WeakClassifier> m_classifiers = null;

    public AdditiveClassifier(List<WeakClassifier> classifiers) {
	m_classifiers = classifiers;
    }

    public double[] classify(float[][] data) {
	double[] output = new double[data.length];
	for(int i = 0; i < data.length; ++i) {
	    output[i] = 0;
	}

	for(WeakClassifier wc: m_classifiers) {
	    output = Utils.addVectors(output, wc.classify(data));
	}

	return output;
    }

    public void listClassifiers() {
	for(int i = 0; i < m_classifiers.size(); ++i) {
	    System.out.println(m_classifiers.get(i));
	}
    }

    public int[] getColumnsUsed() {
	Set<Integer> cols = new HashSet<Integer>();
	for(WeakClassifier wc: m_classifiers) {
	    int[] wc_cols = wc.getColumns();
	    if(wc_cols != null) {
		for(int i =0; i < wc_cols.length; ++i) {
		    cols.add(wc_cols[i]);
		}
	    }
	}
	Object[] nonPrim =  cols.toArray();
	int[] out = new int[nonPrim.length];
	for(int i = 0; i < out.length; ++i) {
	    out[i] = ((Integer)nonPrim[i]).intValue();
	}
	return out;
    }

}