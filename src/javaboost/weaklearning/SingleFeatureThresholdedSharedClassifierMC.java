package javaboost.weaklearning;
import java.io.Serializable;
import javaboost.*;

import java.util.Set;
import java.util.Map;

public SingleFeatureThresholdedSharedClassifierMC extends WeakClassifierMC{
    private float m_a_s;
    private float m_b_s;
    private float m_threshold;
    private int m_featColumn;
    private Set<Integer> m_classes = null;
    private Map<Integer, Float> m_biasMap = null;

    public SingleFeatureThresholdedSharedClassifierMC(int featColumn, float threshold,  float a_s, float b_s,
						      Set<Integer> classes, Map<Integer, Float> biasMap) {
	m_a_s = a_s;
	m_b_s = b_s;
	m_classes = classes;
	m_biasMap = biasMap;
	m_featColumn = featColumn;
    }

    public float[] classify(final float[][] data, int classId) {
	float[] out = new float[data.length];
	for(int i = 0; i < data.length; ++i) {
	    if(m_biasMap.contains(classId)) {
		out[i] = m_biasMap.get(classId);
	    } else {
		if(data[i][featColumn] >= m_threshold) {
		    out[i] = m_a_s;
		} else {
		    out[i] = m_b_s;
		}
	    }
	}
	return out;
    }

    public int[] getColumns() {
	int[] out = {m_featColumn};
	return out;
    }
}
