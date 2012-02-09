package javaboost.util;

import javaboost.conditioning.Conditional;
import java.util.List;

public final class Utils{
    private Utils() {
	// error if it is ever instantiated
	throw new AssertionError();
    }

    public static float[] addVectors(final float[] a, final float[] b) {
	assert(a.length == b.length);

	float[] output = new float[a.length];

	for(int i = 0; i < a.length; ++i) {
	    output[i] = a[i] + b[i];
	}

	return output;

    }

    public static double[] addVectors(final double[] a, final double[] b) {
	assert(a.length == b.length);

	double[] output = new double[a.length];

	for(int i = 0; i < a.length; ++i) {
	    output[i] = a[i] + b[i];
	}

	return output;
    }

    public static float[] dotMultiplyVectors(final float[] a, final float[] b) {
	assert(a.length == b.length);
	float[] output = new float[a.length];
	for(int i = 0; i < output.length; ++i) {
	    output[i] = a[i]*b[i];
	}
	return output;
    }


    public static double[] dotMultiplyVectors(final double[] a, final  double[] b) {
	assert(a.length == b.length);
	double[] output = new double[a.length];
	for(int i = 0; i < output.length; ++i) {
	    output[i] = a[i]*b[i];
	}
	return output;
    }


    public static double[] dotMultiplyVectors(final double[] a, final int[] b) {
	assert(a.length == b.length);
	double[] output = new double[a.length];
	for(int i = 0; i < output.length; ++i) {
	    output[i] = a[i]*b[i];
	}
	return output;
    }


    public static float[] dotMultiplyVectors(final float[] a, final int[] b) {
	assert(a.length == b.length);
	float[] output = new float[a.length];
	for(int i = 0; i < output.length; ++i) {
	    output[i] = a[i]*b[i];
	}
	return output;
    }




    public static void normalizeVector(float[] a) {
	float sum = 0;
	for(int i = 0; i < a.length; ++i) {
	    sum += a[i];
	}
	for(int i = 0; i < a.length; ++i) {
	    a[i] = a[i]/sum;
	}
    }


    public static void normalizeVector(double[] a) {
	double sum = 0;
	for(int i = 0; i < a.length; ++i) {
	    sum += a[i];
	}
	for(int i = 0; i < a.length; ++i) {
	    a[i] = a[i]/sum;
	}
    }

    public static boolean isValid(float[] features, List<Conditional> conditions, byte logicOp) {
	switch(logicOp) {
	case LogicOps.AND:
	    for(Conditional cond : conditions) {
		if(!cond.isValid(features)) {
		    return false;
		}
	    }
	    return true;
	case LogicOps.XOR:
	    int trueCount = 0;
	    for(Conditional cond : conditions) {
		if(cond.isValid(features)) {
		    ++trueCount;
		    if(trueCount > 1) {
			return false;
		    }
		}

	    }
	    if(trueCount == 0){
		return false;
	    }
	    else{
		return true;
	    }
	default:
	    return false;

	}

    }

}