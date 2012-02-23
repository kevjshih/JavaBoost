package javaboost.util;

import javaboost.conditioning.Conditional;
import javaboost.conditioning.LogicOps;
import java.util.List;

public final class Utils{
    private Utils() {
	// error if it is ever instantiated
	throw new AssertionError();
    }

    public static double[] scaleVector(final double[] a, final double factor) {
	double[] out = new double[a.length];
	for(int i = 0; i < a.length; ++i) {
	    out[i] = a[i]*factor;
	}
	return out;
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
    // subtracts b from a
    public static double[] subtractVectors(final double[] a, final double[] b) {
	assert(a.length == b.length);

	double[] output = new double[a.length];

	for(int i = 0; i < a.length; ++i) {
	    output[i] = a[i] - b[i];
	}

	return output;
    }


    public static double[] ebeDivideVectors(final double[] a, final double[] b) {
	assert(a.length == b.length);

	double[] output = new double[a.length];

	for(int i = 0; i < a.length; ++i) {
	    output[i] = a[i]/b[i];
	}

	return output;

    }

    public static float[] ebeMultiplyVectors(final float[] a, final float[] b) {
	assert(a.length == b.length);
	float[] output = new float[a.length];
	for(int i = 0; i < output.length; ++i) {
	    output[i] = a[i]*b[i];
	}
	return output;
    }


    public static double[] ebeMultiplyVectors(final double[] a, final  double[] b) {
	assert(a.length == b.length);
	double[] output = new double[a.length];
	for(int i = 0; i < output.length; ++i) {
	    output[i] = a[i]*b[i];
	}
	return output;
    }


    public static double[] ebeMultiplyVectors(final double[] a, final int[] b) {
	assert(a.length == b.length);
	double[] output = new double[a.length];
	for(int i = 0; i < output.length; ++i) {
	    output[i] = a[i]*b[i];
	}
	return output;
    }


    public static float[] ebeMultiplyVectors(final float[] a, final int[] b) {
	assert(a.length == b.length);
	float[] output = new float[a.length];
	for(int i = 0; i < output.length; ++i) {
	    output[i] = a[i]*b[i];
	}
	return output;
    }

    public static double innerproductVectors(final double[] a, final double[] b) {
	assert(a.length == b.length);
	double[] prod = ebeMultiplyVectors(a,b);
	return vectorSum(prod);

    }

    public static double vectorSum(final double[] a) {
	double out = 0;
	for(int i = 0; i < a.length; ++i) {
	    out+= a[i];
	}
	return out;
    }

    public static double[][] transposeMatrix(final double[][] m) {
	int rows = m.length;
	int cols = m[0].length;
	double[][] out = new double[cols][rows];
	for(int i = 0; i < rows; ++i) {
	    for(int j = 0; j < cols; ++j) {
		out[j][i] = m[i][j];
	    }
	}
	return out;
    }

    // m * x
    public double[] operate(final double[][] m,final double[] x) {

	int rows = m.length;
	int cols = m[0].length;
	assert(cols == x.length);

	double[] out = new double[x.length];
	for(int i = 0; i < m.length; ++i) {
	    for(int j = 0; j < m[i].length; ++j) {
		out[i]+= m[i][j]*x[j];
	    }
	}
	return out;
    }


    // matrix a times matrix b
    public static double[][] multiplyMatrices(final double[][] a, final double [][]b) {
	int rowsA = a.length;
	int colsA = a[0].length;
	int rowsB = b.length;
	int colsB = b[0].length;
	assert(colsA == rowsB);
	double[][] out = new double[rowsA][colsB];
	for(int i = 0; i < rowsA; ++i) {
	    for(int j = 0; j < colsB; ++j) {
		for(int k = 0; k < colsA; ++k) {
		    out[i][j]+=a[i][k]*b[k][j];
		}
	    }
	}
	return out;
    }

    public static double[][] createIdentity(int size) {
	double[][] out = new double[size][size];
	for(int i = 0; i < size; ++i) {
	    out[i][i] = 1.0;
	}
	return out;
    }

    public static double[][] createDiagonalMatrix(final double[] diag) {
	double[][] out = new double[diag.length][diag.length];
	for(int i = 0; i < diag.length; ++i) {
	    out[i][i] = diag[i];
	}
	return out;
    }

    public static double getL1Norm(final double[] a) {
	double norm = 0;
	for(int i = 0; i < a.length; ++i) {
	    norm += Math.abs(a[i]);
	}
	return norm;
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

    public static double mean(double[] a) {
	double sum = 0;
	for(int i = 0; i < a.length; ++i) {
	    sum += a[i];
	}
	return sum/a.length;
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