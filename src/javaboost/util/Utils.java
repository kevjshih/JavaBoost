package javaboost.util;

import javaboost.conditioning.Conditional;
import javaboost.conditioning.LogicOps;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.SortedSet;

public final class Utils{
    private Utils() {
	// error if it is ever instantiated
	throw new AssertionError();
    }

    public static double fastExp(double val) {
	final long tmp = (long) (1512775 * val + 1072632447);
	return Double.longBitsToDouble(tmp << 32);
    }


    public static double[] getBalancedWeights(final int[] labels) {
	double[] weights = new double[labels.length];
	int numPos = 0;
	int numNeg = 0;

	for(int i = 0; i < labels.length; ++i) {
	    if(labels[i] == 1) {
		++numPos;
	    }else if(labels[i] == -1){
		++numNeg;
	    }
	}
	double posWt = 0.5/numPos;
	double negWt = 0.5/numNeg;
	for(int i = 0; i < labels.length; ++i) {
	    if(labels[i] == 1) {
		weights[i] = posWt;
	    }else if(labels[i] == -1){
		weights[i] = negWt;
	    }else {
		weights[i] = 0;
	    }
	}
	return weights;

    }

    public static double[] scaleVector(final double[] a, final double factor) {
	double[] out = new double[a.length];
	for(int i = 0; i < a.length; ++i) {
	    out[i] = a[i]*factor;
	}
	return out;
    }


    public static void scaleVectorInPlace(double[] a, final double factor) {
	for(int i = 0; i < a.length; ++i) {
	    a[i] = a[i]*factor;
	}
    }


    public static double[][] scaleMatrix(final double[][] m, final double factor) {
	double[][] out = new double[m.length][m[0].length];
	for(int i = 0; i < m.length; ++i) {
	    for(int j = 0; j < m[0].length; ++j) {
		out[i][j] = m[i][j]*factor;
	    }
	}
	return out;
    }

    public static void scaleMatrixInPlace(double[][] m, final double factor) {
	for(int i = 0; i < m.length; ++i) {
	    for(int j = 0; j < m[0].length; ++j) {
		m[i][j]*= factor;
	    }
	}
    }

    public static float[] addVectors(final float[] a, final float[] b) {
	assert(a.length == b.length);

	float[] output = new float[a.length];

	for(int i = 0; i < a.length; ++i) {
	    output[i] = a[i] + b[i];
	}

	return output;

    }

    public static void addVectorsInPlace(float[] a, final float[] b) {
	assert(a.length == b.length);

	for(int i = 0; i < a.length; ++i) {
	    a[i] = a[i] + b[i];
	}

    }

    public static void addVectorsInPlace(double[] a, final double[] b) {
	assert(a.length == b.length);

	for(int i = 0; i < a.length; ++i) {
	    a[i] = a[i] + b[i];
	}

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


    public static void subtractVectorsInPlace(double[] a, final double[] b) {
	assert(a.length == b.length);

	for(int i = 0; i < a.length; ++i) {
	    a[i] = a[i] - b[i];
	}

    }



    public static double[][] addMatrices(final double[][] a, final double[][] b) {
	assert(a.length == b.length);
	assert(a[0].length == b[0].length);
	double[][] out = new double[a.length][a[0].length];
	for(int i = 0; i < a.length; ++i) {
	    for(int j = 0; j < a[0].length ; ++j) {
		out[i][j] = a[i][j] + b[i][j];
	    }
	}
	return out;

    }

    public static void addMatricesInPlace(double[][] a, final double[][] b) {
	assert(a.length == b.length);
	assert(a[0].length == b[0].length);
	for(int i = 0; i < a.length; ++i) {
	    for(int j = 0; j < a[0].length ; ++j) {
		a[i][j] = a[i][j] + b[i][j];
	    }
	}
    }


    public static double[][] subtractMatrices(final double[][] a, final double[][]b) {
	assert(a.length == b.length);
	assert(a[0].length == b[0].length);
	double[][] out = new double[a.length][a[0].length];
	for(int i = 0; i < a.length; ++i) {
	    for(int j = 0; j < a[0].length ; ++j) {
		out[i][j] = a[i][j] - b[i][j];
	    }
	}
	return out;

    }


    public static void subtractMatricesInPlace(double[][] a, final double[][]b) {
	assert(a.length == b.length);
	assert(a[0].length == b[0].length);

	for(int i = 0; i < a.length; ++i) {
	    for(int j = 0; j < a[0].length ; ++j) {
		a[i][j] = a[i][j] - b[i][j];
	    }
	}
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

    // assumes the last column is the bias
    public static double evaluateWeightsOnData(final float[] data,
					       final double[] weights,
					       final int[] columns) {
	double out = 0;
	for(int i = 0; i < data.length; ++i) {
	    out += data[columns[i]]*weights[i];
	}
	out+=weights[weights.length-1];
	return out;
    }

    public static double innerProductVectors(final double[] a, final double[] b) {
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

    public static float vectorSum(final float[] a) {
	float out = 0;
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

    public static int[][] transposeMatrix(final int[][] m) {
	int rows = m.length;
	int cols = m[0].length;
	int[][] out = new int[cols][rows];
	for(int i = 0; i < rows; ++i) {
	    for(int j = 0; j < cols; ++j) {
		out[j][i] = m[i][j];
	    }
	}
	return out;
    }

    // m * x
    public static double[] operate(final double[][] m,final double[] x) {

	int rows = m.length;
	int cols = m[0].length;
	assert(cols == x.length);

	double[] out = new double[rows];
	for(int i = 0; i < rows; ++i) {
	    for(int j = 0; j < cols; ++j) {
		out[i]+= m[i][j]*x[j];
	    }
	}
	return out;
    }

    public static void operateWithTarget(double[] target, final double[][] m, final double[] x){

	int rows = m.length;
	int cols = m[0].length;
	assert(cols == x.length);

	for(int i = 0; i < rows; ++i) {
	    target[i] = 0;
	    for(int j = 0; j < cols; ++j) {
		target[i]+= m[i][j]*x[j];
	    }
	}

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


    // matrix a times matrix b
    public static void multiplyMatricesWithTarget(double[][] target, final double[][] a, final double [][]b) {
	int rowsA = a.length;
	int colsA = a[0].length;
	int rowsB = b.length;
	int colsB = b[0].length;
	assert(colsA == rowsB);
	assert(target.length == rowsA && target[0].length == colsB);

	for(int i = 0; i < rowsA; ++i) {
	    for(int j = 0; j < colsB; ++j) {
		target[i][j] = 0;
		for(int k = 0; k < colsA; ++k) {
		    target[i][j]+=a[i][k]*b[k][j];
		}
	    }
	}

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

    public static double getL2Norm(final double[] a) {
	double norm = 0;
	for(int i = 0; i < a.length; ++i) {
	    norm += a[i]*a[i];
	}
	return Math.sqrt(norm);
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
    public static float mean(float[] a) {
	float sum = 0;
	for(int i = 0; i < a.length; ++i) {
	    sum += a[i];
	}
	return sum/a.length;
    }

    public static float std(float[] data) {
	double datamean = (double)mean(data);
	double out = 0;
	for(int i = 0; i < data.length; ++i) {
	    double diff = data[i] - datamean;
	    out += diff*diff;
	}
	out/=data.length;
	out = Math.sqrt(out);
	return (float)out;
    }

    public static double std(double[] data) {
	double datamean = mean(data);
	double out = 0;
	for(int i = 0; i < data.length; ++i) {
	    double diff = data[i] - datamean;
	    out += diff*diff;
	}
	out/=data.length;
	out = Math.sqrt(out);
	return out;
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

    public static int[] intSetUnion(final int[] a, final int[] b) {
	if(a == null && b == null) {
	    return null;
	}
	if(a == null) {
	    return (int[])b.clone();
	}
	if(b == null) {
	    return (int[])a.clone();
	}
	Set<Integer> elts = new TreeSet<Integer>();
	for(int i = 0; i < a.length; ++i) {
	    elts.add(a[i]);
	}
	for(int i = 0; i < b.length; ++i) {
	    elts.add(b[i]);
	}
	int[] out = new int[elts.size()];
	int idx = 0;
	for(Integer inte : elts) {
	    out[idx] = inte;
	    ++idx;
	}
	return out;
    }



}
