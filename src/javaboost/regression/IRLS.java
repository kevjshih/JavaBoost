package javaboost.regression;
import javaboost.util.Utils;
public final class IRLS{


    public static double[] solve(double[][] X, double[] y, double lambda) {
	double[] _Beta = new double[X.length];
	double[] ones = new double[X[0].length];
	for(int i = 0; i < _Beta.length; ++i) {
	    _Beta[i] = 0;
	    ones[i] = 1.0;
	}

	double[][] W = new double[X.length][X[0].length];

	double change = 20.0;
	double epsilon = 0.0001;


	double[][] lambdaI = Utils.scaleMatrix(Utils.createIdentity(X[0].length), lambda);

	while(change > epsilon) {


	    double[] XopB = Utils.operate(X, _Beta);
	    double[] negExp = Utils.scaleVector(XopB, -1);


	    for(int i = 0; i < negExp.length; ++i) {
		negExp[i] = Math.exp(negExp[i]);
	    }

	    double[] u = Utils.ebeDivideVectors(ones, Utils.addVectors(ones,negExp));

	    double[] w = Utils.ebeMultiplyVectors(u, Utils.subtractVectors(ones, u));

	    double[] U = Utils.addVectors(XopB, Utils.ebeDivideVectors(Utils.subtractVectors(y,u),w));

	    for(int i = 0; i < w.length; ++i) {
		W[i][i]= w[i];
	    }

	    double[][] XttimesW = Utils.multiplyMatrices(Utils.transposeMatrix(X),W);

	    double[][] A = Utils.multiplyMatrices(XttimesW, X);

	    A = Utils.addMatrices(A, lambdaI);

	    double[] b = Utils.operate(XttimesW,U);

	    double[] _Beta_new = ConjugateGradient.solve(A, b, _Beta, 100);
	    change = Utils.getL1Norm(Utils.subtractVectors(_Beta_new, _Beta));

	    _Beta = _Beta_new;

	}

	return _Beta;

    }

}