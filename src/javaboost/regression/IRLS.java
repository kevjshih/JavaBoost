package javaboost.regression;
import javaboost.util.Utils;
public final class IRLS{


    public static double[] solve(final double[][] X, final double[] y, double[] example_w,  double lambda) {
	double[] _Beta = new double[X[0].length];
	double[] ones = new double[y.length];
	boolean noW = false;
	if(example_w == null) {
	    noW = true;
	    example_w = new double[y.length];
	}
	for(int i = 0; i < ones.length; ++i) {
	    ones[i] = 1.0;
	    if(noW) {
		example_w[i] = 1;
	    }
	}



	double change = 20.0;
	double epsilon = 0.00001;

	double[][] lambdaI = Utils.scaleMatrix(Utils.createIdentity(X[0].length), lambda);

	double[][] XtWX = new double[X[0].length][X[0].length];
	double[][] A = new double[X[0].length][X[0].length];
	double[] XopB = new double[X.length];


	while(change > epsilon) {

	    Utils.operateWithTarget(XopB, X, _Beta);
	    double[] negExp = Utils.scaleVector(XopB, -1);


	    for(int i = 0; i < negExp.length; ++i) {
		negExp[i] = Math.exp(negExp[i]);
	    }

	    double[] u = Utils.ebeDivideVectors(ones, Utils.addVectors(ones,negExp));

	    double[] w = Utils.ebeMultiplyVectors(u, Utils.subtractVectors(ones, u));

	    double[] U = Utils.addVectors(XopB, Utils.ebeDivideVectors(Utils.subtractVectors(y,u),w));
	    for(int i =0; i < U.length; ++i) {

		w[i]*=example_w[i];
	    }
	    setXtWX(XtWX, X, w);
	    Utils.addMatricesInPlace(XtWX, lambdaI);

	    double[] b = Utils.operate(Utils.transposeMatrix(X), Utils.ebeMultiplyVectors(w, U));

	    double[] _Beta_new = ConjugateGradient.solve(XtWX, b, _Beta, 30);
	    change = Utils.getL2Norm(Utils.subtractVectors(_Beta_new, _Beta));

	    _Beta = _Beta_new;
	}

	return _Beta;

    }

    public static void setXtWX(double[][] target, final double[][] X, final  double[]W_diag) {
	assert(target.length == X[0].length);
	assert(target[0].length == target.length);

	for(int i = 0; i < target[0].length; ++i) {
	    for(int j = 0; j <target[0].length; ++j) {
		target[i][j] = 0;
		for(int l = 0; l < X.length; ++l) {
		    target[i][j] += X[l][i]*W_diag[l]*X[l][j];
		}

	    }
	}
    }

}