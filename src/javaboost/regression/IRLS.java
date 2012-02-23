package javaboost.regression;
import org.apache.commons.math.linear.*;

public final class IRLS{


    public static RealVector solve(RealMatrix X, RealVector y, double lambda) {
	double[] beta_init = new double[X.getColumnDimension()];
	double[] ones_init = new double[X.getColumnDimension()];
	for(int i = 0; i < beta_init.length; ++i) {
	    beta_init[i] = 0;
	    ones_init[i] = 1.0;
	}

	RealVector Beta = new ArrayRealVector(beta_init);
	RealVector ones = new ArrayRealVector(ones_init);

	Array2DRowRealMatrix W = new Array2DRowRealMatrix(X.getColumnDimension(), X.getColumnDimension());

	double change = 20.0;
	double epsilon = 0.0001;

	while(change > epsilon) {
	    RealVector XopB = X.operate(Beta);
	    RealVector exponent = XopB.mapMultiply(-1);

	    double[] expRef = exponent.toArray();
	    for(int i = 0; i < expRef.length; ++i) {
		expRef[i] = Math.exp(expRef[i]);
	    }

	    RealVector u = ones.ebeDivide(ones.add(exponent));

	    RealVector w = u.ebeMultiply(ones.subtract(u));

	    RealVector U = XopB.add(y.subtract(u).ebeDivide(w));

	    double[][] Welts = W.getDataRef();
	    for(int i = 0; i < w.getDimension(); ++i) {
		Welts[i][i]= w.getEntry(i);
	    }

	    RealMatrix XttimesW = X.transpose().multiply(W);

	    RealMatrix A = XttimesW.multiply(X).add(MatrixUtils.createRealIdentityMatrix(X.getColumnDimension()).scalarMultiply(lambda));
	    RealVector b = XttimesW.operate(U);
	    RealVector Beta_new = ConjugateGradient.solve(A, b, Beta, 100);
	    change = Beta_new.subtract(Beta).getL1Norm();
	    Beta = Beta_new;

	}

	return Beta;

    }

}