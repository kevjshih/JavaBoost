mex -L../C++/src CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" -lsigboost sigboost_mex.cpp -I../C++/src
mex -L../C++/src -lsigboost sigboost_classify_mex.cpp -I../C++/src