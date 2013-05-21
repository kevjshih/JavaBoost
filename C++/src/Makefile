CC=g++
CFLAGS=-c -g -Wall -Werror

weaklearningtest: additiveclassifier.o weaklearningtest.o singlefeaturemultithresholdedsigmoidlearner.o singlefeaturesigmoidclassifier.o utils.o boost.o
	$(CC) weaklearningtest.o singlefeaturemultithresholdedsigmoidlearner.o singlefeaturesigmoidclassifier.o utils.o boost.o additiveclassifier.o -o weaklearningtest

utilstest: utilstest.o utils.o
	$(CC) utilstest.o utils.o -o utilstest

weaklearningtest.o: weaklearningtest.cpp utils.h boost.h singlefeaturesigmoidclassifier.h singlefeaturemultithresholdedsigmoidlearner.h weaklearner.h classifier.h
	$(CC) $(CFLAGS) weaklearningtest.cpp

utilstest.o: utilstest.cpp
	$(CC) $(CFLAGS) utilstest.cpp

additiveclassifier.o: additiveclassifier.h additiveclassifier.cpp classifier.h utils.h
	$(CC) $(CFLAGS) additiveclassifier.cpp

boost.o: boost.h boost.cpp weaklearner.h utils.h classifier.h additiveclassifier.h
	$(CC) $(CFLAGS) boost.cpp

utils.o: utils.cpp utils.h
	$(CC) $(CFLAGS) utils.cpp

singlefeaturemultithresholdedsigmoidlearner.o: weaklearner.h singlefeaturemultithresholdedsigmoidlearner.cpp  singlefeaturemultithresholdedsigmoidlearner.h utils.h
	$(CC) $(CFLAGS)  singlefeaturemultithresholdedsigmoidlearner.cpp	

singlefeaturesigmoidclassifier.o: singlefeaturesigmoidclassifier.h singlefeaturesigmoidclassifier.cpp utils.h classifier.h 
	$(CC) $(CFLAGS) singlefeaturesigmoidclassifier.cpp




.PHONY : clean

clean:
	rm -rf *.o utilstest weaklearningtest

