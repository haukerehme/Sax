
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "thrust/version.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

using namespace thrust::placeholders;


using namespace std;
using namespace thrust;

/*
Hinweis: Substring-Suche auf endlichen Automat auf der CUDA-Seite

Fragen:
Wie auf einen bestimmten Index eines device_vector zugreifen?
Quelle für Laufzeiten bestimmter Operator / Funktionen? Evtl sogar für thrust

*/

void printHostVector(host_vector<double> output) {
	for (auto i = 0; i < output.size(); i++) {
		cout << output[i] << ",";
	}
	cout << endl;
}

struct Print {
	__device__ void operator()(double &value) {
		printf("%f, ", value);
	}
};

void printDeviceVector(device_vector<double> output) {
	for_each(output.begin(), output.end(), Print());
}

/*double dGetMean(device_vector<double> values) {
	thrust::inclusive_scan(values.begin(), values.end(), values);
	//host_vector<double> hResult(values.begin(), values.end());
	vector<double> result(values.begin(), values.end());
	return result[result.size() - 1] / result.size();
}*/

double getMean(const vector<double>& values) {
	int sum = 0;
	for (size_t i = 0; i < values.size(); i++) {
		sum += values[i];
	}
	return sum / values.size();
}

double standardDeviation(const vector<double>& values, double mean) {
	double sd = 0.0;
	for (int i = 0; i < values.size(); i++){
		sd += (values[i] - mean) * (values[i] - mean);
	}
	sd /= values.size();
	sd = sqrt(sd);
	return sd;
}

struct zTransformation {
	double mean;
	double sd;
	zTransformation(double _mean, double _sd) : sd(_sd), mean(_mean) {};
	__device__
		void operator()(double &value) {
		value = (value - mean) / sd;
	}
};


thrust::host_vector<double> zNormalization(vector<double> values) {
	thrust::host_vector<double> hostTestVector(values.begin(), values.end());
	thrust::device_vector<double> deviceTestVector = hostTestVector;
	hostTestVector = deviceTestVector;
	double mean = getMean(values);
//	double mean = dGetMean(deviceTestVector);
	double sd = standardDeviation(values, mean);
	thrust::for_each(deviceTestVector.begin(), deviceTestVector.end(), zTransformation(mean, sd));
	hostTestVector = deviceTestVector;
	return hostTestVector;
}

struct Paa {
	double n;
	double m;
	double* dSeries;
	Paa(double _n, double _m, double* _dSeries) :n(_n), m(_m), dSeries(_dSeries) {};
	__device__
		double operator() (const unsigned int& it) const {
		double sum = 0;
		int begin = round((n / m) * it);
		int end = round((n / m) * (it + 1));
		for (int i = begin; i < end; i++) {
			sum += dSeries[i];
		}
		printf("%f\n", (m / n) * sum);
		return (m / n) * sum;
	}
};

__global__ void paa(double* dSeries, double* dSeriesResult, double n, double m) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	double sum = 0;
	int begin = round((n / m) * idx);
	int end = round((n / m) * (idx + 1));

	for (int i = begin; i < end; i++) {
		sum += dSeries[i];
	}
	dSeriesResult[idx] = (m / n) * sum;
}

double* approximation(double* hSeries, int sizeInputSeries, int parts) {
	double* dSeriesResult;
	cudaMalloc((void **)&dSeriesResult, parts * sizeof(double));
	int length = sizeInputSeries;
	if (length == parts) {
		return hSeries;
	}
	else{
		double* dSeries;
		cudaMalloc((void **)&dSeries, sizeInputSeries * sizeof(double));
		cudaMemcpy(dSeries, hSeries, sizeInputSeries * sizeof(double), cudaMemcpyHostToDevice);

		dim3 dimGrid(1);
		dim3 dimBlock(parts);
		//paa << <dimGrid, dimBlock >> > (dSeries, dSeriesResult, (double) sizeInputSeries, (double) parts);

		//Frage 1
		device_vector<double> dThrustSeries(hSeries, hSeries + sizeInputSeries);
		device_vector<double> dThrustSeriesResult(parts);
		counting_iterator<unsigned int> it(0);
		//Ist das hier wirklich schnell???
		Paa paa((double) sizeInputSeries, (double) parts, dThrustSeries.data().get());
		transform(it, it + parts, dThrustSeriesResult.begin(), paa);
		host_vector<double> hThrustSeriesResult(parts);
		hThrustSeriesResult = dThrustSeriesResult;
		printHostVector(hThrustSeriesResult);
		printDeviceVector(dThrustSeriesResult);
		//vector<double> tmp(hThrustSeriesResult.begin(), hThrustSeriesResult.end());
		// tmp.data();
		//return dThrustSeriesResult.data().get();
		//transform(dSeriesResult, dSeriesResult + parts, dSeries, dSeries, paa);

		double* seriesResult = (double*)malloc(parts * sizeof(double));
		cudaMemcpy(seriesResult, dSeriesResult, parts * sizeof(double), cudaMemcpyDeviceToHost);
		return seriesResult;
	}
}



struct Sax {
	__host__ __device__
		char operator()(const double &value) const {
		if (value < 0) {
			return (char) 'a';
		}else {
			return (char) 'b';
		}
	}
};


int main(){
	vector<double> testArray1 = {2.02, 2.33, 2.99, 6.85, 9.20, 8.80, 7.50, 6.00, 5.85, 3.85, 4.85, 3.85, 2.22, 1.45, 1.34};
	vector<double> testArray2 = {-0.12, -0.16, -0.13,  0.28,  0.37,  0.39,  0.18,  0.09,  0.15, -0.06,  0.06, -0.07, -0.13, -0.18, -0.26};
	
	//printHostVector(zNormalization(testArray1));
	//printHostVector(zNormalization(testArray2));
	//thrust::device_vector<double> dApproximationArray(testArray3.begin(), testArray3.end());
	int parts = 5;
	double* hAppResult = approximation(zNormalization(testArray1).data(), testArray1.size(), parts);
	host_vector<double> hThrustAppResult(hAppResult, hAppResult + parts);
	printHostVector(hThrustAppResult);

	hAppResult = approximation(zNormalization(testArray2).data(), testArray2.size(), parts);
	host_vector<double> hThrustAppResult2(hAppResult, hAppResult + parts);
	printHostVector(hThrustAppResult2);

	device_vector<double> dSaxInput = hThrustAppResult2;
	device_vector<char> dSaxResult(parts);
	Sax sax;
	thrust::transform(dSaxInput.begin(), dSaxInput.end(), dSaxResult.begin(), sax);

	return 0;
}
