
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

using namespace std;
using namespace thrust;

/*
Fragen:
Wie auf einen bestimmten Index eines device_vector zugreifen?
paa: Wenn Arraysize nicht durch anzParts teilbar, fallen dann welche weg? Oder Runde ich?
*/

struct testCudaFunction {
	__device__ 
		void operator()(double &testVector) {
		testVector += 1;
	}
};

void printHostVector(thrust::host_vector<double> output) {
	for (auto i = 0; i < output.size(); i++) {
		cout << output[i] << ",";
	}
	cout << endl;
}

double getMean(vector<double> values) {
	int sum = 0;
	for (size_t i = 0; i < values.size(); i++) {
		sum += values[i];
	}
	return sum / values.size();
}

double standardDeviation(vector<double> values, double mean) {
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
	double sd = standardDeviation(values, mean);
	thrust::for_each(deviceTestVector.begin(), deviceTestVector.end(), zTransformation(mean, sd));
	hostTestVector = deviceTestVector;
	return hostTestVector;
}

/*struct Paa {
	int n;
	int m;
	device_vector<double> dSeries;
	Paa(int _n, int _m, device_vector<double> _dSeries) :n(_n), m(_m), dSeries(_dSeries) {};
	__device__
		double operator() (const unsigned int& it) const {
		double sum = 0;
		int begin = (n / m) * (it - 1) + 1;
		int end = (n / m) * it;
		for (int i = begin; i < end; i++) {
			sum += (double) dSeries;
		}
		return (m / n) * sum;
	}
};*/

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
		paa << <dimGrid, dimBlock >> > (dSeries, dSeriesResult, (double) sizeInputSeries, (double) parts);

		double* hSeriesResult = (double*)malloc(parts * sizeof(double));
		cudaMemcpy(hSeriesResult, dSeriesResult, parts * sizeof(double), cudaMemcpyDeviceToHost);
		return hSeriesResult;

		//Frage 1
		/*counting_iterator<unsigned int> it(0);
		Paa paa(sizeInputSeries, parts, dSeries);
		transform(it, it + parts, dSeriesResult.begin(), paa);*/
	}
}


int main(){
	vector<double> testArray1 = {2.02, 2.33, 2.99, 6.85, 9.20, 8.80, 7.50, 6.00, 5.85, 3.85, 4.85, 3.85, 2.22, 1.45, 1.34};
	vector<double> testArray2 = {-0.12, -0.16, -0.13,  0.28,  0.37,  0.39,  0.18,  0.09,  0.15, -0.06,  0.06, -0.07, -0.13, -0.18, -0.26};

	//printHostVector(zNormalization(testArray1));
	//printHostVector(zNormalization(testArray2));
	//thrust::device_vector<double> dApproximationArray(testArray3.begin(), testArray3.end());
	int parts = 7;
	double* hAppResult = approximation(zNormalization(testArray1).data(), testArray1.size(), parts);
	host_vector<double> hThrustAppResult(hAppResult, hAppResult + parts);
	printHostVector(hThrustAppResult);
	return 0;
}
