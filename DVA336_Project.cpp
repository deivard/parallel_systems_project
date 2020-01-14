// DVA336_Project.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include <immintrin.h>
#include <time.h>
#include <omp.h>
//#include <mm_malloc.h>
#define _USE_MATH_DEFINES
#include <math.h>
#define N 10000000

using namespace std;

// Non-parallel version
inline double radians(double degree) {
    return (degree * M_PI) / 180;
}

// Non-parallel version
// Calculate the great circle distance between two points 
// on the earth (specified in decimal degrees)
inline double haversine(double lat1, double lon1, double lat2, double lon2) {
    // convert decimal degrees to radians 
	lat1 = radians(lat1);
	lon1 = radians(lon1);
	lat2 = radians(lat2);
	lon2 = radians(lon2);

    // haversine formula 
	double dlon = lon2 - lon1;
	double dlat = lat2 - lat1;
	double a = pow(sin(dlat / 2), 2) + cos(lat1) * cos(lat2) * pow(sin(dlon / 2), 2);
	double c = 2 * asin(sqrt(a));
	double r = 6372.8; // Radius of earth in kilometers. 
	return c * r;
}

// Creates the data
double ** createLocations(int n) {
	double ** locations = (double**)malloc(sizeof(double) * 2 * n);
	double stepSize = 360.0 / (double)N;

	double baseLat = 0.0;
	double baseLon = -180.0;

	for (int i = 0; i < n; ++i) {
		double *newLoc = (double*)malloc(sizeof(double) * 2);
		newLoc[0] = 0.0; //baseLat + stepSize * (double)(i+1);
		newLoc[1] = baseLon + stepSize * (double)(i+1);
		locations[i] = newLoc;
	}

	return locations;
}


// Non-parallel version
double calculateDistance(double ** locations, int n) {
	double distance = 0;
	for (int i = 0; i < n-1; i++) {
		distance += haversine(locations[i][0], locations[i][1], locations[i + 1][0], locations[i + 1][1]);
	}
	return distance;
}


// Functions related to parallel implementaion:
// --------------------------------------------

// Converts the locations array to a format that is friendlier for SIMD
// This function does not need to be parallelized since it only reads/writes to memory
void convertLocationsArray(double * latitudesDst, double * longitudesDst, double ** locations, int n) {
	for (int i = 0; i < n; ++i) {
		latitudesDst[i] = locations[i][0];
		longitudesDst[i] = locations[i][1];
	}
}


// SIMD version
// Converts 4 degrees into 4 radians
inline __m256d _mm256d_radians(double *degrees) {
	__m256d pi = _mm256_set1_pd(M_PI);
	__m256d half = _mm256_set1_pd(180);
	return _mm256_div_pd(_mm256_mul_pd(_mm256_load_pd(degrees), pi), half);
}

// SIMD power of 
inline __m256d _mm256_pow_pd(__m256d vec, int pow) {
	if (pow == 0) return _mm256_set1_pd(1.0);
	__m256d result = vec;
	for (size_t i = 1; i < pow; i++) {
		result = _mm256_mul_pd(result, vec);
	}
	return result;
}

// SIMD Modulo with doubles
inline __m256d _mm256_modulo_pd(__m256d vec, double mod) {
	// d % p = d - (int(double(d) / double (p))*p
	__m256d _mod = _mm256_set1_pd(mod);
	__m256d quotient = _mm256_div_pd(vec, _mod);
	__m256d product = _mm256_mul_pd(_mm256_floor_pd(quotient), _mod);
	return _mm256_sub_pd(vec, product);
}

// SADLY, we did not even need this function in the end. But it does some cool stuff and we learned A LOT while making it at least
// Creates a mask with 1.0s on index where elements are positive, and -1.0s where the elements are negative
inline __m256d _mm256d_sign_mask(__m256d vec) {
	// First we find out which values are negative:
	__m256d _mask = _mm256_cmp_pd(vec, _mm256_set1_pd(0), 1);
	// Then we create a mask to convert the negative values to positive
	// by multiplying by -1, this can be conveniently achieved by casting the _mask to a m256i
	// and then bit shifting and an XOR to create a -1.0 double
	// The bit pattern we want is: 1 011 1111 1111 0000 0000 0000 ... 0000
	__m256i _iMask = *(__m256i*)&_mask;
	_iMask = _mm256_slli_epi64(_iMask, 52);
	// Now we have: 1 111 1111 1111 0000 ...
	// So we need to create a bit pattern that we can XOR with to set the 2nd left-most bit to 0,
	// The bit pattern we seek is: 0 100 0000 0000 ...
	// If we set _bitPattern to the min of a 64 bit int (-9,223,372,036,854,775,806), we get this pattern: 1 000 0000 ... 
	__m256i _bitPattern = _mm256_set1_epi64x(-LLONG_MIN);
	// So we only need to shift one bit to the right
	_bitPattern = _mm256_srli_epi64(_bitPattern, 1);
	// And then XOR iMask and bitPattern to get -1.0 doubles on the indexes where the negative angles are
	_iMask = _mm256_xor_si256(_iMask, _bitPattern);
	// The problem now is that we get 2.0s where there should be 1.0s. 
	// First we turn the 2s into 0s by XORing with the inverted bit representation of 2.0,
	// which happens to be our _bitPattern inverted, and then ANDing with our current iMask
	_iMask = _mm256_andnot_si256(_bitPattern, _iMask);
	// Now we nee to turn the 0.0s into 1.0s. 
	// 1.0 have the bit pattern: 0 011 1111 1111 000... 
	// This can be achieved by ORing iMask with a vector filled with 1.0 doubles (it will leave the -1.0 unchanged)
	_iMask = _mm256_or_si256(*(__m256i*)&_mm256_set1_pd(1.0), _iMask);

	// Finally we return _iMask to as a __m256d by specifying that the 256 bits in _iMask are a, in fact, __m256d
	return *(__m256d*)&_iMask;
}

// Approximate Sin with help of taylor series
// NOTE: Apparently MUCH slower than sequentially using the math.h sin function 4 times.
inline __m256d _mm256d_sin(__m256d angle) {
	//sin 0.4014257
	// =0.4014257 − 0.40142573^3/3! + 0.40142575^5/5! − 0.40142577^7/7! + ...
	// The result is exponentially less accurate the further away from 0 the angle is,
	// so we need to remove as many periods as possible from the angle.
	// The function is fairly accurate between -pi and pi radians, so we will normalize the angle
	// to a value between that range.

	// First we need to transform the angles to a value between [0, 2pi]
	angle = _mm256_modulo_pd(angle, 2*M_PI);
	// Then we remove half of a period to get values between [-pi, pi] (keep in mind this will mirror all values)
	angle = _mm256_sub_pd(angle, _mm256_set1_pd(M_PI));

	// Next we calculate the sin values for the angles:

	// Hardcoding factorial values to save computing time
	__m256d threeFac = _mm256_set1_pd(6);
	__m256d fiveFac = _mm256_set1_pd(120);
	__m256d sevenFac = _mm256_set1_pd(5040);
	__m256d nineFac = _mm256_set1_pd(362880);
	__m256d elevenFac = _mm256_set1_pd(39916800);
	__m256d thirteenFac = _mm256_set1_pd(6227020800);
	__m256d fifteenFac = _mm256_set1_pd(1307674368000);
	__m256d seventeenFac = _mm256_set1_pd(355687428096000);

	__m256d third, fifth, seventh, ninth, eleventh, thirteenth, fifteenth, seventeenth;
	third = _mm256_div_pd(_mm256_pow_pd(angle, 3), threeFac);
	fifth = _mm256_div_pd(_mm256_pow_pd(angle, 5), fiveFac);
	seventh = _mm256_div_pd(_mm256_pow_pd(angle, 7), sevenFac);
	ninth = _mm256_div_pd(_mm256_pow_pd(angle, 9), nineFac);
	eleventh = _mm256_div_pd(_mm256_pow_pd(angle, 11), elevenFac);
	thirteenth = _mm256_div_pd(_mm256_pow_pd(angle, 13), thirteenFac);
	fifteenth = _mm256_div_pd(_mm256_pow_pd(angle, 15), fifteenFac);
	seventeenth = _mm256_div_pd(_mm256_pow_pd(angle, 17), seventeenFac);

	 __m256d result = _mm256_add_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_sub_pd(angle, third), fifth), seventh), ninth), eleventh), thirteenth), fifteenth), seventeenth);
	 //__m256d result = _mm256_sub_pd(_mm256_add_pd(_mm256_sub_pd(_mm256_add_pd(_mm256_sub_pd(angle, third), fifth), seventh), ninth), eleventh);

	// Since we removed half a period, all values are mirrored, so multiply by -1 to correct them
	return _mm256_mul_pd(result, _mm256_set1_pd(-1));
}

// Calculates cos by adding pi/2 to the angle and then running the sin function
// NOTE: Apparently MUCH slower than sequentially using the math.h cos function 4 times.
inline __m256d _mm256d_cos(__m256d angle) {
	angle = _mm256_add_pd(angle, _mm256_set1_pd(M_PI/2));
	return _mm256d_sin(angle);
}

// SIMD version of our haversine function
// Calculate the great circle distance between two points 
// on the earth (specified in decimal degrees)
inline __m256d _mm256d_haversine(double * lat1Start, double * lon1Start, double * lat2Start, double *lon2Start) {
    // convert decimal degrees to radians 
	__m256d _lat1Start, _lat2Start, _lon1Start, _lon2Start;
	_lat1Start = _mm256d_radians(lat1Start);
	_lon1Start = _mm256d_radians(lon1Start);
	_lat2Start = _mm256d_radians(lat2Start);
	_lon2Start = _mm256d_radians(lon2Start);

    // Haversine formula 

	//double dlon = lon2 - lon1;
	//double dlat = lat2 - lat1;
	__m256d _dLon = _mm256_sub_pd(_lon2Start, _lon1Start);
	__m256d _dLat = _mm256_sub_pd(_lat2Start, _lat1Start);
	// double a = pow(sin(dlat / 2), 2) + cos(lat1) * cos(lat2) * pow(sin(dlon / 2), 2);
	__declspec(align(64)) double sin1Arr[4];
	__declspec(align(64)) double sin2Arr[4];
	__declspec(align(64)) double cos1Arr[4];
	__declspec(align(64)) double cos2Arr[4];
	_mm256_store_pd(sin1Arr, _mm256_div_pd(_dLat, _mm256_set1_pd(2.0)));
	_mm256_store_pd(sin2Arr, _mm256_div_pd(_dLon, _mm256_set1_pd(2.0)));
	_mm256_store_pd(cos1Arr, _lat1Start);
	_mm256_store_pd(cos2Arr, _lat2Start);
	for (int i = 0; i < 4; ++i) { sin1Arr[i] = sin(sin1Arr[i]); }
	for (int i = 0; i < 4; ++i) { sin2Arr[i] = sin(sin2Arr[i]); }
	for (int i = 0; i < 4; ++i) { cos1Arr[i] = cos(cos1Arr[i]); }
	for (int i = 0; i < 4; ++i) { cos2Arr[i] = cos(cos2Arr[i]); }

	__m256d _a = _mm256_add_pd(
		_mm256_pow_pd(_mm256_load_pd(sin1Arr), 2),
		_mm256_mul_pd(
			_mm256_mul_pd(
				_mm256_load_pd(cos1Arr),
				_mm256_load_pd(cos2Arr)
			),
			_mm256_pow_pd(_mm256_load_pd(sin2Arr), 2)
		)
	);
	
	// Compute _a with our own sin and cos functions. 
	// It turns out that it was much slower to use our own functions over the math.h trigonometric functions,
	// so we didn't use this approach...
	//__m256d _a = _mm256_add_pd(
	//	_mm256_pow_pd(_mm256d_sin(_mm256_div_pd(_dLat, _mm256_set1_pd(2.0))), 2),
	//	_mm256_mul_pd(
	//		_mm256_mul_pd(
	//			_mm256d_cos(_lat1Start),
	//			_mm256d_cos(_lat2Start)
	//		),
	//		_mm256_pow_pd(_mm256d_sin(_mm256_div_pd(_dLon, _mm256_set1_pd(2.0))), 2)
	//	)
	//);

	// We did not have time to implement arcsin in avx so we have to use the sequential version.
	// But since our trigonometric functions are so slow anyway, it doesn't matter.
	__declspec(align(64)) double temp[4];
	_mm256_store_pd(temp, _mm256_sqrt_pd(_a));
	for (int i = 0; i < 4; ++i) {
		temp[i] = asin(temp[i]);
	}
	//               ^^^^^^^^^^^^
	//double c = 2 * asin(sqrt(a));
	__m256d _c = _mm256_mul_pd(_mm256_load_pd(temp), _mm256_set1_pd(2));
	//double r = 6372.8; // Radius of earth in kilometers. 
	//return c * r;
	__m256d _distances =_mm256_mul_pd(_c, _mm256_set1_pd(6372.8));
	return _distances;
}

// Parallel
double _mm256d_calculateDistance(double * latitudes, double * longitudes, int n) {
	__m256d _totalDistance = _mm256_set1_pd(0.0);
	double totalDistance = 0;
	int shared_i = 0;
	#pragma omp parallel num_threads(4)
	{
		__m256d _localTotalDistance = _mm256_set1_pd(0.0);
		double localTotalDistance = 0;
		int rank = omp_get_thread_num();
		//int threads = omp_get_num_threads();
		//cout << rank << endl;

		#pragma omp for nowait schedule(dynamic)
		for (int i = 0 ; i < n-4; i += 4) {
			//printf("Thread %d handling i=%d\n",rank, i);
			_localTotalDistance = _mm256_add_pd(_localTotalDistance, _mm256d_haversine(latitudes+i, longitudes+i, latitudes+i+1, longitudes+i+1));
			#pragma omp critical 
			{
				shared_i+=4;
			}
		}

		#pragma omp critical
		{
			_totalDistance = _mm256_add_pd(_totalDistance, _localTotalDistance);
			totalDistance += localTotalDistance;
		}
		#pragma omp barrier
		// Let rank 0 compute the sequential rest (if there is any)
		if (rank == 0) {
			for (; shared_i < n-1; ++shared_i) {
				totalDistance += haversine(latitudes[shared_i], longitudes[shared_i], latitudes[shared_i + 1], longitudes[shared_i + 1]);
			}
		}
	}
	
	// Sum the distances into the double variable
	__declspec(align(64)) double aux[4];
	_mm256_store_pd(aux, _totalDistance);
	for(int i = 0; i < 4; ++i){
		totalDistance += aux[i];
	}

	return totalDistance;
}

void printLocations(double**locations, int n) {
	for (int i = 0; i < n; i++) {
		printf("[%f, %f]\n", locations[i][0], locations[i][1]);
	}
}

int main()
{
	// Testing the trigonometric functions
	//double testValues2[4] = { 4.2, -4.0, -13.0, 1.0, };
	//__m256d test2 = _mm256_loadu_pd(testValues2);
	//test2 =_mm256d_cos(test2);
	//cout << *(double*)&test2 << endl;
	//cout << *(((double*)&test2)+1) << endl;
	//cout << *(((double*)&test2)+2) << endl;
	//cout << *(((double*)&test2)+3) << endl;


	double ** locations = createLocations(N);
	double * latitudes = (double*) _mm_malloc(sizeof(double)*N, 64);
	double * longitudes = (double*) _mm_malloc(sizeof(double)*N, 64);
	convertLocationsArray(latitudes, longitudes, locations, N);
	//printLocations(locations, N);

	double tAcc = 0;
	double t;
	double distance;
	for (size_t i = 0; i < 10; ++i) {
		t = omp_get_wtime();
		distance = calculateDistance(locations, N);
		t = omp_get_wtime()  - t;
		tAcc += t;
	}
	// Sequential time
	cout <<"Sequential - Calculations for " << N << " locations" << endl
		<< "Result: " << distance << endl
		<< "Time (avg. from 10 iterations): " << tAcc*1000/10 << "ms" << endl;

	double p_distance;
	tAcc = 0;
	for (size_t i = 0; i < 10; ++i) {
		t = omp_get_wtime();
		p_distance = _mm256d_calculateDistance(latitudes, longitudes, N);
		t = omp_get_wtime() - t;
		tAcc += t;
	}
	cout <<"Parallel - Calculations for " << N << " locations" << endl
		<< "Result: " << p_distance << endl
		<< "Time (avg. from 10 iterations): " << tAcc*1000/10 << "ms" << endl;
}

