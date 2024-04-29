/*=========================================================
FFT for real input arrays
Author: Ehsan Ghasemi
phone: +98-9904690571
Email: Ethenghasemi@gmail.com
===========================================================*/

/*
#include <iostream>
#include <cmath>
#include <complex>
#include <fftw3.h>

const int N = 256; // Number of points
const double PI = 3.14159265358979323846;

int main() {
    double in[N]; // Declare input array for real values
    fftw_complex out[N]; // Declare output array for complex values
    fftw_plan p = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE); // Create FFTW plan

    // Fill input array with a sine wave
    for(int i = 0; i < N; i++) {
        in[i] = sin(2 * PI * i / N);
    }

    fftw_execute(p); // Execute FFT

    // Print FFT results
    for(int i = 0; i < N; i++) {
        std::cout << i << ": " << "(" << out[i][0] << ", " << out[i][1] << ")\n";
    }

    fftw_destroy_plan(p); // Clean up
    return 0;
}
*/



/*=========================================================
FFT for complex input arrays
Author: Ehsan Ghasemi
phone: +98-9904690571
Email: Ethenghasemi@gmail.com
===========================================================*/


#include <iostream>
#include <cmath>
#include <complex>
#include <fftw3.h>

const int N = 256; // Number of points
const double PI = 3.14159265358979323846;

int main() {
    fftw_complex in[N], out[N]; // Declare input and output arrays
    fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE); // Create FFTW plan

    // Fill input array with a sine and cosine wave
    for(int i = 0; i < N; i++) {
        in[i][0] = sin(2 * PI * i / N);
        //in[i][1] = cos(2 * PI * i / N);
		in[i][1] = 0.0; 
    }

    fftw_execute(p); // Execute FFT

    // Print FFT results
    for(int i = 0; i < N; i++) {
        std::cout << i << ": " << "(" << out[i][0] << ", " << out[i][1] << ")\n";
    }

    fftw_destroy_plan(p); // Clean up
    return 0;
}

