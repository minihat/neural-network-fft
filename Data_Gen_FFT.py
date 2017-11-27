import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import csv

sess = tf.Session()

sample_length = 202 # Written sample length = sample_length - 2
number_of_samples = 250
show_plots = 0 # Change to 1 to see plots of the samples and their FFTs
save_data = 1 # Change to 0 if you want to see plots without writing any data

data_csv = []
fft_data_csv = []
fft_data_csv_imag = []

# Each iteration creates a noisy signal, takes the FFT, and adds this to lists
for sample in range(number_of_samples):
    # Generate a random list
    upper_bound = random.randint(1,10000)
    lower_bound = random.randint(-10000,-1)

    mysample = np.complex64([])
    for i in range(sample_length):
        mysample = np.append(mysample, np.complex64(random.uniform(lower_bound,upper_bound)))

    #print(mysample)
    # Take the Fast Fourier Transform of the list
    fft_input = tf.constant(mysample)
    fft_output = tf.fft(fft_input,name='FFT')
    fftd_signal = np.complex64(sess.run(fft_output))

    # Create some plots to see what the FFT is doing
    plotsample = mysample[1:-1].real
    plotfftreal = fftd_signal[1:-1].real
    plotfftimag = fftd_signal[1:-1].imag

    if show_plots == 1:
        plt.figure(1)
        plt.subplot(311)
        plt.plot(plotsample)
        plt.title('Original Sample')

        plt.subplot(312)
        plt.plot(plotfftreal,'r')
        plt.title('Real FFT')

        plt.subplot(313)
        plt.plot(plotfftimag,'g')
        plt.title('Imaginary FFT')
        plt.show()

    data_csv.append(plotsample)
    fft_data_csv.append(plotfftreal)
    fft_data_csv_imag.append(plotfftimag)


# Print all of the generated samples to files
if save_data == 1:
    outfile = open('noisydata_test.csv','w')
    csvwrite = csv.writer(outfile,delimiter=',')
    for row in data_csv:
        csvwrite.writerow(row)
    outfile.close()

    outfile = open('noisydata_fft_real_test.csv','w')
    csvwrite = csv.writer(outfile,delimiter=',')
    for row in fft_data_csv:
        csvwrite.writerow(row)
    outfile.close()

    outfile = open('noisydata_fft_imag_test.csv','w')
    csvwrite = csv.writer(outfile,delimiter=',')
    for row in fft_data_csv_imag:
        csvwrite.writerow(row)
    outfile.close()
