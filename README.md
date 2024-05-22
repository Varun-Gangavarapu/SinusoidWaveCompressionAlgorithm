This is just a project I made to test out some ideas i had for a new Image compression algorithm.
It works on greyscaled images where the image is represented as a bunch of pixels with x and y values and a brightness values.
It then represents each of those parameters as frequency, amplitude, and phase shift in a sin wave
We then combine all the sin waves corrosponding to each pixel into one big wave.
That wave is the compressed format for the image
We then run a Fourier Transformation on the compressed wave to decomress the image
This will give us a bunch of sin waves which we will use to reconstruct the pixels
