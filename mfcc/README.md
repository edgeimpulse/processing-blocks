# Notes on parameters

Summary:  Number of coefficients, filter number (really should say “Number of mel filters”) and Window size are worth playing with

Number of coefficients
6 ( “Six coefficients succeed in capturing most of the relevant information.  The importance of the higher cepstrum coefficients appears to depend on the speaker” (1) )
13, I see this default a lot ( also, “A set of 10 mel-frequency cepstrum coefficients computed…” (1) )
Cannot be larger than Filter number!

Frame length
0.024 (24 mS)
Related to the max length of a syllable in speech
Not much wiggle room here.  (3) mentions that if segments are too long, the sound will change too much from start to finish.  If segments are too short, there will not be enough of the signal to get useful information.
Was used in (1).
Other sources mention 10-20 mS as ideal
Set it and forget it

Frame stride
0.006 (6 mS)  (6.4 mS was used in (1) )
Let’s set it and forget it.

Filter number
20,
26,
40
(see (4))

FFT size
Let frame_size = Frame_length * sample_rate
Choose FFT size to be the next power of 2 >frame_size

Window size ( need to rename “Normalization window size ( # of cepstral frames ) )
300
100

Low frequency:
0, set it and forget it.  (3) mentions that fricative and plosive sounds occupy the low end of the cepstrum, so you would not really want to change this

High frequency:
Should just set to ½ the sample rate (Nyquist limit), or 8KHz, whichever is lower

Sources:
Davis and Mermelstein, 1980
Viikki Laurila, 1998
Oppenheim and Schafer, Discrete Time Signal Processing
http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
