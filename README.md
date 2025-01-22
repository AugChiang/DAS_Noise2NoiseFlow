# camera_noise_modeling
## Fork from Noise2NoiseFlow

Trained with DAS paired data from scratch.
- Data sampling rate = 50 Hz.
- Data Amount ~ 1,500
- Data Shape ~ (30000,2000) along time frames and channels, respectively.

## Denoiseing Results
Use a very small DnCNN model to perform the denoising operation. And the results are not bad as follow.
![image](my_results/plots/test_denoised.png)

## Problem
The sample function call of the NoiseFlow **NOT** work correctly, that generated noises are not correct.

However, the prediction of NoiseFlow does seems to generate DAS noises.

![image](my_results/plots/test_noiseflow.png)
