# Noise2Score: Tweedie's Approach to Self-Supervised Image Denoising without Clean Images (NeurIPS 2021)

## Abstract:
In this work, we propose a novel self-supervised image denoising apporach without clean data, called Noise2Score. 
Our novel innovation came from the Tweedie’s formula, which provides explicit representation of denoise images through the score function. By combining with the score-function estimation using AR-DAE, our Noise2Score can be
applied to image denoising problem from any exponential family noises. Furthermore, an identical neural network training can be universally used regardless of the noise models, which leads to the
noise parameter estimation with minimal complexity. The links to SURE and existing Noise2X were also explained, which clearly showed why our method is a better generalization.