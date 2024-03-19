#Train the network for Gaussian noise model
#python train.py --model Gaussian_blind --parameter 25  --dataroot /your_path/ --name BSD_ours_unet_25 --gpu_ids '0' --direction BtoA 

#Test the Noise2SCore for Gaussian noise model
#python test.py --model Gaussian_blind --parameter 25 --dataroot /your_path/ --name BSD_ours_unet_25 --direction BtoA  --gpu_ids '0' --epoch best --results_dir /your_results/

#Train the network for Poisson noise model
#python train.py --model Poisson_blind --parameter 0.01  --dataroot /your_path/ --name BSD_ours_unet_0.01 --gpu_ids '0' --direction BtoA 

#Test the Noise2SCore for Poisson noise model
#python test.py --model Poisson_blind --parameter 0.01 --dataroot /your_path/ --name BSD_ours_unet_0.01 --direction BtoA  --gpu_ids '0' --epoch best --results_dir /your_results/

#Train the network for Gamma noise model
#python train.py --model Gamma_blind --parameter 100  --dataroot /your_path/ --name CBSD_ours_unet_100 --gpu_ids '0' --direction BtoA 

#Test the Noise2SCore for Gamma noise model
#python test.py --model Gamma_blind --parameter 100 --dataroot /your_path/ --name CBSD_ours_unet_100 --direction BtoA  --gpu_ids '0' --epoch best --results_dir /your_results/

#python test_blind.py --model Gaussian_blind --parameter 25 --dataroot ./testdata/Set12 --name BSD_ours_unet_25_blind --direction BtoA  --gpu_ids '0' --epoch best --results_dir ./results/

#python test_blind.py --model Poisson_blind  --scale_param 0.01 --dataroot ./testdata/Set12 --name BSD_ours_unet_0.01_blind --direction BtoA  --gpu_ids '0' --epoch best --results_dir ./results/

#python test_blind.py --model Gamma_blind --parameter 100 --dataroot ./testdata/Set12 --name BSD_ours_unet_gamma_100_blind --direction BtoA  --gpu_ids '0' --epoch best --results_dir ./results/
