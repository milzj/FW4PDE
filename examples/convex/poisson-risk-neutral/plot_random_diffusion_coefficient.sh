#!/bin/bash

rm -rf random_diffusion_coefficient

python plot_random_diffusion_coefficient.py

cd random_diffusion_coefficient

convert -dispose 2 -delay 80 -loop 0 exp_kappa_sample\=*.png random_diffusion_coefficient.gif

cd ..

