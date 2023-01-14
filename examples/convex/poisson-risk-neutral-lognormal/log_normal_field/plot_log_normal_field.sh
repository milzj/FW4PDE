#!/bin/bash

rm -rf log_normal_field

python plot_log_normal_field.py

cd log_normal_field

convert -dispose 2 -delay 80 -loop 0 exp_kappa_sample\=*.png log_normal_field.gif

cd ..

