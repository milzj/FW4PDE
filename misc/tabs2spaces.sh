#!/bin/bash

for file in $(find ../ -name '*.py' ); do
    echo $file
    expand -i -t 4 $file | sponge $file
done
