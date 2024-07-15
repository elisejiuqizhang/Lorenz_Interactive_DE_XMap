#!/bin/bash

# Define the parameter sets
noiseTypes=("gNoise" "lpNoise")
noiseWhens=("in" "post")
noiseAddTypes=("add" "mult" "both")

delays=(1 2 3 4 5)
n_neighbors=(5 10 15)

# Generate the combinations of parameters
combinations=()
for noiseType in "${noiseTypes[@]}"; do
  for noiseWhen in "${noiseWhens[@]}"; do
    for noiseAddType in "${noiseAddTypes[@]}"; do
        for delay in "${delays[@]}"; do
          for n_neighbor in "${n_neighbors[@]}"; do
            combinations+=("$noiseType $noiseWhen $noiseAddType $delay $n_neighbor")
          done
        done
    done
  done
done

# Export the function to parallel
export -f main

# Run the experiments in parallel
parallel -j 32 --colsep ' ' python processLorenzNNDensity.py --noiseType {1} --noiseWhen {2} --noiseAddType {3} --delay {4} --n_neighbors {5} ::: "${combinations[@]}"
