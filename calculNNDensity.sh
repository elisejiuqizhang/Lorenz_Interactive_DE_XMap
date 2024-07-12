#!/bin/bash

# Define the parameter sets
noiseTypes=("gNoise" "lpNoise")
noiseWhens=("in" "post")
noiseAddTypes=("add" "mult" "both")
noiseLevels=(0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75)
delays=(1 2 3 4 5)
n_neighbors=(5 10 15)

# Generate the combinations of parameters
combinations=()
for noiseType in "${noiseTypes[@]}"; do
  for noiseWhen in "${noiseWhens[@]}"; do
    for noiseAddType in "${noiseAddTypes[@]}"; do
      for noiseLevel in "${noiseLevels[@]}"; do
        for delay in "${delays[@]}"; do
          for n_neighbor in "${n_neighbors[@]}"; do
            combinations+=("$noiseType $noiseWhen $noiseAddType $noiseLevel $delay $n_neighbor")
          done
        done
      done
    done
  done
done

# Export the function to parallel
export -f main

# Run the experiments in parallel
parallel -j 8 --colsep ' ' python calculNNDensity.py --noiseType {1} --noiseWhen {2} --noiseAddType {3} --noiseLevel {4} --delay {5} --n_neighbors {6} ::: "${combinations[@]}"
