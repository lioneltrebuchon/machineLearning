#!/bin/bash
for i in `seq 0 27`
    do bsub -W 08:00 -r -o output_train_cube_features_$((i*10+1)\to$((i*10+10)).txt -R "rusage[mem=8000]" python compute_cube_features.py train $((i*10+1) $((i*10+10)
done
for i in `seq 0 13`
    do bsub -W 08:00 -r -o output_test_cube_features_$((i*10+1)\to$((i*10+10)).txt -R "rusage[mem=8000]" python compute_cube_features.py test $((i*10+1) $((i*10+10)
done
