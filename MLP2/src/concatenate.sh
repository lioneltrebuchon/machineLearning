for i in `seq 0 13`
    do cat ../features/cube/test_cube_features_from$((i*10+1))\to$((i*10+10))\.csv
done > ../features/cube_test_features.csv
for i in `seq 0 27`
    do cat ../features/cube/train_cube_features_from$((i*10+1))\to$((i*10+10))\.csv
done > ../features/cube_train_features.csv
