bash_file_name=$(basename $0)

for dataset in "imagenet"
do
      for seed in 0
      do
            for severity in 5
            do
                  for tta_method in "HeML" 
                  do
                        for arch in "resnet18"
                        do
                              for ALPHA_REGMEAN in 1.0
                              do
                                    for SCALING_COEFF in 1.5
                                    do
                                    python CS.py \
                                          -acfg configs/adapter/${dataset}/${tta_method}.yaml \
                                          -dcfg configs/dataset/${dataset}.yaml \
                                          -ocfg configs/order/${dataset}/0.yaml \
                                          SEED $seed \
                                          TEST.BATCH_SIZE 64 \
                                          CORRUPTION.SEVERITY "[${severity}]" \
                                          NOTE "test" \
                                          bash_file_name $bash_file_name
                                    done
                              done
                        done
                  done
            done
      done
done