#!/bin/bash

DATA="${1-MUTAG}"
fold=${2-1}
test_number=${3-0}

gm=GRD
gpu_or_cpu=gpu
GPU=0
CONV_SIZE="32-32-32-1"
sortpooling_k=.6
FP_LEN=0
n_hidden=128
bsize=1
dropout=True

case ${DATA} in
DFDC)
  num_epochs=100
  learning_rate=0.0001
  ;;
DFDC_sample)
  num_epochs=5
  learning_rate=.0001
  ;;
FF)
  num_epochs=30
  learning_rate=0.0001
  ;;
deepfake)
  num_epochs=100
  learning_rate=.0001
  ;;
DF40)
  num_epochs=100
  learning_rate=.00001
  ;;
deepfake3)
  num_epochs=100
  learning_rate=.000001
  ;;
MUTAG)
  num_epochs=30
  learning_rate=0.0001
  ;;
*)
  num_epochs=50
  learning_rate=0.00001
  ;;
esac

if [ ${fold} == 0 ]; then
  echo "Running 10-fold cross validation"
  start=`date +%s`
  for i in $(seq 1 10)
  do
    CUDA_VISIBLE_DEVICES=${GPU} python main.py \
        -seed 1 \
        -data $DATA \
        -fold $i \
        -learning_rate $learning_rate \
        -num_epochs $num_epochs \
        -hidden $n_hidden \
        -latent_dim $CONV_SIZE \
        -sortpooling_k $sortpooling_k \
        -out_dim $FP_LEN \
        -batch_size $bsize \
        -gm $gm \
        -mode $gpu_or_cpu \
        -dropout $dropout
  done
  stop=`date +%s`
  echo "End of cross-validation"
  echo "The total running time is $[stop - start] seconds."
  echo "The accuracy results for ${DATA} are as follows:"
  cat acc_results.txt
  echo "Average accuracy is"
  tail -10 acc_results.txt | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }'
else
  CUDA_VISIBLE_DEVICES=${GPU} python main.py \
      -seed 1 \
      -data $DATA \
      -fold $fold \
      -learning_rate $learning_rate \
      -num_epochs $num_epochs \
      -hidden $n_hidden \
      -latent_dim $CONV_SIZE \
      -sortpooling_k $sortpooling_k \
      -out_dim $FP_LEN \
      -batch_size $bsize \
      -gm $gm \
      -mode $gpu_or_cpu \
      -dropout $dropout \
      -test_number ${test_number}
fi

