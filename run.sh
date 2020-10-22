#!/bin/bash

JOBS=40
JOBS_FC=8


# static
for model in LR RF LASSO SVC GB; do
  for data in FC REHO ALFF fALFF; do
    python ~/schizo_fmri/static_approach.py --jobs $JOBS --model $model --include $data -- /mnt/users/bwojcik/local/schizo_fmri/STATIC /mnt/users/bwojcik/local/schizo_fmri/Demographics_questionnaires.csv /mnt/users/bwojcik/local/schizo_fmri/static_results
    python ~/schizo_fmri/static_approach.py --jobs $JOBS --model $model --pca --include $data -- /mnt/users/bwojcik/local/schizo_fmri/STATIC /mnt/users/bwojcik/local/schizo_fmri/Demographics_questionnaires.csv /mnt/users/bwojcik/local/schizo_fmri/static_results
  done
  python ~/schizo_fmri/static_approach.py --jobs $JOBS --model $model --include FC REHO ALFF fALFF -- /mnt/users/bwojcik/local/schizo_fmri/STATIC /mnt/users/bwojcik/local/schizo_fmri/Demographics_questionnaires.csv /mnt/users/bwojcik/local/schizo_fmri/static_results
  python ~/schizo_fmri/static_approach.py --jobs $JOBS --model $model --pca --include FC REHO ALFF fALFF -- /mnt/users/bwojcik/local/schizo_fmri/STATIC /mnt/users/bwojcik/local/schizo_fmri/Demographics_questionnaires.csv /mnt/users/bwojcik/local/schizo_fmri/static_results
done

# static 116
for model in LR RF LASSO SVC GB; do
  for data in FC REHO ALFF fALFF; do
    python ~/schizo_fmri/static_approach.py --jobs $JOBS --model $model --include $data -- /mnt/users/bwojcik/local/schizo_fmri/STATIC_116 /mnt/users/bwojcik/local/schizo_fmri/Demographics_questionnaires.csv /mnt/users/bwojcik/local/schizo_fmri/static_116_results
    python ~/schizo_fmri/static_approach.py --jobs $JOBS --model $model --pca --include $data -- /mnt/users/bwojcik/local/schizo_fmri/STATIC_116 /mnt/users/bwojcik/local/schizo_fmri/Demographics_questionnaires.csv /mnt/users/bwojcik/local/schizo_fmri/static_116_results
  done
  python ~/schizo_fmri/static_approach.py --jobs $JOBS --model $model --include FC REHO ALFF fALFF -- /mnt/users/bwojcik/local/schizo_fmri/STATIC_116 /mnt/users/bwojcik/local/schizo_fmri/Demographics_questionnaires.csv /mnt/users/bwojcik/local/schizo_fmri/static_116_results
  python ~/schizo_fmri/static_approach.py --jobs $JOBS --model $model --pca --include FC REHO ALFF fALFF -- /mnt/users/bwojcik/local/schizo_fmri/STATIC_116 /mnt/users/bwojcik/local/schizo_fmri/Demographics_questionnaires.csv /mnt/users/bwojcik/local/schizo_fmri/static_116_results
done


# dynamic
# for model in LR RF LASSO SVC GB; do
#  python ~/schizo_fmri/dynamic_approach.py --jobs $JOBS --model $model --include REHO ALFF fALFF -- /mnt/users/bwojcik/local/schizo_fmri/DYNAMIC /mnt/users/bwojcik/local/schizo_fmri/Demographics_questionnaires.csv /mnt/users/bwojcik/local/schizo_fmri/dynamic_results
#  python ~/schizo_fmri/dynamic_approach.py --jobs $JOBS --model $model --pca --include REHO ALFF fALFF -- /mnt/users/bwojcik/local/schizo_fmri/DYNAMIC /mnt/users/bwojcik/local/schizo_fmri/Demographics_questionnaires.csv /mnt/users/bwojcik/local/schizo_fmri/dynamic_results
#  for data in REHO ALFF fALFF; do
#    python ~/schizo_fmri/dynamic_approach.py --jobs $JOBS --model $model --include $data -- /mnt/users/bwojcik/local/schizo_fmri/DYNAMIC /mnt/users/bwojcik/local/schizo_fmri/Demographics_questionnaires.csv /mnt/users/bwojcik/local/schizo_fmri/dynamic_results
#    python ~/schizo_fmri/dynamic_approach.py --jobs $JOBS --model $model --pca --include $data -- /mnt/users/bwojcik/local/schizo_fmri/DYNAMIC /mnt/users/bwojcik/local/schizo_fmri/Demographics_questionnaires.csv /mnt/users/bwojcik/local/schizo_fmri/dynamic_results
#  done
# done

# dynamic FC with PCA
# for model in LR RF; do
#  python ~/schizo_fmri/dynamic_approach.py --jobs $JOBS_FC --model $model --pca --include FC -- /mnt/users/bwojcik/local/schizo_fmri/DYNAMIC /mnt/users/bwojcik/local/schizo_fmri/Demographics_questionnaires.csv /mnt/users/bwojcik/local/schizo_fmri/dynamic_results
#  python ~/schizo_fmri/dynamic_approach.py --jobs $JOBS_FC --model $model --pca --include FC REHO ALFF fALFF -- /mnt/users/bwojcik/local/schizo_fmri/DYNAMIC /mnt/users/bwojcik/local/schizo_fmri/Demographics_questionnaires.csv /mnt/users/bwojcik/local/schizo_fmri/dynamic_results
# done

# dynamic 116
# for model in LR RF LASSO SVC GB; do
#  python ~/schizo_fmri/dynamic_approach.py --jobs $JOBS --model $model --include REHO ALFF fALFF -- /mnt/users/bwojcik/local/schizo_fmri/DYNAMIC_116 /mnt/users/bwojcik/local/schizo_fmri/Demographics_questionnaires.csv /mnt/users/bwojcik/local/schizo_fmri/dynamic_116_results
#  python ~/schizo_fmri/dynamic_approach.py --jobs $JOBS --model $model --pca --include REHO ALFF fALFF -- /mnt/users/bwojcik/local/schizo_fmri/DYNAMIC_116 /mnt/users/bwojcik/local/schizo_fmri/Demographics_questionnaires.csv /mnt/users/bwojcik/local/schizo_fmri/dynamic_116_results
#  for data in REHO ALFF fALFF; do
#    python ~/schizo_fmri/dynamic_approach.py --jobs $JOBS --model $model --include $data -- /mnt/users/bwojcik/local/schizo_fmri/DYNAMIC_116 /mnt/users/bwojcik/local/schizo_fmri/Demographics_questionnaires.csv /mnt/users/bwojcik/local/schizo_fmri/dynamic_116_results
#    python ~/schizo_fmri/dynamic_approach.py --jobs $JOBS --model $model --pca --include $data -- /mnt/users/bwojcik/local/schizo_fmri/DYNAMIC_116 /mnt/users/bwojcik/local/schizo_fmri/Demographics_questionnaires.csv /mnt/users/bwojcik/local/schizo_fmri/dynamic_116_results
#  done
# done

# dynamic 116 FC with PCA
# for model in LR RF; do
#  python ~/schizo_fmri/dynamic_approach.py --jobs $JOBS_FC --model $model --pca --include FC -- /mnt/users/bwojcik/local/schizo_fmri/DYNAMIC_116 /mnt/users/bwojcik/local/schizo_fmri/Demographics_questionnaires.csv /mnt/users/bwojcik/local/schizo_fmri/dynamic_116_results
#  python ~/schizo_fmri/dynamic_approach.py --jobs $JOBS_FC --model $model --pca --include FC REHO ALFF fALFF -- /mnt/users/bwojcik/local/schizo_fmri/DYNAMIC_116 /mnt/users/bwojcik/local/schizo_fmri/Demographics_questionnaires.csv /mnt/users/bwojcik/local/schizo_fmri/dynamic_116_results
# done
