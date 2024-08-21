for dataset in "german"
do
    for source in "smote" "stasy" "tabddpm" "tabsyn" "fairtabddpm" "real" "fairsmote" "fairtabgan" "goggle" "great" "codi"
    # for source in "tabsyn"
    do
        for option in "best" "mean"
        do
            for dist in "original" "uniform"
            do
                python learning.py --dataset $dataset --source $source --option $option --dist $dist
            done
        done
    done
done
