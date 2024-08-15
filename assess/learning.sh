for dataset in "german"
do
    for source in "real" "synt"
    do
        for option in "best" "mean"
        do
            for rand in "original" "uniform"
            do
                python learning.py --dataset $dataset --source $source --option $option --rand $rand
            done
        done
    done
done
