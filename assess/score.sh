for dataset in "compass" "german" "adult" "bank"
do
    python quality.py --dataset $dataset --density --dcr
done