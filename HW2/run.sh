learning_rate=(0.00001 0.0005 0.01)

source ../env/bin/activate

for lr in ${learning_rate[@]}; do
    echo "Running Q2 with learning rate = $lr..."
    python hw2-q2/hw2-q2.py -learning_rate $lr
    echo "_____________________"
done

echo "Done."