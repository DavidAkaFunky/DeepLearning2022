learning_rate=(0.001 0.01 0.1)
hidden_size=(100 200)
dropout=(0.3 0.5)
activation_function=(relu tanh)

source ../env/bin/activate

echo "Running Q1.1b..."
python hw1-q1.py logistic_regression
echo "_____________________"

rm -rf results
mkdir results
for lr in ${learning_rate[@]}; do
    echo "Running logistic regression with learning rate = $lr..."
    python hw1-q2.py logistic_regression -learning_rate $lr
    echo "Done."
    for hs in ${hidden_size[@]}; do
        for dout in ${dropout[@]}; do
            for af in ${activation_function[@]}; do
                echo "Running logistic regression with: Learning rate = $lr"
                echo "Hidden size = $hs"
                echo "Dropout = $ds"
                echo "Activation function = $af"
                python hw1-q2.py mlp -learning_rate $lr -hidden_size $hs -dropout $dout -activation $af
                echo "Done."
            done
        done
    done
    echo "_____________________"
done

echo "Done."