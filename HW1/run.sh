learning_rate=(0.001 0.01 0.1)
hidden_size=(100 200)
dropout=(0.3 0.5)
activation_function=(relu tanh)

source ../env/bin/activate

rm -rf results
mkdir results

echo "Running Q1.1a)..."
python hw1-q1.py perceptron

echo "Running Q1.1b)..."
python hw1-q1.py logistic_regression

echo "Running Q1.2b)..."
python hw1-q1.py mlp

echo "_____________________"

for lr in ${learning_rate[@]}; do
    echo "Running logistic regression with learning rate = $lr..."
    python hw1-q2.py logistic_regression -learning_rate $lr
    echo "Done."
    for hs in ${hidden_size[@]}; do
        for dout in ${dropout[@]}; do
            for af in ${activation_function[@]}; do
                echo "Running MLP with:"
                echo "Learning rate = $lr"
                echo "Hidden size = $hs"
                echo "Dropout = $dout"
                echo "Activation function = $af"
                python hw1-q2.py mlp -learning_rate $lr -hidden_size $hs -dropout $dout -activation $af
                echo "Done."
            done
        done
    done
    echo "_____________________"
done

echo "Done."