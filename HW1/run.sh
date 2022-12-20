learning_rate=(0.001 0.01 0.1)
hidden_size=200
dropout=0.5
activation_function=tanh

source ../env/bin/activate

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
    echo "_____________________"
    echo "Running MLP with:"
    echo "Learning rate = $lr"
    python hw1-q2.py mlp -learning_rate $lr
    echo "_____________________"
done
echo "Running MLP with:"
echo "Hidden size = $hidden_size"
python hw1-q2.py mlp -hidden_size $hidden_size
echo "_____________________"
echo "Running MLP with:"
echo "Dropout = $dropout"
python hw1-q2.py mlp -dropout $dropout
echo "_____________________"
echo "Running MLP with:"
echo "Activation function = $activation_function"
python hw1-q2.py mlp -activation $activation_function
echo "_____________________"

echo "Done."