learning_rate=(0.001 0.01 0.1)
hidden_size=200
dropout=0.5
activation_function=tanh
layers=(2 3)

source ../env/bin/activate

echo "Running Q1.1a)..."
python hw1-q1.py perceptron
echo "_____________________"

echo "Running Q1.1b)..."
python hw1-q1.py logistic_regression
echo "_____________________"

echo "Running Q1.2b)..."
python hw1-q1.py mlp
echo "_____________________"

for lr in ${learning_rate[@]}; do
    echo "Running logistic regression with learning rate = $lr..."
    python hw1-q2.py logistic_regression -learning_rate $lr
    echo "_____________________"

    echo "Running MLP with learning rate = $lr..."
    python hw1-q2.py mlp -learning_rate $lr
    echo "_____________________"
done

echo "Running MLP with: hidden size = $hidden_size..."
python hw1-q2.py mlp -hidden_size $hidden_size
echo "_____________________"

echo "Running MLP with: dropout = $dropout..."
python hw1-q2.py mlp -dropout $dropout
echo "_____________________"

echo "Running MLP with activation function = $activation_function..."
python hw1-q2.py mlp -activation $activation_function
echo "_____________________"

for l in ${layers[@]}; do
    echo "Running MLP with $l layers..."
    python hw1-q2.py mlp -layers $l
    echo "_____________________"
done

echo "Done."