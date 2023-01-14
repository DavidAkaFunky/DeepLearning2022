source ../../env/bin/activate

echo "Running Q2 without attention..."
python hw2-q3.py > result.txt
echo "_____________________"
echo "Running Q2 with attention..."
python hw2-q3.py --use_attn > result_attn.txt
echo "Done."