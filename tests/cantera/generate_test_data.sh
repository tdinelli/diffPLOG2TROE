#!/bin/zsh

for file in *.py
do
    if [ -f "$file" ]; then
        echo "Running $file..."
        python "$file"
        echo "Finished running $file"
        echo "-----------------------"
    fi
done

echo "All Python files have been executed and the test data generated."
