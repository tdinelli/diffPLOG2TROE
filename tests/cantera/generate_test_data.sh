#!/bin/zsh

for file in *.py
do
    if [ -f "$file" ]; then
        echo "-----------------------"
        echo " ****** Running $file..."
        python "$file"
        echo "Finished running $file"
    fi
done
echo "\n-----------------------"
echo "All Python files have been executed and the test data generated."
