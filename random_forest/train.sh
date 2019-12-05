#!/bin/bash
for VAR in 1 2 3 4 5
do
	python3 random-forest.py mnist_gini_100_50_$VAR >> mnist_gini_100_50.txt
done
