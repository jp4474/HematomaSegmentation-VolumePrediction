#!/bin/bash

# Define the directories
folder1="mask_test"
folder2="img_train_val_test"
folder3="img_test"
if [ ! -d "$folder3" ]; then
    mkdir "$folder3"
fi

# Iterate over all files in folder1
for file1 in "$folder1"/*; do
    filename=$(basename -- "$file1")
    IFS='.' read -ra ADDR <<< "$filename"
    first_element=${ADDR[0]}
    for file in "$folder2"/*; do
        if [[ $(basename "$file") == "$first_element"* ]]; then
            echo "File $file starts with $first_element"
            mv "$file" "$folder3"
        fi
    done
done
