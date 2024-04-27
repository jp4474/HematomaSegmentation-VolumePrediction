# Define the directories
directories=("data/img_train" "data/mask_train" "data/img_val" "data/mask_val")

# Check each directory
for dir in "${directories[@]}"
do
  # If the directory doesn't exist
  if [ ! -d "$dir" ]
  then
    # Create the directory
    mkdir -p "$dir"
  fi
done

while IFS= read -r file
do
  mv "data/img_train_val/$file" "data/img_train/$file"
done < data/img_train_pathes.txt

while IFS= read -r file
do
  mv "data/mask_train_val/$file" "data/mask_train/$file"
done < data/mask_train_pathes.txt

while IFS= read -r file
do
  mv "data/img_train_val/$file" "data/img_val/$file"
done < data/img_val_pathes.txt

while IFS= read -r file
do
  mv "data/mask_train_val/$file" "data/mask_val/$file"
done < data/mask_val_pathes.txt