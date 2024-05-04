# Define the directories
directories=("img_train" "mask_train" "img_val" "mask_val")

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
  mv "img_train_val/$file" "img_train/$file"
done < img_train_pathes.txt

while IFS= read -r file
do
  mv "mask_train_val/$file" "mask_train/$file"
done < mask_train_pathes.txt

while IFS= read -r file
do
  mv "img_train_val/$file" "img_val/$file"
done < img_val_pathes.txt

while IFS= read -r file
do
  mv "mask_train_val/$file" "mask_val/$file"
done < mask_val_pathes.txt