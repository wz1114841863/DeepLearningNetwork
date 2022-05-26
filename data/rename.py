import os


folders = ["./daisy", "./dandelion", "./roses", "./sunflowers", "./tulips"]
count = 0
for folder in folders:
    file_names = os.listdir(folder)
    for file_name in file_names:
        zeros = "0" * (5 - len(str(count)))
        new_name = zeros + str(count) + ".jpg"
        old_path = os.path.join(folder, file_name)
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)
        count += 1