"""
Created on Wed Nov 20

@author: Anh Phan

A script that reads a folder, parses the file names to extract the numbers, finds the smallest and biggest numbers, and identifies any missing numbers. 
It iterates over multiple subdirectories (l1f, l1r, l1c, l1a, l2a, l3b, l3c) and performs the same analysis for each.
Before running, change folder_path accordingly. 

"""


import os

def analyze_files(base_folder, subdirectories):
    for subdir in subdirectories:
        folder_path = os.path.join(base_folder, subdir, "2024", "01") #
        print(f"\nFiles in: {folder_path}")
        
        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        # Collect numbers from filenames
        numbers = []

        for file_name in os.listdir(folder_path):
            if file_name.endswith(".nc"):
                try:
                    # Extract the number from the filename
                    number = int(file_name.split("_")[4])
                    numbers.append(number)
                except (IndexError, ValueError):
                    continue  # Skip files that don't match the expected format

        if not numbers:
            print("No valid files found in this folder.")
            continue

        # Sort the numbers
        numbers.sort()

        # Find the smallest and largest numbers
        smallest = numbers[0]
        largest = numbers[-1]

        # Identify missing numbers
        all_numbers = set(range(smallest, largest + 1))
        missing_numbers = sorted(all_numbers - set(numbers))

        # Display results
        #print("Folder: " + folder_path)
        print(f"Orbits: {smallest} - {largest}")
        print(f"Missing orbits: {missing_numbers}")


# Example usage
base_folder = r"\\smb1.physics.usu.edu\awewrite\soc"
subdirectories = ["l1f", "l1r", "l1c", "l1a", "l2a", "l3a", "l3c"]
analyze_files(base_folder, subdirectories)
