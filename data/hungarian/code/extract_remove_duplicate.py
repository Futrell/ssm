import csv

def remove_duplicates_and_filter(child_file_path, adult_file_path, child_output_path, adult_output_path):
    child_data = []
    adult_data = []

    # Read the adult data and remove duplicates
    adult_sf = set()
    with open(adult_file_path, 'r', encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            if row['SF'] not in adult_sf:
                adult_sf.add(row['SF'])
                adult_data.append(row)

    # Read the child data
    with open(child_file_path, 'r', encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            if row['SF'] not in adult_sf:  # Check against adult SF
                child_data.append(row)

    # Remove duplicates from child data based on SF and keep the row with the most segments in Segmentation
    child_data.sort(key=lambda x: (-len(x['Segmentation'].split('-')), int(x['Freq'])), reverse=True)
    seen_sf = set()
    cleaned_child_data = []
    for row in child_data:
        if row['SF'] not in seen_sf:
            seen_sf.add(row['SF'])
            cleaned_child_data.append(row)

    # Sort the cleaned child data based on Freq
    cleaned_child_data.sort(key=lambda x: int(x['Freq']), reverse=True)

    # Save the cleaned child data
    with open(child_output_path, 'w', encoding="utf-8", newline='') as file:
        writer = csv.DictWriter(file, fieldnames=cleaned_child_data[0].keys(), delimiter="\t")
        writer.writeheader()
        for row in cleaned_child_data:
            writer.writerow(row)

    # Save the cleaned adult data
    with open(adult_output_path, 'w', encoding="utf-8", newline='') as file:
        writer = csv.DictWriter(file, fieldnames=adult_data[0].keys(), delimiter="\t")
        writer.writeheader()
        for row in adult_data:
            writer.writerow(row)

# Call the function
remove_duplicates_and_filter("data/hungarian/CHILDES_w_unimorph_CHI.txt", 
                             "data/hungarian/CHILDES_w_unimorph_ADULT.txt",
                             "data/hungarian/CHILDES_w_unimorph_CHI_cleaned.txt",
                             "data/hungarian/CHILDES_w_unimorph_ADULT_cleaned.txt")
