# vowels = ['i', 'ü', 'u', 'ö', 'o', 'e', 'a', 'í', 'ű', 'ú', 'ő', 'ó', 'é', 'á']

front_rounded = ['ü', 'ö', 'ő', 'ű']
#front_unrounded = ['i', 'é', 'í', 'e']  # Commented out to omit from sequences
back = ['a', 'o', 'ó', 'u', 'ú', 'á']
vowels = front_rounded + back

def is_grammatical(v1, v2):
    # Both vowels are from the same category
    if (v1 in front_rounded and v2 in front_rounded) or \
       (v1 in back and v2 in back):
        return "grammatical"
    # Vowels are from different categories
    return "ungrammatical"

sequences = []

for v1 in vowels:
    # Skip vowels from front_unrounded category
    #if v1 in front_unrounded:  
    #    continue
    for v2 in vowels:
        # Skip combinations where either vowel is from front_unrounded category
        #if v2 in front_unrounded:
        #    continue
        sequence = f"t {v1} k {v2} z"
        grammaticality = is_grammatical(v1, v2)
        sequences.append(f"{sequence}\t{grammaticality}")

# Writing to blick_test.txt
with open("data/hungarian/blick_test.txt", "w", encoding='utf-8') as file:
    for sequence in sequences:
        file.write(sequence + "\n")
