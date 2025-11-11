import sys

# Define the mapping from structures to their matching genre combos
STRUCTURES = {
    "Freytag's Pyramid": [
        "Tragedy",
        "Horror + Thriller/Suspense",
        "Drama + Tragedy + Romance",
        "Mystery + Crime"
    ],
    "The Hero's Journey": [
        "Fantasy",
        "Adventure + Sci-Fi",
        "Comedy + Fantasy + Adventure",
        "Drama + Romance"
    ],
    "Three Act Structure": [
        "Thriller/Suspense",
        "Comedy + Romance",
        "Sci-Fi + Mystery + Thriller",
        "Crime + Drama"
    ],
    "Dan Harmon's Story Circle": [
        "Comedy",
        "Drama + Comedy + Tragedy",
        "Sci-Fi + Adventure",
        "Romance + Mystery"
    ],
    "Fichtean Curve": [
        "Horror",
        "Thriller/Suspense + Crime",
        "Adventure + Fantasy + Horror",
        "Mystery + Thriller"
    ],
    "Save the Cat Beat Sheet": [
        "Drama + Tragedy",
        "Romance + Comedy",
        "Adventure + Crime",
        "Sci-Fi + Thriller + Mystery"
    ],
    "Seven-Point Story Structure": [
        "Fantasy + Adventure",
        "Horror + Mystery",
        "Crime + Thriller + Drama",
        "Sci-Fi + Tragedy + Romance"
    ]
}

# List of valid main genres for validation
VALID_GENRES = [
    "Romance", "Thriller/Suspense", "Mystery", "Horror", "Comedy",
    "Drama", "Tragedy", "Adventure", "Crime", "Fantasy", "Sci-Fi"
]

def normalize_combo(genres):
    """
    Normalize the list of genres: sort alphabetically and join with ' + '
    """
    sorted_genres = sorted(genres)
    return " + ".join(sorted_genres)

def find_best_structure(input_genres):
    """
    Smartly pick the best structure:
    1. Look for exact combo match.
    2. If no exact, look for structures where the input is a subset of a combo (for arbitrary scenarios).
    3. If multiple, pick the first in order of STRUCTURES keys.
    4. If none, default to 'Three Act Structure' as a versatile fallback.
    """
    if not 1 <= len(input_genres) <= 3:
        raise ValueError("Input must be 1-3 genres.")
    
    # Validate genres
    for genre in input_genres:
        if genre not in VALID_GENRES:
            raise ValueError(f"Invalid genre: {genre}")
    
    combo = normalize_combo(input_genres)
    
    # First pass: exact match
    for struct, combos in STRUCTURES.items():
        if combo in combos:
            return struct
    
    # Second pass: subset match (input genres all in a combo's genres)
    for struct, combos in STRUCTURES.items():
        for c in combos:
            c_genres = set(c.split(" + "))
            input_set = set(input_genres)
            if input_set.issubset(c_genres):
                return struct
    
    # Fallback for arbitrary
    return "Three Act Structure"

if __name__ == "__main__":
    # Expect input as space-separated genres, e.g., python script.py Horror Thriller/Suspense
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python script.py Genre1 [Genre2] [Genre3]")
        sys.exit(1)
    
    input_genres = sys.argv[1:]
    try:
        structure = find_best_structure(input_genres)
        print(structure)  # Just output the name
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)