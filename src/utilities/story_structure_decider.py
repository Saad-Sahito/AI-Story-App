
FREYTAG_PYRAMID = """
Freytag's Pyramid: Classical Five-Act Structure

ACT I - EXPOSITION (20%): Introduce protagonist, world, and initial conflict. End with inciting incident.

ACT II - RISING ACTION (25%): Build tension with obstacles and escalating challenges.

ACT III - CLIMAX (10%): Peak tension; protagonist faces ultimate test.

ACT IV - FALLING ACTION (25%): Resolve consequences and subplots.

ACT V - DENOUEMENT (20%): Establish new equilibrium and character transformations.

FLEXIBILITY: Adjust act lengths and climax placement for pacing. Add mini-climaxes or blend acts. Scale for 3-5 acts and any story length.
"""

HEROS_JOURNEY = """
The Hero's Journey: 12-Step Circular Arc

ACT I - DEPARTURE (25%):
1. Ordinary World (5%): Hero's normal life and inner needs.
2. Call to Adventure (5%): Disrupting incident.
3. Refusal (5%): Hesitation and fears.
4. Meeting Mentor (5%): Guidance received.
5. Crossing Threshold (5%): Enter special world.

ACT II - INITIATION (50%):
6. Tests, Allies, Enemies (15%): Learn and build alliances.
7. Approach Inmost Cave (10%): Prepare for ordeal.
8. Ordeal (15%): Face greatest fear; symbolic death.
9. Reward (10%): Gain prize but journey continues.

ACT III - RETURN (25%):
10. Road Back (10%): Return with complications.
11. Resurrection (10%): Final test applying lessons.
12. Return with Elixir (5%): Transform world with growth.

FLEXIBILITY: Condense/expand steps; internal or physical journeys. Merge ordeals; multiple mentors/heroes. Adapt to 3-5 acts and varying lengths.
"""

THREE_ACT_STRUCTURE = """
Three Act Structure: Standard Narrative Framework

ACT I - SETUP (25%):
- Opening Image (1%): Hero's initial state.
- Setup (10%): Characters, world, flaws.
- Inciting Incident (12%): Disrupt equilibrium.
- Debate (2%): Weigh options.
- Turning Point (25%): Commit to journey.

ACT II - CONFRONTATION (50%):
- Rising Action A (25-50%): Pursue goal; fun and games.
- Midpoint (50%): Major twist; stakes rise.
- Rising Action B (50-75%): Intensifying complications.
- All Is Lost (75%): Lowest point.
- Turning Point (75%): New resolve.

ACT III - RESOLUTION (25%):
- Climax Build (75-90%): Execute plan.
- Climax (90-95%): Final confrontation.
- Resolution (95-99%): Aftermath and changes.
- Closing Image (99-100%): Transformed state.

FLEXIBILITY: Shift midpoint 5-10%; split Act II. Layer climaxes/subplots. Expand to 4-5 acts or compress for shorter stories.
"""

DAN_HARMON_STORY_CIRCLE = """
Dan Harmon's Story Circle: 8-Step Circular Structure

1. YOU (12.5%): Character in comfort zone, hint at need.
2. NEED (12.5%): Desire something (want vs. true need).
3. GO (12.5%): Enter unfamiliar situation.
4. SEARCH (12.5%): Adapt through trials.
5. FIND (12.5%): Achieve goal (possibly hollow).
6. TAKE (12.5%): Pay heavy price; face consequences.
7. RETURN (12.5%): Head back with lessons.
8. CHANGE (12.5%): Transformed in new equilibrium.

FLEXIBILITY: Vary beat lengths; nest multiple circles. Apply to scenes/subplots/characters. Scale for 3-5 acts and any word count.
"""

FICHTEAN_CURVE = """
Fichtean Curve: Crisis-Driven Rising Tension in 3 Acts

ACT I - INCITING CRISIS (0-25%): Open in action with immediate trouble. Introduce initial crises to hook and build early tension.

ACT II - ESCALATING CRISES (25-75%): Series of escalating crises (3-7 typical), each raising stakes. No tension relief; build through complications and revelations.

ACT III - CLIMAX AND RESOLUTION (75-100%): Ultimate crisis resolves main conflict, followed by brief falling action to establish new normal.

FLEXIBILITY: Adjust crisis count/types; add flashbacks. Layer parallel crises. Works for any length; group crises as needed for pacing.
"""

SAVE_THE_CAT_BEAT_SHEET = """
Save the Cat: 15-Beat Structure in 3 Acts

ACT I - SETUP (0-20%):
1. Opening Image (0-1%): Hero's initial flawed state.
2. Theme Stated (5%): Core thematic question hinted.
3. Setup (1-10%): Introduce world, characters, and what's missing.
4. Catalyst (10%): Life-changing event disrupts status quo.
5. Debate (10-20%): Hero hesitates before committing.
6. Break into Two (20%): Proactive choice; enters new world.

ACT II - CONFRONTATION (20-80%):
7. B Story (22%): Subplot (often relationship) begins.
8. Fun and Games (20-50%): Deliver premise; hero explores new world.
9. Midpoint (50%): Major twist—false victory or defeat; stakes rise.
10. Bad Guys Close In (50-75%): Pressure mounts; complications intensify.
11. All Is Lost (75%): Lowest point; major defeat or loss.
12. Dark Night of the Soul (75-80%): Despair leads to epiphany.

ACT III - RESOLUTION (80-100%):
13. Break into Three (80%): Renewed resolve with new understanding.
14. Finale (80-99%): Gather allies, confront antagonist, demonstrate growth.
15. Closing Image (99-100%): Transformed hero; mirror of opening.

FLEXIBILITY: Shift beats ±5%; merge or condense for pacing. Subplots can follow mini-beats. Adjust lengths for any story size while preserving three-act flow.
"""

SEVEN_POINT_STORY_STRUCTURE = """
Seven-Point Structure: Causality-Focused Arc

NARRATIVE ORDER:
1. Hook (0-15%): Starting weakness/state.
2. Plot Turn 1 (15-25%): Change thrusts into conflict.
3. Pinch 1 (37.5%): Antagonist pressure.
4. Midpoint (50%): Shift to action/knowledge.
5. Pinch 2 (62.5%): Harder strike back.
6. Plot Turn 2 (75-85%): Final piece for victory.
7. Resolution (85-100%): Transformed end state.

FLEXIBILITY: Plan backwards; ±10% shifts. Nest arcs; try-fail cycles. Offset for threads; adapt to 3-5 acts by grouping points.
"""
# Helper mapping and accessor for convenience in code
STRUCTURE_VARIABLES = {
    "Freytag's Pyramid": FREYTAG_PYRAMID,
    "The Hero's Journey": HEROS_JOURNEY,
    "Three Act Structure": THREE_ACT_STRUCTURE,
    "Dan Harmon's Story Circle": DAN_HARMON_STORY_CIRCLE,
    "Fichtean Curve": FICHTEAN_CURVE,
    "Save the Cat Beat Sheet": SAVE_THE_CAT_BEAT_SHEET,
    "Seven-Point Story Structure": SEVEN_POINT_STORY_STRUCTURE,
}
# List of valid main genres for validation
VALID_GENRES = [
    "Romance", "Thriller/Suspense", "Mystery", "Horror", "Comedy",
    "Drama", "Tragedy", "Adventure", "Crime", "Fantasy", "Sci-Fi"
]

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


def get_structure(name: str) -> str:
    """Return the prompt string for a structure name (case-insensitive key match).

    Example: get_structure("Freytag")
    """
    for k, v in STRUCTURE_VARIABLES.items():
        if k == name:
            return v
    raise KeyError(f"Unknown structure: {name}")


def normalize_combo(genres):
    """
    Normalize the list of genres: sort alphabetically and join with ' + '
    """
    sorted_genres = sorted(genres)
    return " + ".join(sorted_genres)


def find_best_structure_name(input_genres: list[str]) -> str:
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
        if genre == "Fiction":
            return "Three Act Structure"
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

def find_best_structure(structure_name: str) -> str:
    
    return get_structure(structure_name)


# if __name__ == "__main__":
#     # Expect input as space-separated genres, e.g., python script.py Horror Thriller/Suspense
#     if len(sys.argv) < 2 or len(sys.argv) > 4:
#         print("Usage: python script.py Genre1 [Genre2] [Genre3]")
#         sys.exit(1)
    
#     input_genres = sys.argv[1:]
#     try:
#         structure = find_best_structure(input_genres)
#         print(structure)  # Just output the name
#     except ValueError as e:
#         print(f"Error: {e}")
#         sys.exit(1)