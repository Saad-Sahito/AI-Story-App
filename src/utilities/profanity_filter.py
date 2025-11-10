import re
import unicodedata
from better_profanity import profanity
from config_vars import allowed_words_list, banned_words_list


# --- Load and customize profanity word list ---

# Initialize the default profanity list
profanity.load_censor_words()

# Convert internal words to strings (avoiding VaryingString issue)
current_words = {str(w).lower() for w in profanity.CENSOR_WORDSET}

# Apply custom allow/deny rules
current_words -= {w.lower() for w in allowed_words_list}
current_words |= {w.lower() for w in banned_words_list}

# Reinitialize profanity module with updated set
profanity.load_censor_words(list(current_words))


# --- Leetspeak / Obfuscation handling map ---
char_map = str.maketrans({
    "1": "i", "!": "i", "@": "a", "3": "e",
    "4": "a", "5": "s", "7": "t", "0": "o",
    "$": "s", "€": "e", "£": "l",
})


# --- Normalization helper ---
def _normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")  # remove accents
    text = text.translate(char_map)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    return text


# --- Profanity detection function ---
async def has_profanity(text: str) -> bool:
    normalized = _normalize_text(text)
    return profanity.contains_profanity(normalized)
