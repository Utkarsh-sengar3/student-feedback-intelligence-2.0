import re, emoji

def add_space_around_emoji(text):
    text = text.strip()  # Remove leading and trailing spaces

    # Get the list of emojis in the text
    emojis_found = emoji.emoji_list(text)

    # If no emojis are found, return the original text
    if not emojis_found:
        return text

    # Use regex to find emojis and add spaces accordingly
    def add_spaces(match):
        emoji_char = match.group(0)
        return f" {emoji_char} "

    # Create regex pattern to match all emojis
    emoji_pattern = re.compile(r'(' + '|'.join(re.escape(e['emoji']) for e in emojis_found) + r')')

    # Add space before and after emojis
    result = emoji_pattern.sub(add_spaces, text)

    # Normalize spaces (remove duplicates) and strip the result
    return re.sub(r'\s+', ' ', result).strip()

#----------------------------------------------------
# # Test the function
# test_strings = [
#     "masters❤️",
#     "Hello👋World🌍",
#     "No emojis here",
#     "Spaced emoji 😊",
#     "Multiple😀emojis🎉in😎a🚀row"
# ]
#
# for string in test_strings:
#     print(f"Original: {string}")
#     print(add_space_before_emoji(string))
#     print()

#----------------------------------------------------