def roman_to_int(s: str) -> int:
    roman_numerals = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000

    }

    total = 0
    prev_value = 0

    for char in reversed(s):
        value = roman_numerals[char]
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value

    return total


# Esempio di utilizzo:
roman_number = "MCMXCIV"
print(roman_to_int(roman_number))  # Output: 1994
