def sum_invalid_in_range(start: int, end: int):
    total = 0
    max_half = len(str(end)) // 2
    for h in range(1, max_half + 1):
        mult = 10**h + 1
        min_x = (start + mult - 1) // mult
        max_x = end // mult
        low = max(min_x, 10**(h - 1))
        high = min(max_x, 10**h - 1)
        if low <= high:
            count = high - low + 1
            sum_x = (low + high) * count // 2
            total += mult * sum_x
    return total


with open('day2.txt', 'r') as f:
    data = f.readline().strip()

ranges = [r.strip() for r in data.split(',') if r.strip()]

grand_total = 0
for rng in ranges:
    s, e = rng.split('-')
    start, end = int(s), int(e)
    subtotal = sum_invalid_in_range(start, end)
    print(f"{start}-{end} -> subtotal sum of invalid IDs: {subtotal}")
    grand_total += subtotal

print("Grand total sum:", grand_total)
