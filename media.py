# Define the courses with their respective credits and grades
grades_credits = [
    (30, 9),
    (27, 9),
    (24, 9),
    (27, 9),
    (27, 9),
    (24, 9),
    (30, 6),
    (29, 9),
    (27, 6),
    (30, 9),
    (28, 9),
    (28, 6),
    (27, 6),
    (21, 9),
    (27, 12),
    (26, 9),  # calcolo
    (25, 6),  # fisica
    (27, 6),  # statistica
    (30, 6)


]

# Calculate the weighted average
total_weighted_scores = sum(
    grade * credits for grade, credits in grades_credits)
total_credits = sum(credits for _, credits in grades_credits)
weighted_average = total_weighted_scores / total_credits

total_weighted_scores, total_credits, weighted_average

print(f"Media ponderata: {weighted_average}")
