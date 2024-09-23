# Define the courses with their respective credits and grades
grades_credits = [
    (30, 9),  # Algebra
    (20, 9),  # Architettura
    (24, 9),  # Analisi 1
    (20, 9),  # Algoritmi
    (20, 9),  # Programmazione 1
    (24, 9),   # Programmazione 2
    (30, 6),   # Oop
    (29, 9),   # Basi di dati
    (27, 6),   # oli
    (30, 9),   # AeSa
    (28, 9),   # SO
    (28, 6),   # Cprog
    (27, 6),   # progSW
    (21, 9),   # Protocolli
    (27, 12),  # Compilatori
    (20, 6),   # fisica
    (25, 6),   # statistica
    (27, 9),   # Calcolo numerico
    (28, 6)    # TechWeb


]

# Calculate the weighted average
total_weighted_scores = sum(
    grade * credits for grade, credits in grades_credits)
total_credits = sum(credits for _, credits in grades_credits)
weighted_average = total_weighted_scores / total_credits

total_weighted_scores, total_credits, weighted_average

print(f"Media ponderata: {weighted_average}")
