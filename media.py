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
    (26, 9),   # Calcolo numerico
    (30, 6)    # TechWeb
]

# 28.10 if adding extra credit for graduation in time
# 28.36 should be lower bound for 110
# 28.63 should be lower bound for lode

magistrale = {
    (28, 9),  # High Performance Computing
    (30, 6),  # Metodologie Sviluppo SW
    (30, 9),  # Big Data Analytics
    (30, 9),  # Sviluppo SW Sicuro
    (28, 6),  # Fondamenti di Machine Learning
    (30, 6),  # Sistemi Complessi
    (30, 6),  # Privacy | Diritto
    (30, 6),  # Teoria dei Giochi
    (27, 9),  # Algorimi Distribuiti
    (30, 6),  # IoT Systems
    (30, 6),  # Cloud and Edge Computing
    (29, 6),  # Computational and Statistical Learning
    (27, 6)   # Algoritmi di Ottimizzazione
}

# Calculate the weighted average
total_weighted_scores = sum(
    grade * credits for grade, credits in magistrale)
total_credits = sum(credits for _, credits in magistrale)
weighted_average = total_weighted_scores / total_credits


total_weighted_scores, total_credits, weighted_average
voto_ingresso = (weighted_average * 110)/30

print(f"Media ponderata: {weighted_average}")
# print(f'Crediti: {}')
print(f'Voto ingresso {voto_ingresso}')
