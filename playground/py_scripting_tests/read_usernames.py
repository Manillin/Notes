import json

# Legge un file JSON e restituisce una lista di username.


def extract_usernames(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # Carica il contenuto del file JSON
            usernames = [entry['username']
                         for entry in data]  # Estrae gli username
        return usernames
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Errore durante la lettura del file: {e}")
        return []


# Sostituisci con il percorso del tuo file JSON
file_followers = "/Users/chris/Notes/followers.json"
file_following = "/Users/chris/Notes/following.json"
follower_list = extract_usernames(file_followers)
following_list = extract_usernames(file_following)


def check_duplicates():
    follower_set = set()
    for elem in follower_list:
        if elem not in follower_set:
            follower_set.add(elem)
        else:
            print(f"{elem} -> already in the set")
    if len(follower_set) == len(follower_list):
        return True

    return False


def check_not_following_back():
    print('#---------------------------------------#\n')
    print(f"       Users not following back: \n")
    print('#---------------------------------------#\n\n')
    for follower in following_list:
        if follower not in follower_list:
            print(f"{follower}")


# check if there are any duplicates
if check_duplicates():
    print(f"Files are valid - no duplicates found\n\n")
    check_not_following_back()
