import requests
import json
import time


def get_followers_and_following(user_id, sessionid, csrftoken):
    headers = {
        "authority": "www.instagram.com",
        "accept": "*/*",
        "cookie": f"sessionid={sessionid}; csrftoken={csrftoken}; ds_user_id={user_id};",
        "referer": f"https://www.instagram.com/username/followers/",  # Sostituisci "username"
        "user-agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1",
        "x-ig-app-id": "936619743392459",
        "x-csrftoken": csrftoken
    }

    followers = []
    following = []
    delay_seconds = 1  # Delay aumentato per sicurezza

    # === Recupera FOLLOWING ===
    next_max_id = None
    page_count = 0
    print("Inizio recupero FOLLOWING...")
    while True:
        params = {"count": 12, "search_surface": "follow_list_page"}
        if next_max_id:
            params["max_id"] = next_max_id

        try:
            response = requests.get(
                f"https://www.instagram.com/api/v1/friendships/{user_id}/following/",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            data = response.json()

            # Deduplica subito gli username
            new_users = [user["username"] for user in data.get(
                "users", []) if user["username"] not in following]
            following.extend(new_users)

            page_count += 1
            print(
                f"Following - Pagina {page_count}: Trovati {len(following)}")

            next_max_id = data.get("next_max_id")
            if not next_max_id:
                print("Following completati!")
                break

            time.sleep(delay_seconds)

        except Exception as e:
            print(f"Errore: {str(e)}")
            break

    # === Recupera FOLLOWERS ===
    next_max_id = None  # Resetta next_max_id!
    page_count = 0
    delay_seconds = 5
    print("\nInizio recupero FOLLOWERS...")
    while True:
        params = {"count": 12, "search_surface": "follow_list_page"}
        if next_max_id:
            params["max_id"] = next_max_id

        try:
            response = requests.get(
                f"https://www.instagram.com/api/v1/friendships/{user_id}/followers/",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            data = response.json()

            # Deduplica e salva solo username unici
            new_users = [user["username"] for user in data.get(
                "users", []) if user["username"] not in followers]
            followers.extend(new_users)

            page_count += 1
            print(
                f"Followers - Pagina {page_count}: Trovati {len(followers)}")

            next_max_id = data.get("next_max_id")
            if not next_max_id:
                print("🏁 Followers completati!")
                break

            time.sleep(delay_seconds)

        except Exception as e:
            print(f"Errore: {str(e)}")
            break

    return followers, following


def save_to_txt(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(data))
    print(f"Salvati {len(data)} username in {filename}")


if __name__ == "__main__":
    # === DATI DI ACCESSO ===
    USER_ID = ""  # Sostituisci
    SESSIONID = ""
    CSRFToken = ""  # Sostituisci

    followers, following = get_followers_and_following(
        USER_ID, SESSIONID, CSRFToken)

    print(
        f"\nTotale Followers: {len(followers)} | Following: {len(following)}")

    save_to_txt(followers, "followers_sorella.txt")
    save_to_txt(following, "following_sorella.txt")
