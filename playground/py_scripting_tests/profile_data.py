import requests
import json
import time


def get_followers(user_id, sessionid, csrftoken, username, delay_following, delay_followers):
    headers = {
        "authority": "www.instagram.com",
        "accept": "*/*",
        "accept-language": "it-IT,it;q=0.9",
        "cookie": f"sessionid={sessionid}; ds_user_id={user_id};",
        "referer": f"https://www.instagram.com/{username}/followers/",
        "user-agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1",
        "x-ig-app-id": "936619743392459",
        "x-csrftoken": csrftoken
    }

    followers = []
    following = []
    next_max_id = None
    page_count = 0

    print(f"Checking for: \nAccount: {username} - UserID: {user_id}\n")
    time.sleep(2)

    # Following
    while True:
        # Costruisci i parametri della richiesta
        params = {
            "count": 25,
            "search_surface": "follow_list_page"
        }
        if next_max_id:
            params["max_id"] = next_max_id

        try:
            # Effettua la richiesta
            response = requests.get(
                f"https://www.instagram.com/api/v1/friendships/{user_id}/following/",
                headers=headers,
                params=params
            )

            # Controlla lo status code
            if response.status_code != 200:
                print(f"Errore {response.status_code}: {response.text}")
                break

            data = response.json()

            # Aggiungi i follower alla lista
            following.extend(data.get("users", []))
            page_count += 1

            # Messaggio di successo
            print(
                f"Pagina {page_count} - Trovati {len(following)} following (max_id: {next_max_id})")

            # Aggiorna next_max_id per la prossima pagina
            next_max_id = data.get("next_max_id")

            # Interrompi se non ci sono altre pagine
            if not next_max_id:
                print(
                    "Paginazione following completata! procedo con paginazione follower")
                break

            # Aspetta per evitare rate limit
            print(f"⏳ Attendo {delay_following} secondi...")
            time.sleep(delay_following)

        except Exception as e:
            print(f"❌ Errore durante la richiesta: {str(e)}")
            break

    # reset max id
    next_max_id = None

    # Followers
    while True:
        params = {
            "count": 20,
            "search_surface": "follow_list_page"
        }
        if next_max_id:
            params["max_id"] = next_max_id
        try:
            response = requests.get(
                f"https://www.instagram.com/api/v1/friendships/{user_id}/followers/",
                headers=headers,
                params=params
            )
            if response.status_code != 200:
                print(f"Errore {response.status_code}: {response.text}")
                break

            data = response.json()
            followers.extend(data.get("users", []))
            page_count += 1
            print(
                f"Pagina {page_count} - Trovati {len(followers)} follower (max_id: {next_max_id})")

            next_max_id = data.get("next_max_id")
            if not next_max_id:
                print("Paginazione completata!")
                break
            print(f"Attendo {delay_followers} secondi...")
            time.sleep(delay_followers)

        except Exception as e:
            print(f"Errore durante la richiesta: {str(e)}")
            break

    return [followers, following]


def save_to_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Salvati {len(data)} follower in {filename}")


if __name__ == "__main__":

    USER_ID = ""
    SESSIONID = ""
    CSRFToken = ""
    USERNAME = ""
    FOLLOWING_FILE = "following2.json"
    FOLLOWERS_FILE = "followers2.json"
    DELAY_FOLLOWERS, DELAY_FOLLOWING = 1.55, 1.55  # high delays to avoid rate limits
    follow = get_followers(USER_ID, SESSIONID, CSRFToken,
                           USERNAME, DELAY_FOLLOWING, DELAY_FOLLOWERS)
    followers = follow[0]
    following = follow[1]

    print(
        f"User has Followers: {len(followers)} - Following: {len(following)} ")

    save_to_json([{
        "username": user.get("username"),
    } for user in followers], FOLLOWERS_FILE)
    save_to_json([{
        "username": user.get("username"),
    } for user in following], FOLLOWING_FILE)
