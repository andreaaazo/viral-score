import requests
import json
import time
from statistics import mean
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.align import Align
from rich.markdown import Markdown
from rich.rule import Rule
from itertools import cycle
from proxy import fetch_and_check_proxies
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import xml.etree.ElementTree as ET

console = Console()

# Lock per accesso ai proxy
proxy_lock = threading.Lock()


def get_next_proxy(proxy_cycle):
    with proxy_lock:
        return next(proxy_cycle)


def get_initial_profile_data(username, session, proxy_cycle, max_retries=50):
    url = (
        f"https://www.instagram.com/api/v1/users/web_profile_info/?username={username}"
    )
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": f"https://www.instagram.com/{username}/",
        "X-IG-App-ID": "936619743392459",
        "X-ASBD-ID": "198387",
        "X-Requested-With": "XMLHttpRequest",
    }

    for _ in range(max_retries):
        proxy = get_next_proxy(proxy_cycle)
        proxies = {"http": f"http://{proxy}", "https": f"http://{proxy}"}
        try:
            r = session.get(
                url,
                headers=headers,
                proxies=proxies,
                timeout=20,
            )
            if r.status_code == 200:
                return r.json()

        except:
            continue
    # Se arriva qui, tutti i tentativi sono falliti
    raise Exception("Errore nella richiesta iniziale dopo diversi tentativi")


def get_paginated_data(user_id, end_cursor, session, proxy_cycle, max_retries=50):
    query_hash = "58b6785bea111c67129decbe6a448951"
    variables = {"id": user_id, "first": 12, "after": end_cursor}
    params = {"query_hash": query_hash, "variables": json.dumps(variables)}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "X-IG-App-ID": "936619743392459",
        "X-ASBD-ID": "198387",
        "X-Requested-With": "XMLHttpRequest",
    }

    for _ in range(max_retries):
        proxy = get_next_proxy(proxy_cycle)
        proxies = {"http": f"http://{proxy}", "https": f"http://{proxy}"}
        try:
            r = session.get(
                "https://www.instagram.com/graphql/query/",
                params=params,
                headers=headers,
                proxies=proxies,
                timeout=20,
            )
            if r.status_code == 200:
                return r.json()

        except:
            continue

    raise Exception("Errore nella paginazione dopo diversi tentativi")


def extract_video_info(node, followers, username):
    if node.get("is_video"):
        video_url = node.get("video_url")
        likes = node.get("edge_media_preview_like", {}).get("count")
        comments = node.get("edge_media_to_comment", {}).get("count")
        views = node.get("video_view_count")
        posted_time = node.get("taken_at_timestamp")
        video_duration = get_video_duration(node)
        dimensions = node.get("dimensions", {})
        width = dimensions.get("width")
        height = dimensions.get("height")
        if likes == -1:  # Disabled likes
            return None
        elif video_duration == None:
            return None
        elif posted_time <= 1000000000:
            print(posted_time + video_url)
            with open("errors.json", "a") as file_name:
                file_name.write(json.dumps(node))
                file_name.close()
            return None
        else:
            return {
                "username": username,  # Aggiunto il campo username
                "url": video_url,
                "likes": likes,
                "comments": comments,
                "views": views,
                "followers": followers,
                "posted_time": posted_time,
                "video_duration": video_duration,
                "dimensions": {
                    "width": width,
                    "height": height,
                },
            }
    return None


def get_video_duration(node):
    try:
        # Extract the XML string
        xml_string = node.get("dash_info", {}).get("video_dash_manifest")

        # Parse the XML
        root = ET.fromstring(xml_string)

        # Find the mediaPresentationDuration attribute in the MPD tag
        duration = root.attrib.get("mediaPresentationDuration")

        # Parse ISO 8601 duration format (e.g., PT0H0M18.100S)
        duration = duration.replace("PT", "")
        hours, minutes, seconds = 0, 0, 0.0

        if "H" in duration:
            hours, duration = duration.split("H")
            hours = int(hours)

        if "M" in duration:
            minutes, duration = duration.split("M")
            minutes = int(minutes)

        if "S" in duration:
            seconds = float(duration.replace("S", ""))

        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds

    except Exception:
        return None


def scrape_user(username, session, proxy_cycle, max_posts=300):
    initial_data = get_initial_profile_data(username, session, proxy_cycle)
    user = initial_data["data"]["user"]
    user_id = user["id"]
    username = user["username"]
    followers = user.get("edge_followed_by", {}).get("count", 0)

    edges = user["edge_owner_to_timeline_media"]["edges"]
    page_info = user["edge_owner_to_timeline_media"]["page_info"]

    videos_data = []
    count = 0

    # Primo batch di edges
    for edge in edges:
        node = edge["node"]
        count += 1
        video_info = extract_video_info(node, followers, username)  # Passato username
        if video_info is not None:
            videos_data.append(video_info)
        if count >= max_posts:
            break

    # Pagination
    has_next_page = page_info["has_next_page"]
    end_cursor = page_info["end_cursor"]

    while has_next_page and end_cursor and count < max_posts:
        data_page = get_paginated_data(user_id, end_cursor, session, proxy_cycle)
        media_edges = data_page["data"]["user"]["edge_owner_to_timeline_media"]["edges"]
        page_info = data_page["data"]["user"]["edge_owner_to_timeline_media"][
            "page_info"
        ]

        for edge in media_edges:
            node = edge["node"]
            count += 1
            video_info = extract_video_info(
                node, followers, username
            )  # Passato username
            if video_info is not None:
                videos_data.append(video_info)
            if count >= max_posts:
                break

        if count >= max_posts:
            break
        has_next_page = page_info["has_next_page"]
        end_cursor = page_info["end_cursor"]

    return videos_data


def calculate_engagement_rate(data):
    result = []
    for post in data:
        if post is not None:
            likes = post.get("likes", 0)
            comments = post.get("comments", 0)
            views = post.get("views", 0)
            followers = post.get("followers", 1)
            engagement_rate = ((likes + comments + views) / followers) * 100

            new_post = dict(post)
            new_post["engagement_rate"] = engagement_rate
            result.append(new_post)
        else:
            pass

    result.sort(key=lambda x: x["engagement_rate"], reverse=True)
    return result


def get_list_of_accounts(file_name):
    accounts = set()
    with open(file_name, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                accounts.add(line)
    return list(accounts)


def main():
    # Lista di proxy funzionanti
    proxies_list = fetch_and_check_proxies()
    console.log(f"[green]Proxy funzionanti trovati: {len(proxies_list)}[/green]")

    if not proxies_list:
        console.log("[red]Nessun proxy funzionante trovato. Esco dal programma.[/red]")
        return

    # Crea un ciclo di proxy
    proxy_cycle = cycle(proxies_list)
    usernames_file = "cars_profiles.txt"
    usernames = get_list_of_accounts(usernames_file)

    # Banner iniziale
    banner_text = """[bold magenta]
   ____       _                 
  / ___|  ___| |_ _   _ _ __   
  \___ \ / _ \ __| | | | '_ \  
   ___) |  __/ |_| |_| | |_) | 
  |____/ \___|\__|\__,_| .__/  
                       |_|     
Scraping Instagram in modo estremo
[/bold magenta]"""
    console.print(Panel(Align.center(banner_text), box=box.DOUBLE, style="magenta"))

    # Step iniziali con spinner per effetto
    with console.status(
        "[bold cyan]Verifica ambiente...[/bold cyan]", spinner="line"
    ) as status:
        time.sleep(1)
    console.log("[green]Ambiente OK![/green]")

    with console.status(
        "[bold cyan]Connessione a Instagram...[/bold cyan]", spinner="growVertical"
    ) as status:
        time.sleep(1.5)
    console.log("[green]Connessione stabilita![/green]")

    with console.status(
        "[bold cyan]Caricamento configurazioni utenti...[/bold cyan]", spinner="dots"
    ) as status:
        time.sleep(1)
    console.log(f"[green]{len(usernames)} utenti caricati![/green]\n")

    session = requests.Session()
    all_videos = []
    user_results = []

    console.print(Rule("[bold blue]Inizio scraping dei profili[/bold blue]"))

    # Utilizziamo un ThreadPoolExecutor per parallelizzare lo scraping
    max_workers = 100  # Puoi modificare questo valore
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
        refresh_per_second=5,
    ) as progress:
        task = progress.add_task("Scraping utenti...", total=len(usernames))

        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for user in usernames:
                future = executor.submit(scrape_user, user, session, proxy_cycle)
                futures[future] = user

            for future in as_completed(futures):
                user = futures[future]
                try:
                    user_videos = future.result()
                    all_videos.extend(user_videos)
                    user_results.append((user, len(user_videos), "OK"))
                    console.log(
                        f"[green]Utente {user} completato: {len(user_videos)} video estratti.[/green]"
                    )
                except Exception as e:
                    user_results.append((user, 0, f"Errore: {e}"))
                    console.log(f"[red]Errore con l'utente {user}: {e}[/red]")
                progress.advance(task)

    console.print(Rule("[bold blue]Calcolo statistiche[/bold blue]"))

    with console.status(
        "[bold cyan]Calcolo engagement rate e statistiche...[/bold cyan]",
        spinner="earth",
    ) as status:
        final_data = calculate_engagement_rate(all_videos)
        time.sleep(1)

    with open(
        ("data" + "/" + os.path.splitext(usernames_file)[0] + ".json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    total_videos = len(final_data)
    likes_list = [v["likes"] for v in final_data]
    comments_list = [v["comments"] for v in final_data]
    views_list = [v["views"] for v in final_data]
    eng_list = [v["engagement_rate"] for v in final_data]

    avg_likes = mean(likes_list) if likes_list else 0
    avg_comments = mean(comments_list) if comments_list else 0
    avg_views = mean(views_list) if views_list else 0
    avg_eng = mean(eng_list) if eng_list else 0
    min_eng = min(eng_list) if eng_list else 0
    max_eng = max(eng_list) if eng_list else 0
    top_5 = final_data[:5]

    console.print(Rule("[bold green]Resoconto Finale[/bold green]"))

    # Tabella dei risultati per utente
    user_table = Table(title="Risultati per Utente", box=box.MINIMAL_DOUBLE_HEAD)
    user_table.add_column("Utente", justify="left", style="cyan", no_wrap=True)
    user_table.add_column("Video Estratti", justify="right", style="green")
    user_table.add_column("Stato", justify="left", style="magenta")

    for user, count, status in user_results:
        user_table.add_row(user, str(count), status)

    # Tabella dei top 5 post
    top_table = Table(
        title="Top 5 Post per Engagement Rate", box=box.MINIMAL_DOUBLE_HEAD
    )
    top_table.add_column("Username", justify="left", style="cyan")  # Aggiunto Username
    top_table.add_column("URL", justify="left", style="cyan")
    top_table.add_column("Likes", justify="right", style="green")
    top_table.add_column("Comments", justify="right", style="green")
    top_table.add_column("Views", justify="right", style="green")
    top_table.add_column("Engagement Rate (%)", justify="right", style="yellow")

    for post in top_5:
        top_table.add_row(
            post.get("username", "N/A"),  # Mostra il username
            post.get("url", "N/A"),
            str(post["likes"]),
            str(post["comments"]),
            str(post["views"]),
            f"{post['engagement_rate']:.2f}",
        )

    stats_markdown = f"""
# Statistiche Finali

- **Video totali salvati:** {total_videos}
- **Engagement medio:** {avg_eng:.2f}%
- **Engagement min:** {min_eng:.2f}%
- **Engagement max:** {max_eng:.2f}%
- **Likes medi per video:** {avg_likes:.2f}
- **Commenti medi per video:** {avg_comments:.2f}
- **Visualizzazioni medie:** {avg_views:.2f}

_Dati salvati in_ videos.json  
"""

    console.print(Markdown(stats_markdown))
    console.print(user_table)
    console.print(top_table)
    console.print(
        Panel(
            "[bold green]Scraping completato con successo![/bold green]",
            style="green",
            box=box.DOUBLE,
        )
    )
    console.print(Rule("[bold red]FINE[/bold red]"))
    console.print(
        "[bold blue]Grazie per aver utilizzato il nostro scraper Instagram estremo![/bold blue]\n"
    )


if __name__ == "__main__":
    main()
