import requests
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_and_check_proxies(timeout=20):
    """
    Questa funzione:
    1. Recupera automaticamente una lista di proxy unendo più fonti.
    2. Verifica quali proxy sono validi per accedere a Instagram in due fasi:
        a. Prima verifica se il proxy può accedere all'URL.
        b. Poi verifica se la risposta contiene il campo 'data.user'.
    3. Se sono meno di 10 i proxy validi trovati, ripete il controllo sull'intera lista di proxy.
    4. Restituisce la lista dei proxy validi.
    5. Mostra in console un feedback in tempo reale sul processo di validazione.
    6. Inoltre, mostra in console il numero di proxy estratte da ogni singola fonte.
    """

    def get_text_proxies(url):
        """Ritorna una lista di proxy ip:port da una URL che restituisce testo raw in formato ip:port."""
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return [
                    line.strip() for line in response.text.split("\n") if line.strip()
                ]
        except:
            pass
        return []

    def get_proxies_from_proxyscrape_api():
        url = "https://api.proxyscrape.com/v4/free-proxy-list/get?request=display_proxies&proxy_format=ipport&format=text"
        return get_text_proxies(url)

    def get_proxies_from_html_table(url):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return []
            proxies = []
            rows = re.findall(r"<tr>(.*?)</tr>", response.text, flags=re.DOTALL)
            for row in rows:
                cols = re.findall(r"<td>(.*?)</td>", row)
                if len(cols) >= 2:
                    ip = cols[0].strip()
                    port = cols[1].strip()
                    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", ip) and port.isdigit():
                        proxies.append(f"{ip}:{port}")
            return proxies
        except:
            return []

    def get_proxies_from_proxy_list_download():
        url = "https://www.proxy-list.download/api/v1/get?type=http"
        return get_text_proxies(url)

    def get_proxies_from_socks_lists():
        # Queste liste sono di socks4, socks5, http in formato ip:port.
        sources = [
            "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/socks5.txt",
            "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/socks4.txt",
            "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt",
            "https://sunny9577.github.io/proxy-scraper/proxies.txt",
            "https://raw.githubusercontent.com/roosterkid/openproxylist/refs/heads/main/HTTPS_RAW.txt",
            "https://raw.githubusercontent.com/roosterkid/openproxylist/refs/heads/main/SOCKS4_RAW.txt",
            "https://raw.githubusercontent.com/roosterkid/openproxylist/refs/heads/main/SOCKS5_RAW.txt",
            "https://raw.githubusercontent.com/MuRongPIG/Proxy-Master/refs/heads/main/http.txt",
            "https://raw.githubusercontent.com/MuRongPIG/Proxy-Master/refs/heads/main/socks4.txt",
            "https://raw.githubusercontent.com/MuRongPIG/Proxy-Master/refs/heads/main/socks5.txt",
            "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/socks4.txt",
            "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/socks5.txt",
            "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/http.txt",
            "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/https.txt",
        ]

        proxies = []
        for src in sources:
            proxies += get_text_proxies(src)
        validated = []
        for p in proxies:
            parts = p.split(":")
            if (
                len(parts) == 2
                and re.match(r"^\d{1,3}(\.\d{1,3}){3}$", parts[0])
                and parts[1].isdigit()
            ):
                validated.append(p)
        return validated

    def get_proxies_from_monosans():
        # Questo file ha la forma http://port:ip o http://ip:port
        url = "https://raw.githubusercontent.com/monosans/proxy-list/refs/heads/main/proxies_anonymous/all.txt"
        raw_lines = get_text_proxies(url)
        proxies = []
        for line in raw_lines:
            if line.startswith("http://"):
                line = line[len("http://") :]
                parts = line.split(":")
                if len(parts) == 2:
                    part1, part2 = parts[0], parts[1]
                    # part1 e part2 possono essere ip o port
                    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", part1) and part2.isdigit():
                        proxies.append(f"{part1}:{part2}")
                    elif (
                        re.match(r"^\d{1,3}(\.\d{1,3}){3}$", part2) and part1.isdigit()
                    ):
                        proxies.append(f"{part2}:{part1}")
        return proxies

    def get_all_proxies():
        html_sources = {
            "free-proxy-list.net": "https://free-proxy-list.net/",
            "sslproxies.org": "https://sslproxies.org/",
            "us-proxy.org": "https://us-proxy.org/",
            "free-proxy-list.net/uk-proxy.html": "https://free-proxy-list.net/uk-proxy.html",
            "free-proxy-list.net/anonymous-proxy.html": "https://free-proxy-list.net/anonymous-proxy.html",
        }

        sources_count = {}

        from_api = get_proxies_from_proxyscrape_api()
        sources_count["proxyscrape.com"] = len(from_api)

        from_html = []
        for name, src in html_sources.items():
            res = get_proxies_from_html_table(src)
            from_html += res
            sources_count[name] = len(res)

        from_download = get_proxies_from_proxy_list_download()
        sources_count["proxy-list.download"] = len(from_download)

        from_socks_lists = get_proxies_from_socks_lists()
        sources_count["TheSpeedX (socks/http lists)"] = len(from_socks_lists)

        from_monosans = get_proxies_from_monosans()
        sources_count["monosans"] = len(from_monosans)

        all_proxies = (
            from_api + from_html + from_download + from_socks_lists + from_monosans
        )
        unique_proxies = list(set(all_proxies))

        # Stampa in console il numero di proxy estratte da ogni fonte
        sys.stdout.write("Conteggio proxy per fonte:\n")
        total_count = 0
        for source, count in sources_count.items():
            sys.stdout.write(f"{source}: {count}\n")
            total_count += count
        sys.stdout.write(
            f"Totale proxy (prima di unire e rimuovere duplicati): {total_count}\n"
        )
        sys.stdout.write(f"Totale proxy uniche: {len(unique_proxies)}\n")

        return unique_proxies

    def is_proxy_accessible(proxy):
        """Verifica se il proxy può accedere all'URL senza controllare il contenuto."""
        url = "https://www.instagram.com/"
        headers = {
            "User-Agent": "Mozilla/5.0",
        }
        proxies_dict = {"http": f"http://{proxy}", "https": f"http://{proxy}"}
        try:
            response = requests.get(
                url, headers=headers, proxies=proxies_dict, timeout=timeout
            )
            if response.status_code == 200:
                return True
        except:
            pass
        return False

    def check_proxies_accessible(proxies, max_workers_accessible):
        """Prima fase: controlla se il proxy può accedere all'URL."""
        total = len(proxies)
        accessible = []
        with ThreadPoolExecutor(max_workers=max_workers_accessible) as executor:
            futures = {
                executor.submit(is_proxy_accessible, proxy): proxy for proxy in proxies
            }
            for i, future in enumerate(as_completed(futures), start=1):
                proxy = futures[future]
                try:
                    if future.result():
                        accessible.append(proxy)

                except Exception as e:
                    print(f"[{i}/{total}] {proxy} -> ERRORE: {e}")
                sys.stdout.flush()
        return accessible

    def check_proxies(proxies):
        workers = 1000
        # Prima fase: accessibilità
        print("\nInizio della prima fase di verifica (Accessibilità)...")
        accessible_proxies = check_proxies_accessible(proxies, 800)
        print(f"\nProxy accessibili trovati: {len(accessible_proxies)}")

        if len(accessible_proxies) < 40:
            while len(accessible_proxies) < 40:
                print(
                    "Meno di 40 proxy accessibili trovati. Riprovo con l'intera lista..."
                )
                accessible_proxies = check_proxies_accessible(proxies, workers)
                print(f"\nProxy accessibili trovati: {len(accessible_proxies)}")
                if workers >= 100:
                    workers = int(workers / 1.1)
            return accessible_proxies
        return accessible_proxies

    proxy_list = get_all_proxies()
    print(f"\nNumero totale di proxy raccolte: {len(proxy_list)}")

    valid_proxies = check_proxies(proxy_list)

    return valid_proxies
