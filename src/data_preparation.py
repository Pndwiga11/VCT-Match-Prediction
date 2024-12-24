import requests
from bs4 import BeautifulSoup
import pandas as pd

# Function to fetch HTML content
def fetch_html(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch {url}")
        return None

# Function to parse matches from HTML
def parse_matches(html):
    soup = BeautifulSoup(html, 'html.parser')
    match_data = []

    # Example: Scraping match info from VLR.gg
    matches_cards = soup.find_all('div', class_='wf-card')
    #matches_dates = soup.find_all('div', class_='wf-label.mod-large')
    
    tally = -1
    # matches = matches_card.find_all('div', class_='match-item-vs')  # Adjust this based on actual HTML structure
    for match_group in matches_cards:
        tally += 1
        try:
            matches = match_group.find_all('div', class_='match-item-vs')  # Adjust this based on actual HTML structure
            #match_date = matches_dates[0].text.strip()
            #match_date = 12
            for match in matches:
                try:
                    team_names = match.find_all('div', class_='text-of')
                    team_scores = match.find_all('div', class_='match-item-vs-team-score')
                    
                    team_a = team_names[0].text.strip()  # Team A is always the first
                    team_b = team_names[1].text.strip()  # Team B is always the second
                    team_a_score = team_scores[0].text.strip()  # Team B is always the second
                    team_b_score = team_scores[1].text.strip()  # Team B is always the second
                    match_data.append({
                        'Team A': team_a,
                        'Team B': team_b,
                        'Score A': team_a_score,
                        'Score B': team_b_score,
                        #'Date': match_date
                    })
                except AttributeError:
                    print("inner error")
                    continue  # Skip if any field is missing
                
                
            #match_details = match.find('div', class_='match-item-vs-team mod-winner').text.strip()
            
            #score_a = match.find('div', class_='match-item-time').text.strip()
            #score_b = match.find('div', class_='match-item-time').text.strip()
           # date = match.find('div', class_='match-item-time').text.strip()

            
        except AttributeError:
            print("outer error")
            continue  # Skip if any field is missing

    return match_data

def scrape_match_details(html):
    """Scrape details of a single match from its URL."""
    
    soup = BeautifulSoup(html, 'html.parser')
    match_data = []
    
    try:
        # Extract teams
        team_names = soup.find_all('div', class_='wf-title-med')
        team_a = team_names[0].text.strip()
        team_b = team_names[1].text.strip()

        # Extract scores
        team_scores = soup.find_all('div', class_='js-spoiler')
        team_a_score =  team_scores[0].text.strip()[0]
        team_b_score = team_scores[0].text.strip()[33]

        # Extract date
        match_date_list = soup.find('div', class_='moment-tz-convert').text.strip()
        match_date = soup.find('div', class_='match-header-date').text.strip()

        match_data.append({
            'Team A': team_a,
            'Team B': team_b,
            'Score A': team_a_score,
            'Score B': team_b_score,
            'Date': match_date_list
        })
    except AttributeError:
        print(f"Failed to parse details from url")
        return None
    
    return match_data

def scrape_tournament_history(html):
    """Scrape all matches from a tournament page and return detailed data."""

    soup = BeautifulSoup(html, 'html.parser')
    match_data = []
    # Find all match links
    match_links = soup.find_all('a', href=True, class_= 'wf-module-item')  # Update based on the site structure
        
        
    
    
    base_url = "https://www.vlr.gg"  # Base URL for relative links
    link_num = 1
    for link in match_links:
        match_html = base_url + link['href']  # Construct full URL
        match_html_content = fetch_html(match_html)
        
        #print(f"Scraping match: {link_num}")
        link_num += 1
        match_details = scrape_match_details(match_html_content)
        #match_details = False
        if match_details:
            match_data.append(match_details)
    return match_data


# Function to save data to CSV
def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# Main script
if __name__ == '__main__':
    base_url = 'https://www.vlr.gg/9109/100-thieves-vs-the-five-emperors-champions-tour-north-america-stage-1-challengers-1-ro128'  # Example VLR URL
    html_content = fetch_html(base_url)
    tournament_url = 'https://www.vlr.gg/event/matches/291/champions-tour-north-america-stage-1-challengers-1/?series_id=all'
    tournament_html_content = fetch_html(tournament_url)
    
    if html_content:
        match_data = scrape_tournament_history(tournament_html_content)
        #match_data = scrape_match_details(html_content)
        save_to_csv(match_data, 'data/matches.csv')
