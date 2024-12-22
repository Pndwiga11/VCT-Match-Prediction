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
    matches = soup.find_all('div', class_='match-item-vs')  # Adjust this based on actual HTML structure
    for match in matches:
        try:
            team_names = match.find_all('div', class_='text-of')
            team_scores = match.find_all('div', class_='match-item-vs-team-score')
            
            #match_details = match.find('div', class_='match-item-vs-team mod-winner').text.strip()
            team_a = team_names[0].text.strip()  # Team A is always the first
            team_b = team_names[1].text.strip()  # Team B is always the second
            team_a_score = team_scores[0].text.strip()  # Team B is always the second
            team_b_score = team_scores[1].text.strip()  # Team B is always the second
            #score_a = match.find('div', class_='match-item-time').text.strip()
            #score_b = match.find('div', class_='match-item-time').text.strip()
           # date = match.find('div', class_='match-item-time').text.strip()

            match_data.append({
                'Team A': team_a,
                'Team B': team_b,
                'Score A': team_a_score,
                'Score B': team_b_score,
                #'Date': date
            })
        except AttributeError:
            continue  # Skip if any field is missing

    return match_data

# Function to save data to CSV
def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# Main script
if __name__ == '__main__':
    base_url = 'https://www.vlr.gg/matches/results'  # Example VLR URL
    html_content = fetch_html(base_url)
    
    if html_content:
        match_data = parse_matches(html_content)
        save_to_csv(match_data, 'data/matches.csv')
