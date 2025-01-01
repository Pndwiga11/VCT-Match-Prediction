import requests
import time
import random
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

def scrape_match_details(html):
    """Scrape details of a single match from its URL."""
    
    soup = BeautifulSoup(html, 'html.parser')
    
    def clean_text(text):
            return text.replace('\t', '').replace('\n', '').strip()
    
    try:
        # Extract teams
        team_names = soup.find_all('div', class_='wf-title-med')
        team_a = clean_text(team_names[0].text)
        team_b = clean_text(team_names[1].text)

        # Extract scores
        team_scores = soup.find_all('div', class_='js-spoiler')
        team_a_score =  team_scores[0].text.strip()[0]
        team_b_score = team_scores[0].text.strip()[33]

        # Extract date
        match_date_element = soup.find('div', class_='moment-tz-convert')
        match_date = match_date_element['data-utc-ts'].split(' ')[0]  # Extract the date (YYYY-MM-DD)

        # Match Outcome
        if team_a_score > team_b_score:
            outcome = 1
        elif team_b_score > team_a_score:
            outcome = -1
        else:
            outcome = 0
            
        return {
            'Outcome': outcome,
            'Team A': team_a,
            'Team B': team_b,
            'Score A': team_a_score,
            'Score B': team_b_score,
            'Date': match_date
        }
    except AttributeError:
        print(f"Failed to parse details from url")
        return None
    
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
        
        print(f"Scraping match: {link_num} of {len(match_links)}")
        link_num += 1
        match_details = scrape_match_details(match_html_content)
        #match_details = False
        if match_details:
            match_data.append(match_details)
        time.sleep(random.uniform(2, 10)) # Sleep for a random delay of 2-10 seconds
    return match_data

def combine_tournaments(file1, file2, output_file):
    try:
        # Load both datasets
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # Combine datasets
        combined_df = pd.concat([df1, df2], ignore_index=True)

        # Save to the specified output file
        combined_df.to_csv(output_file, index=False)
        print(f"Combined dataset saved to {output_file}")
    except Exception as e:
        print(f"An error occurred while combining files: {e}")


# Function to save data to CSV
def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# Main script
if __name__ == '__main__':
    tournament_url = 'https://www.vlr.gg/event/matches/324/champions-tour-north-america-stage-1-challengers-3/?series_id=all'
    tournament_html_content = fetch_html(tournament_url)
    
    file1 = "data/combined_tournaments.csv"
    file2 = "data/2021_VCT_NA_Stage1Challengers3.csv"
    output_file = "data/combined_tournaments.csv"
    
    combine_tournaments(file1, file2, output_file)
    


    #if tournament_html_content:
        #match_data = scrape_tournament_history(tournament_html_content)
        #save_to_csv(match_data, 'data/2021_VCT_NA_Stage1Challengers3.csv')
    