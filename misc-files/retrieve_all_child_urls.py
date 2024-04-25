# import requests
# from bs4 import BeautifulSoup


# url = 'https://docs.mindsdb.com'
# reqs = requests.get(url)
# soup = BeautifulSoup(reqs.text, 'html.parser')

# urls = []
# for link in soup.find_all('a'):
# 	print(link.get('href'))

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import csv

# Function to get all URLs from a page recursively
def get_all_urls(url, visited_urls=set()):
    print("Visiting URL:", url)

    # Check if the URL has already been visited to avoid infinite loops
    if url in visited_urls or "mindsdb.com/blog" not in url or "https://" not in url:
        print("Already visited:", url)
        return []
	

    # Send a request to the URL
    try:
        # Send a request to the URL with SSL certificate verification disabled
        response = requests.get(url, verify=False)
    except requests.exceptions.SSLError as e:
        print("SSL certificate verification failed for URL:", url)
        return []
    # Update the set of visited URLs
    visited_urls.add(url)

    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract all the links from the page
    links = soup.find_all('a', href=True)

    # List to store all URLs found on this page
    all_urls = []

    for link in links:
        href = link.get('href')
        # Make the URL absolute
        absolute_url = urljoin(url, href)
        # Add the absolute URL to the list
        all_urls.append(absolute_url)
        # Recursively get URLs from this link
        all_urls.extend(get_all_urls(absolute_url, visited_urls))

    return all_urls

def save_urls_to_csv(urls, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["URL"])  # Write header
        for url in urls:
            writer.writerow([url])

# Example usage:
starting_url = "https://mindsdb.com/blog"
all_urls = get_all_urls(starting_url)

# Print all the URLs found
for url in all_urls:
    print(url)

save_urls_to_csv(all_urls, "urls-part2.csv")
