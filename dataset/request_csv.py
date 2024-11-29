import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm  # Progress bar library

# URL of the directory containing ZIP files
base_url = "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/"

# Output directory to save the files
output_dir = "CICIDS2017_CSVs"
os.makedirs(output_dir, exist_ok=True)

# Fetch the directory page
response = requests.get(base_url)
if response.status_code != 200:
    print("Failed to access the URL. Status code:", response.status_code)
    exit()

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Find all links to ZIP files
zip_links = []
for link in soup.find_all('a'):
    href = link.get('href')
    if href and href.endswith('.zip'):
        zip_links.append(base_url + href)

# Download each ZIP file with progress bar
for zip_url in zip_links:
    filename = os.path.join(output_dir, os.path.basename(zip_url))
    print(f"Downloading {zip_url} to {filename}...")
    try:
        # Stream the download to get file size
        with requests.get(zip_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(filename, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Failed to download {zip_url}: {e}")

print("All downloads completed!")
