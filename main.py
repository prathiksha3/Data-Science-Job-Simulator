import requests
from bs4 import BeautifulSoup
import pandas as pd

base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
pages = 10
page_size = 100

reviews = []

for i in range(1, pages + 1):
    print(f"Scraping page {i}")

    # Create URL to collect links from paginated data
    url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"

    # Collect HTML data from this page
    response = requests.get(url)

    # Parse content
    content = response.content
    parsed_content = BeautifulSoup(content, 'html.parser')
    for para in parsed_content.find_all("div", {"class": "text_content"}):
        review_text = para.get_text()
        
        # Clean the review text (remove "✅ Trip Verified |" if it exists)
        review_text = review_text.replace("✅ Trip Verified |", "").strip()
        
        reviews.append(review_text)
    
    print(f"   ---> {len(reviews)} total reviews")

# Create DataFrame
df = pd.DataFrame()
df["reviews"] = reviews

# Display first few rows of the DataFrame
print(df.head())

# Save DataFrame to CSV file
df.to_csv("data/BA_reviews.csv", index=False)
