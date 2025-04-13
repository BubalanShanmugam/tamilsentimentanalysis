
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.keys import Keys
# from webdriver_manager.chrome import ChromeDriverManager
# import time

# # Set up Chrome options
# options = webdriver.ChromeOptions()
# options.add_argument("--headless=new")  # New headless mode
# options.add_argument("--disable-gpu")
# options.add_argument("--disable-software-rasterizer")
# options.add_argument("--disable-dev-shm-usage")
# options.add_argument("--no-sandbox")

# # Start Chrome driver
# driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# # YouTube video URL
# video_url = "https://youtu.be/IJA6wovq7lI?si=htm7wD66eXuV_ChO"SUPER STAR only rajani sir SEMA mass🤗🤗
# driver.get(video_url)

# # Wait for comments to load
# time.sleep(5)

# # Scroll down to load more comments
# for _ in range(5):  # Scroll multiple times to load all comments
#     driver.find_element(By.TAG_NAME, "body").send_keys(Keys.PAGE_DOWN)
#     time.sleep(2)

# # Extract comments
# comments = driver.find_elements(By.XPATH, '//*[@id="content-text"]')
# comment_list = [comment.text for comment in comments if comment.text.strip()]

# # Save to text file
# with open("youtube_comments.txt", "w", encoding="utf-8") as f:
#     for comment in comment_list:
#         f.write(comment + "\n")

# print(f"Saved {len(comment_list)} comments to youtube_comments.txt")

# # Close browser
# driver.quit()






# instagram

# import requests

# # Your Instagram Access Token (Get from Meta Developer)
# ACCESS_TOKEN = "YOUR_INSTAGRAM_ACCESS_TOKEN"
# MEDIA_ID = "YOUR_REEL_MEDIA_ID"  # Get this from your Instagram reel URL

# # Fetch Comments API
# url = f"https://graph.facebook.com/v18.0/{MEDIA_ID}/comments?access_token={ACCESS_TOKEN}"

# response = requests.get(url)
# comments = response.json()
# print (comments)

# # Save to a file
# with open("instagram_comments.txt", "w", encoding="utf-8") as file:
#     for comment in comments.get("data", []):
#         file.write(comment["text"] + "\n")

# print(f"Saved {len(comments.get('data', []))} comments to instagram_comments.txt")
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time

def scrape_youtube_comments(video_url):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(video_url)

    time.sleep(5)

    for _ in range(5):
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.PAGE_DOWN)
        time.sleep(2)

    comments = driver.find_elements(By.XPATH, '//*[@id="content-text"]')
    comment_list = [comment.text for comment in comments if comment.text.strip()]

    with open("youtube_comments.txt", "w", encoding="utf-8") as f:
        for comment in comment_list:
            f.write(comment + "\n")

    print(f"Saved {len(comment_list)} comments to youtube_comments.txt")
    driver.quit()

video_url = input("Enter YouTube video URL: ")
scrape_youtube_comments(video_url)
