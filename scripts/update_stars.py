# scripts/update_stars.py
import re

import requests

OWNER = "Undiluted7027"
REPO = "allos-agent-sdk"
README_PATH = "README.md"
IMAGES_PER_ROW = 10

url = f"https://api.github.com/repos/{OWNER}/{REPO}/stargazers"
headers = {"Accept": "application/vnd.github.v3.star+json"}
params = {"per_page": 100, "page": 1}

print("Fetching first 100 stargazers...")
response = requests.get(url, headers=headers, params=params)
response.raise_for_status()
data = response.json()

stargazers_html = []
for entry in data:
    user = entry["user"]
    login = user["login"]
    html_url = user["html_url"]
    avatar_url = user["avatar_url"] + "&s=50"  # Add size parameter

    # ✅ USE HTML, NOT MARKDOWN!
    html_img = f'<a href="{html_url}"><img src="{avatar_url}" width="50" height="50" alt="{login}"/></a>'
    stargazers_html.append(html_img)

# Handle empty state
if not stargazers_html:
    stars_section = "*No stargazers yet. Be the first!* ⭐"
else:
    # Create table rows
    rows = []
    for i in range(0, len(stargazers_html), IMAGES_PER_ROW):
        row_items = stargazers_html[i : i + IMAGES_PER_ROW]
        cells = "".join([f"<td align='center'>{item}</td>" for item in row_items])
        rows.append(f"<tr>{cells}</tr>")

    table = "<table>\n" + "\n".join(rows) + "\n</table>"

    # Add count
    count_text = f"**{len(stargazers_html)} Amazing Stargazer{'s' if len(stargazers_html) != 1 else ''}:**\n\n"
    stars_section = count_text + table

new_content = f"<!-- STARGAZERS:START -->\n{stars_section}\n<!-- STARGAZERS:END -->"

# Update README
with open(README_PATH, "r", encoding="utf-8") as f:
    readme = f.read()

updated = re.sub(
    r"<!-- STARGAZERS:START -->.*<!-- STARGAZERS:END -->",
    new_content,
    readme,
    flags=re.DOTALL,
)

with open(README_PATH, "w", encoding="utf-8") as f:
    f.write(updated)

print(f"✅ README updated with {len(stargazers_html)} stargazers!")
