import re

import requests

OWNER = "Undiluted7027"
REPO = "allos-agent-sdk"
README_PATH = "README.md"
url = f"https://api.github.com/repos/{OWNER}/{REPO}/stargazers"
headers = {"Accept": "application/vnd.github.v3.star+json"}
params = {"per_page": 100, "page": 1}
print("Fetching first 100 stargazers...")
response = requests.get(url, headers=headers, params=params)
response.raise_for_status()
data = response.json()
stargazers_md = []
for entry in data:
    user = entry["user"]
    login = user["login"]
    html_url = user["html_url"]
    avatar_url = f"{html_url}.png?size=50"
    stargazers_md.append(f"[![{login}]({avatar_url})]({html_url})")

if not stargazers_md:
    stars_section = "*No stargazers yet. Be the first!*"
else:
    stars_section = " ".join(stargazers_md)
    # Optional: Add count
    stars_section = f"**{len(stargazers_md)} Amazing Stargazers:**\n\n{stars_section}"

# Join avatars
stars_section = " ".join(stargazers_md)
new_content = f"<!-- STARGAZERS:START -->\n{stars_section}\n<!-- STARGAZERS:END -->"
# Update README.md
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
print("✅ README updated with latest stargazers!")
