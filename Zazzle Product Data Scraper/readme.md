# Zazzle Product Scraper

An easy to use and understand structured data scraping and analysis project that uses Selenium and BeautifulSoup to collect product metadata from Zazzle listings, clean and normalize the results, and export them for reporting.

## Project Overview

This project demonstrates a complete scraping workflow:
- collect product URLs from a category listing page
- navigate each product page
- extract title, view count, creation date, and tags
- convert and clean scraped values
- create a frequency table for product tags to see which tags are used most
- export the final dataset to CSV


## Prerequisites

- Python
- Google Chrome installed
- ChromeDriver matching your installed Chrome version. Visit: https://developer.chrome.com/docs/chromedriver/downloads
- Python packages:
  - selenium
  - beautifulsoup4
  - pandas
  - regex

## Install dependencies:
pip install selenium beautifulsoup4 pandas regex

## Configuration
Update the ChromeDriver path inside .ipynb before running:
s = Service("C:/path/to/chromedriver.exe")
**If needed, change the initial Zazzle category URL to scrape a different product collection.**

## Usage
Open Zazzle Scraper.ipynb.
Run the cells sequentially:
* import packages
* collect product links
* scrape product pages
* clean and transform data
* analyze tag frequencies
* export the final dataset


## Notebook Workflow

- The notebook is organized in these sections:

- **Necessary Imports**
  - load numpy, pandas, Selenium, BeautifulSoup, and other helpers

- **Getting Product Links from a Category Page**
  - navigate Zazzle search results
  - scroll to load dynamic content
  - parse product link elements

- **Scraping Data From Each Product**
  - visit each product page
  - wait for elements to load
  - extract title, views, publication date, and tags

- **Data Cleaning**
  - parse dates to pandas datetime
  - normalize view counts from strings to integers
  - sort products by latest date

- **Tag Frequency Analysis**
  - expand tags into a frequency table
  - append frequency metadata back to each product

  
## Output

- `Bow Graduation Invitation.csv`
- contains all scraped products
- includes cleaned columns:
  - `Link`
  - `Title`
  - `View`
  - `Created Date`
  - `Tag`
  - `Tag_with_freq`

## Notes

- Zazzle page structure can change; selectors may need updates.
- Use explicit waits to avoid scraping incomplete content.
- Do not overload the site with too-fast requests.
- If scraping fails, inspect the loaded HTML and adjust class selectors.

## Recommended Improvements

- move scraping logic into reusable Python script files
- add retry handling for network or loading failures
- support pagination for multiple result pages
- add logging and error reporting
- build a command-line interface for flexible scraping sessions

## Contribution

Contributions and improvements are welcome:

- add more robust error handling
- refine tag parsing logic
- add automated tests
- document the repository structure and workflow more clearly