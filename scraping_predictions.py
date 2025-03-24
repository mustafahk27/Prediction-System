import os
from dotenv import load_dotenv
from hyperbrowser import Hyperbrowser
from bs4 import BeautifulSoup
from hyperbrowser.models import (
    ScrapeOptions,
    StartScrapeJobParams,
    CreateSessionParams
)
import json
from datetime import datetime

async def get_coin_predictions(coin_name: str) -> dict:
    """
    Get 30-day predictions for a given coin from digitalcoinprice.com
    
    Args:
        coin_name: Name/symbol of the cryptocurrency
    Returns:
        Dictionary containing the predictions data
    """
    # Load environment variables from .env file
    load_dotenv()

    # Get API key
    api_key = os.getenv("HYPERBROWSER_API_KEY")
    if not api_key:
        raise ValueError("HYPERBROWSER_API_KEY not found in environment variables. Please check your .env file.")

    client = Hyperbrowser(api_key=api_key)
    url = f"https://digitalcoinprice.com/forecast/{coin_name.lower()}"

    try:
        scrape_result = client.scrape.start_and_wait(
            StartScrapeJobParams(
                url=url,
                session_options=CreateSessionParams(
                    accept_cookies=False,
                    use_stealth=False,
                    use_proxy=False,
                    solve_captchas=False,
                ),
                scrape_options=ScrapeOptions(
                    formats=["html"],
                    only_main_content=True,
                    exclude_tags=[],
                    include_tags=[],
                ),
            )
        )
        
        # Get the HTML content from the scrape result
        if hasattr(scrape_result, 'content'):
            html_content = scrape_result.content
        elif hasattr(scrape_result, 'html'):
            html_content = scrape_result.html
        else:
            html_content = str(scrape_result)  # Fallback to string representation
        
        # Parse the HTML content
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Initialize prediction data
        prediction_data = {
            "coin": coin_name.upper(),
            "source_url": url,
            "timestamp": datetime.now().isoformat(),
            "daily_predictions": []
        }
        
        # Find the daily predictions section
        daily_heading = soup.find(string=lambda text: text and "Price Prediction for Tomorrow and Next Week" in text)
        
        if daily_heading:
            # Find the next table after this heading
            table = daily_heading.find_next("table")
            
            if table:
                # Process all rows including header
                rows = table.find_all("tr")
                
                # Skip header row and process data rows, limit to 30 days
                processed_days = 0
                for row in rows[1:]:  # Skip header row
                    if processed_days >= 30:  # Stop after 30 days
                        break
                        
                    cols = row.find_all("td")
                    if len(cols) >= 3:  # Ensure we have date and price columns
                        date = cols[0].text.strip()
                        price = cols[1].text.strip()
                        change = cols[2].text.strip()
                        
                        # Clean the data
                        price = price.replace("$", "").strip()
                        change = change.replace("%", "").strip()
                        
                        try:
                            prediction_data["daily_predictions"].append({
                                "date": date,
                                "price": float(price),
                                "change_percentage": float(change)
                            })
                            processed_days += 1
                        except ValueError as ve:
                            print(f"Warning: Could not convert value for row {date}: {str(ve)}")
                            continue
                
                if prediction_data["daily_predictions"]:
                    return prediction_data
                else:
                    raise ValueError("No valid daily predictions were found in the table.")
            else:
                raise ValueError("Could not find the daily prediction table.")
        else:
            raise ValueError("Could not find the daily predictions section.")
            
    except Exception as e:
        raise ValueError(f"Error occurred while scraping predictions: {str(e)}")

if __name__ == "__main__":
    # This code only runs when the script is executed directly
    coin_name = input("Enter the coin name: ").strip().lower()
    import asyncio
    prediction_data = asyncio.run(get_coin_predictions(coin_name))
    print("\nDaily Prediction Data:")
    print(json.dumps(prediction_data, indent=2))
