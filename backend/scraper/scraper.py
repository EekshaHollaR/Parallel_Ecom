import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import quote, urljoin
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .cache import RedisCache

logger = logging.getLogger(__name__)

@dataclass
class ScrapedProduct:
    """Structured data for scraped product information"""
    site: str
    price: float
    product_name: str
    product_url: str
    timestamp: float
    currency: str = "USD"
    available: bool = True
    rating: Optional[float] = None
    review_count: Optional[int] = None

class BaseSiteScraper(ABC):
    """Base class for site-specific scrapers"""
    
    def __init__(self, site_name: str, base_url: str):
        self.site_name = site_name
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })
    
    @abstractmethod
    def search_url(self, product_name: str) -> str:
        """Generate search URL for product"""
        pass
    
    @abstractmethod
    def parse_search_results(self, html: str, product_name: str) -> List[ScrapedProduct]:
        """Parse search results page and extract product data"""
        pass
    
    def scrape_product(self, product_name: str) -> List[ScrapedProduct]:
        """Main scraping method for a product"""
        try:
            search_url = self.search_url(product_name)
            logger.debug(f"Scraping {self.site_name} for: {product_name}")
            
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            products = self.parse_search_results(response.text, product_name)
            logger.info(f"Found {len(products)} products on {self.site_name} for '{product_name}'")
            
            return products
            
        except requests.RequestException as e:
            logger.error(f"Request failed for {self.site_name} - {product_name}: {e}")
            return []
        except Exception as e:
            logger.error(f"Parsing failed for {self.site_name} - {product_name}: {e}")
            return []

class AmazonScraper(BaseSiteScraper):
    """Mock Amazon scraper"""
    
    def __init__(self):
        super().__init__("amazon", "https://www.amazon.com")
    
    def search_url(self, product_name: str) -> str:
        encoded_name = quote(product_name)
        return f"https://www.amazon.com/s?k={encoded_name}"
    
    def parse_search_results(self, html: str, product_name: str) -> List[ScrapedProduct]:
        # Mock implementation - in real scenario, parse actual HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Generate mock products
        products = []
        for i in range(random.randint(1, 3)):
            price = round(random.uniform(10.0, 500.0), 2)
            product_id = f"AMZ{random.randint(1000, 9999)}"
            
            products.append(ScrapedProduct(
                site=self.site_name,
                price=price,
                product_name=f"{product_name} - {product_id}",
                product_url=f"https://www.amazon.com/dp/{product_id}",
                timestamp=time.time(),
                rating=round(random.uniform(3.5, 5.0), 1),
                review_count=random.randint(10, 1000)
            ))
        
        return products

class EbayScraper(BaseSiteScraper):
    """Mock eBay scraper"""
    
    def __init__(self):
        super().__init__("ebay", "https://www.ebay.com")
    
    def search_url(self, product_name: str) -> str:
        encoded_name = quote(product_name)
        return f"https://www.ebay.com/sch/i.html?_nkw={encoded_name}"
    
    def parse_search_results(self, html: str, product_name: str) -> List[ScrapedProduct]:
        # Mock implementation
        products = []
        for i in range(random.randint(1, 4)):
            price = round(random.uniform(5.0, 300.0), 2)
            product_id = f"EBAY{random.randint(1000, 9999)}"
            
            products.append(ScrapedProduct(
                site=self.site_name,
                price=price,
                product_name=f"{product_name} - Auction {product_id}",
                product_url=f"https://www.ebay.com/itm/{product_id}",
                timestamp=time.time(),
                rating=round(random.uniform(3.0, 5.0), 1),
                review_count=random.randint(5, 500)
            ))
        
        return products

class WalmartScraper(BaseSiteScraper):
    """Mock Walmart scraper"""
    
    def __init__(self):
        super().__init__("walmart", "https://www.walmart.com")
    
    def search_url(self, product_name: str) -> str:
        encoded_name = quote(product_name)
        return f"https://www.walmart.com/search/?query={encoded_name}"
    
    def parse_search_results(self, html: str, product_name: str) -> List[ScrapedProduct]:
        products = []
        for i in range(random.randint(1, 3)):
            price = round(random.uniform(8.0, 200.0), 2)
            product_id = f"WMT{random.randint(1000, 9999)}"
            
            products.append(ScrapedProduct(
                site=self.site_name,
                price=price,
                product_name=f"{product_name} - Walmart {product_id}",
                product_url=f"https://www.walmart.com/ip/{product_id}",
                timestamp=time.time(),
                rating=round(random.uniform(3.2, 4.8), 1),
                review_count=random.randint(20, 800)
            ))
        
        return products

class BestBuyScraper(BaseSiteScraper):
    """Mock BestBuy scraper"""
    
    def __init__(self):
        super().__init__("bestbuy", "https://www.bestbuy.com")
    
    def search_url(self, product_name: str) -> str:
        encoded_name = quote(product_name)
        return f"https://www.bestbuy.com/site/searchpage.jsp?st={encoded_name}"
    
    def parse_search_results(self, html: str, product_name: str) -> List[ScrapedProduct]:
        products = []
        for i in range(random.randint(1, 2)):
            price = round(random.uniform(50.0, 1000.0), 2)
            product_id = f"BBY{random.randint(1000, 9999)}"
            
            products.append(ScrapedProduct(
                site=self.site_name,
                price=price,
                product_name=f"{product_name} - BestBuy {product_id}",
                product_url=f"https://www.bestbuy.com/site/{product_id}.p",
                timestamp=time.time(),
                rating=round(random.uniform(3.7, 4.9), 1),
                review_count=random.randint(15, 600)
            ))
        
        return products

class TargetScraper(BaseSiteScraper):
    """Mock Target scraper"""
    
    def __init__(self):
        super().__init__("target", "https://www.target.com")
    
    def search_url(self, product_name: str) -> str:
        encoded_name = quote(product_name)
        return f"https://www.target.com/s?searchTerm={encoded_name}"
    
    def parse_search_results(self, html: str, product_name: str) -> List[ScrapedProduct]:
        products = []
        for i in range(random.randint(1, 3)):
            price = round(random.uniform(12.0, 250.0), 2)
            product_id = f"TGT{random.randint(1000, 9999)}"
            
            products.append(ScrapedProduct(
                site=self.site_name,
                price=price,
                product_name=f"{product_name} - Target {product_id}",
                product_url=f"https://www.target.com/p/{product_id}",
                timestamp=time.time(),
                rating=round(random.uniform(3.5, 4.7), 1),
                review_count=random.randint(25, 400)
            ))
        
        return products

class NeweggScraper(BaseSiteScraper):
    """Mock Newegg scraper (electronics focus)"""
    
    def __init__(self):
        super().__init__("newegg", "https://www.newegg.com")
    
    def search_url(self, product_name: str) -> str:
        encoded_name = quote(product_name)
        return f"https://www.newegg.com/p/pl?d={encoded_name}"
    
    def parse_search_results(self, html: str, product_name: str) -> List[ScrapedProduct]:
        products = []
        for i in range(random.randint(1, 4)):
            price = round(random.uniform(30.0, 800.0), 2)
            product_id = f"NWG{random.randint(1000, 9999)}"
            
            products.append(ScrapedProduct(
                site=self.site_name,
                price=price,
                product_name=f"{product_name} - Newegg {product_id}",
                product_url=f"https://www.newegg.com/p/{product_id}",
                timestamp=time.time(),
                rating=round(random.uniform(3.8, 5.0), 1),
                review_count=random.randint(8, 300)
            ))
        
        return products

class PriceScraper:
    """
    Main price scraper class with ThreadPoolExecutor for concurrent scraping
    """
    
    def __init__(self, cache: RedisCache = None, max_workers: int = 8, request_delay: float = 0.5):
        self.cache = cache or RedisCache()
        self.max_workers = max_workers
        self.request_delay = request_delay
        
        # Initialize site scrapers
        self.scrapers = [
            AmazonScraper(),
            EbayScraper(),
            WalmartScraper(),
            BestBuyScraper(),
            TargetScraper(),
            NeweggScraper()
        ]
        
        logger.info(f"Initialized PriceScraper with {len(self.scrapers)} sites and {max_workers} workers")
    
    def scrape_single_product(self, product_name: str, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Scrape prices for a single product across all sites
        """
        cache_key = f"scrape:{product_name.lower()}"
        
        # Check cache first
        if use_cache:
            cached_results = self.cache.get(cache_key)
            if cached_results:
                logger.info(f"Cache hit for product: {product_name}")
                return cached_results
        
        # Scrape from all sites concurrently
        all_products = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=min(len(self.scrapers), self.max_workers)) as executor:
            # Submit all scraping tasks
            future_to_scraper = {
                executor.submit(scraper.scrape_product, product_name): scraper 
                for scraper in self.scrapers
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_scraper):
                scraper = future_to_scraper[future]
                try:
                    products = future.result()
                    all_products.extend(products)
                    logger.debug(f"Completed scraping {scraper.site_name}")
                except Exception as e:
                    logger.error(f"Scraper {scraper.site_name} failed: {e}")
                
                # Small delay to be respectful
                time.sleep(self.request_delay)
        
        # Convert to serializable format
        results = [self._product_to_dict(product) for product in all_products]
        
        # Cache the results
        if results and use_cache:
            self.cache.set(cache_key, results, expire_seconds=600)  # 10 minutes
        
        elapsed_time = time.time() - start_time
        logger.info(f"Scraped {len(results)} results for '{product_name}' in {elapsed_time:.2f}s")
        
        return results
    
    def scrape_multiple_products(self, product_names: List[str], use_cache: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scrape prices for multiple products concurrently
        """
        results = {}
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all product scraping tasks
            future_to_product = {
                executor.submit(self.scrape_single_product, product_name, use_cache): product_name 
                for product_name in product_names
            }
            
            # Collect results
            for future in as_completed(future_to_product):
                product_name = future_to_product[future]
                try:
                    product_results = future.result()
                    results[product_name] = product_results
                    logger.debug(f"Completed scraping product: {product_name}")
                except Exception as e:
                    logger.error(f"Failed to scrape product {product_name}: {e}")
                    results[product_name] = []
        
        elapsed_time = time.time() - start_time
        logger.info(f"Scraped {len(product_names)} products in {elapsed_time:.2f}s")
        
        return results
    
    def get_scraping_stats(self) -> Dict[str, Any]:
        """Get statistics about available scrapers"""
        return {
            "total_sites": len(self.scrapers),
            "site_names": [scraper.site_name for scraper in self.scrapers],
            "max_workers": self.max_workers,
            "cache_enabled": self.cache.is_connected()
        }
    
    def add_scraper(self, scraper: BaseSiteScraper):
        """Add a new site scraper"""
        self.scrapers.append(scraper)
        logger.info(f"Added new scraper for: {scraper.site_name}")
    
    def remove_scraper(self, site_name: str):
        """Remove a site scraper"""
        self.scrapers = [s for s in self.scrapers if s.site_name != site_name]
        logger.info(f"Removed scraper for: {site_name}")
    
    def _product_to_dict(self, product: ScrapedProduct) -> Dict[str, Any]:
        """Convert ScrapedProduct to serializable dictionary"""
        return {
            "site": product.site,
            "price": product.price,
            "product_name": product.product_name,
            "product_url": product.product_url,
            "timestamp": product.timestamp,
            "currency": product.currency,
            "available": product.available,
            "rating": product.rating,
            "review_count": product.review_count
        }


# API-ready factory functions
def create_price_scraper(redis_host: str = 'localhost', redis_port: int = 6379, 
                        max_workers: int = 8) -> PriceScraper:
    """Create a PriceScraper instance with Redis cache"""
    cache = RedisCache(host=redis_host, port=redis_port, expire_seconds=600)
    return PriceScraper(cache=cache, max_workers=max_workers)

def create_scraper_without_cache(max_workers: int = 8) -> PriceScraper:
    """Create a PriceScraper instance without caching"""
    return PriceScraper(cache=None, max_workers=max_workers)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create scraper
    scraper = create_price_scraper(max_workers=6)
    
    # Test single product
    results = scraper.scrape_single_product("iPhone 15")
    print(f"Found {len(results)} results")
    
    # Test multiple products
    products = ["laptop", "headphones", "monitor"]
    all_results = scraper.scrape_multiple_products(products)
    
    for product, results in all_results.items():
        print(f"{product}: {len(results)} results")
    
    # Print stats
    stats = scraper.get_scraping_stats()
    print("Scraper stats:", stats)