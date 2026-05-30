from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
import requests
from datetime import datetime

class DatasetSearchResult:
    def __init__(self, title: str, url: str, size: str, last_updated: str, vote_count: int, description: str, source: str):
        self.title = title
        self.url = url
        self.size = size
        self.last_updated = last_updated
        self.vote_count = vote_count
        self.description = description
        self.source = source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "size": self.size,
            "last_updated": self.last_updated,
            "vote_count": self.vote_count,
            "description": self.description,
            "source": self.source
        }

class BaseDatasetProvider(ABC):
    @abstractmethod
    def search(self, query: str, limit: int = 20, page: int = 1) -> List[Dict[str, Any]]:
        pass

class KaggleDatasetProvider(BaseDatasetProvider):
    def __init__(self, api_token: Optional[str] = None):
        # Allow passing token or reading from environment variable
        self.api_token = api_token or os.getenv("KAGGLE_API_TOKEN")
        self.base_url = "https://www.kaggle.com/api/v1"

    def format_size(self, size_bytes: Optional[int]) -> str:
        if size_bytes is None:
            return "Unknown Size"
        try:
            size_float = float(size_bytes)
        except (ValueError, TypeError):
            return "Unknown Size"
            
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_float < 1024.0:
                return f"{size_float:.1f} {unit}"
            size_float /= 1024.0
        return f"{size_float:.1f} PB"

    def format_date(self, date_str: Optional[str]) -> str:
        if not date_str:
            return "Unknown"
        try:
            # Parse ISO-8601 string, e.g. "2022-06-06T19:39:40.923Z"
            # Standard fromisoformat doesn't support 'Z' on older Python versions, so replace with offset
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.strftime("%B %d, %Y")
        except Exception:
            # Fallback to date portion if iso format parse fails
            if "T" in date_str:
                return date_str.split("T")[0]
            return date_str

    def search(self, query: str, limit: int = 20, page: int = 1) -> List[Dict[str, Any]]:
        if not self.api_token:
            raise ValueError("Kaggle API Token is not configured. Please set KAGGLE_API_TOKEN in your environment.")

        url = f"{self.base_url}/datasets/list"
        params = {
            "search": query,
            "page": page,
            "sortBy": "hottest"  # Default sort option (can be customized in the future)
        }
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "User-Agent": "ResearchMateAI"
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            if response.status_code == 401:
                raise ValueError("Unauthorized: Kaggle API token is invalid or expired.")
            elif response.status_code != 200:
                raise Exception(f"Kaggle API returned status code {response.status_code}: {response.text}")

            datasets = response.json()
            results = []
            for ds in datasets:
                title = ds.get("title") or ds.get("titleNullable") or "Untitled Dataset"
                url_path = ds.get("url") or ds.get("urlNullable")
                if not url_path and ds.get("ref"):
                    url_path = f"https://www.kaggle.com/datasets/{ds.get('ref')}"
                
                size_bytes = ds.get("totalBytes") or ds.get("totalBytesNullable")
                size_str = self.format_size(size_bytes)
                
                last_updated = ds.get("lastUpdated")
                date_str = self.format_date(last_updated)
                
                vote_count = ds.get("voteCount") or 0
                
                # Extract description or subtitle beautifully
                description = (
                    ds.get("subtitle") 
                    or ds.get("subtitleNullable") 
                    or ds.get("description") 
                    or ds.get("descriptionNullable") 
                    or "No description available."
                )

                results.append(
                    DatasetSearchResult(
                        title=title,
                        url=url_path,
                        size=size_str,
                        last_updated=date_str,
                        vote_count=int(vote_count),
                        description=description,
                        source="Kaggle"
                    ).to_dict()
                )
            return results
        except requests.RequestException as e:
            raise Exception(f"Failed to reach Kaggle API: {str(e)}")

class UCIDatasetProvider(BaseDatasetProvider):
    def __init__(self):
        # Local high-quality database of the most popular UCI datasets for machine learning
        self.local_datasets = [
            {
                "title": "UCI Heart Disease Dataset",
                "url": "https://archive.ics.uci.edu/dataset/45/heart+disease",
                "size": "12.1 KB",
                "last_updated": "July 25, 1988",
                "vote_count": 412,
                "description": "Medical data for heart disease prediction with 14 attributes and 303 instances. Crucial for binary classification benchmarks.",
                "source": "UCI"
            },
            {
                "title": "UCI Iris Dataset",
                "url": "https://archive.ics.uci.edu/dataset/53/iris",
                "size": "4.6 KB",
                "last_updated": "July 01, 1988",
                "vote_count": 892,
                "description": "The classic pattern recognition database. Contains 3 classes of 50 instances each, where each class refers to a type of iris plant.",
                "source": "UCI"
            },
            {
                "title": "UCI Wine Quality Dataset",
                "url": "https://archive.ics.uci.edu/dataset/186/wine+quality",
                "size": "264 KB",
                "last_updated": "October 07, 2009",
                "vote_count": 318,
                "description": "Two datasets containing wine properties (red and white) from the north of Portugal. Goal is to model wine quality based on physicochemical tests.",
                "source": "UCI"
            },
            {
                "title": "UCI Breast Cancer Wisconsin (Diagnostic) Dataset",
                "url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
                "size": "124 KB",
                "last_updated": "November 01, 1995",
                "vote_count": 254,
                "description": "Features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. 30 features with 569 instances.",
                "source": "UCI"
            },
            {
                "title": "UCI Adult Census Income Dataset",
                "url": "https://archive.ics.uci.edu/dataset/2/adult",
                "size": "3.8 MB",
                "last_updated": "May 01, 1996",
                "vote_count": 605,
                "description": "Predict whether income exceeds $50K/yr based on census data. Also known as \"Census Income\" dataset. 14 features and 48,842 instances.",
                "source": "UCI"
            },
            {
                "title": "UCI Student Performance Dataset",
                "url": "https://archive.ics.uci.edu/dataset/320/student+performance",
                "size": "204 KB",
                "last_updated": "November 27, 2014",
                "vote_count": 180,
                "description": "Predict student performance (grade) in secondary education (high school) using demographic, social and school-related features.",
                "source": "UCI"
            },
            {
                "title": "UCI Mushroom Classification Dataset",
                "url": "https://archive.ics.uci.edu/dataset/73/mushroom",
                "size": "374 KB",
                "last_updated": "April 27, 1987",
                "vote_count": 290,
                "description": "Gilled mushrooms described in terms of physical characteristics. Classify as edible or poisonous. 8,124 instances.",
                "source": "UCI"
            },
            {
                "title": "UCI Abalone Dataset",
                "url": "https://archive.ics.uci.edu/dataset/1/abalone",
                "size": "191 KB",
                "last_updated": "December 01, 1995",
                "vote_count": 145,
                "description": "Predict the age of abalone from physical measurements. Includes 9 attributes and 4,177 instances.",
                "source": "UCI"
            },
            {
                "title": "UCI Dry Bean Dataset",
                "url": "https://archive.ics.uci.edu/dataset/602/dry+bean+dataset",
                "size": "4.7 MB",
                "last_updated": "July 18, 2020",
                "vote_count": 340,
                "description": "Images of 13,611 grains of 7 different registered dry beans were taken with a high-resolution camera for classification.",
                "source": "UCI"
            },
            {
                "title": "UCI Ames Housing Dataset",
                "url": "https://archive.ics.uci.edu/dataset/120/ames+housing",
                "size": "1.2 MB",
                "last_updated": "May 15, 2010",
                "vote_count": 210,
                "description": "A dataset containing Ames housing data for regression models with 79 explanatory variables describing residential homes.",
                "source": "UCI"
            }
        ]

    def search(self, query: str, limit: int = 20, page: int = 1) -> List[Dict[str, Any]]:
        clean_query = query.strip().lower()
        if clean_query in ["", "ai", "machine learning", "ml", "all"]:
            results = self.local_datasets
        else:
            results = [
                ds for ds in self.local_datasets
                if clean_query in ds["title"].lower() 
                or clean_query in ds["description"].lower()
            ]
        
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        return results[start_idx:end_idx]

class DatasetService:
    def __init__(self):
        self.providers = {
            "kaggle": KaggleDatasetProvider(),
            "uci": UCIDatasetProvider()
        }

    def search_all(self, query: str, provider: str = "kaggle", page: int = 1, limit: int = 20) -> List[Dict[str, Any]]:
        provider_key = provider.lower()
        if provider_key == "all":
            # Search both providers and interleave them dynamically
            kaggle_results = []
            try:
                kaggle_results = self.providers.get("kaggle").search(query, limit=limit, page=page)
            except Exception as e:
                print(f"[DatasetService] Kaggle search failed in 'all': {e}")
            
            uci_results = []
            try:
                uci_results = self.providers.get("uci").search(query, limit=limit, page=page)
            except Exception as e:
                print(f"[DatasetService] UCI search failed in 'all': {e}")
            
            combined = []
            max_len = max(len(kaggle_results), len(uci_results))
            for i in range(max_len):
                if i < len(kaggle_results):
                    combined.append(kaggle_results[i])
                if i < len(uci_results):
                    combined.append(uci_results[i])
            return combined
        
        prov = self.providers.get(provider_key)
        if not prov:
            raise ValueError(f"Unsupported dataset provider: {provider}")
        return prov.search(query, limit=limit, page=page)
