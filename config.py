
from __future__ import annotations

import os
from pathlib import Path
from typing import Final
from dotenv import load_dotenv


load_dotenv()



BASE_DIR: Final[Path] = Path(__file__).parent
DATA_DIR: Final[Path] = BASE_DIR / "data" / "pdfs"          
CHROMA_DIR: Final[Path] = BASE_DIR / "chroma_db"            
LOGS_DIR: Final[Path] = BASE_DIR / "logs"


for _d in (DATA_DIR, CHROMA_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)



OPENAI_API_KEY: Final[str] = os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_MODEL: Final[str] = "text-embedding-3-small"
EMBEDDING_DIMENSIONS: Final[int] = 1536          



CHROMA_COLLECTION_NAME: Final[str] = "pak_banking_car_finance_v1"



PDF_STRATEGY: Final[str] = os.environ.get("PDF_STRATEGY", "hi_res")


ELEMENT_TYPES_STRUCTURAL: Final[tuple[str, ...]] = ("Title", "Header")
ELEMENT_TYPES_CONTENT: Final[tuple[str, ...]] = (
    "NarrativeText",
    "ListItem",
    "Text",
    "FigureCaption",
    "Footer",
)
ELEMENT_TYPE_TABLE: Final[str] = "Table"


MAX_CHUNK_CHARS: Final[int] = 1_200

CHUNK_OVERLAP_CHARS: Final[int] = 150

MIN_CHUNK_CHARS: Final[int] = 40


KNOWN_BANKS: Final[dict[str, str]] = {
    
    "Meezan Bank": "meezan",
    "Bank Alfalah": "alfalah",
    "HBL": "hbl|habib bank",
    "MCB Bank": "mcb|muslim commercial",
    "UBL": "ubl|united bank",
    "Allied Bank": "allied bank|abl",
    "Faysal Bank": "faysal",
    "Bank Al Habib": "al habib|bahl",
    "Dubai Islamic Bank": "dubai islamic|dib",
    "Standard Chartered": "standard chartered|scb",
    "Askari Bank": "askari",
    "Soneri Bank": "soneri",
    "Bank of Punjab": "bank of punjab|bop",
    "Habib Metropolitan Bank": "habib metro|hmb",
    "JS Bank": "js bank",
    "Silk Bank": "silk bank",
    "Summit Bank": "summit bank",
    "NRSP Microfinance Bank": "nrsp",
    "First Women Bank": "first women",
    "Zarai Taraqiati Bank": "ztbl|zarai taraqiati",
    "National Bank of Pakistan": "national bank|nbp",
    "Sindh Bank": "sindh bank",
}



ISLAMIC_KEYWORDS: Final[tuple[str, ...]] = (
    "ijarah",
    "diminishing musharakah",
    "musharakah",
    "murabaha",
    "shariah",
    "sharia",
    "islamic",
    "takaful",
    "halal",
    "riba",
)

CONVENTIONAL_KEYWORDS: Final[tuple[str, ...]] = (
    "kibor",
    "interest rate",
    "conventional",
    "markup rate",
    "insurance",
)


METADATA_SCHEMA: Final[dict[str, type]] = {
    
    "source_file":         str,   
    "page_number":         int,  
    
    "bank_name":           str,   
    "bank_short":          str,   
    
    "element_type":        str,   
    "document_type":       str,   
    "section_header":      str,  
    
    "financing_type":      str,   
    "customer_segment":    str,   
             
    
    "has_rate_info":       bool,  
    "has_tenure_info":     bool,  
    "has_down_payment":    bool,  
    "has_limit_info":      bool,  
    
    "vehicle_type":        str,   
    
    "chunk_index":         int,   
    "total_chunks":        int,   
}
