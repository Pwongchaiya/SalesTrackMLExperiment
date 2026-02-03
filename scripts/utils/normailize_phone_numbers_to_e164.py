from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum
from typing import Dict, List, Optional, Tuple
import pandas as pd

class TrunkRule(StrEnum):
    DROP_ZERO = "drop_zero"
    KEEP_ZERO = "keep_zero"
    FRANCE_10DIGIT = "france_10digit"
    NONE = "none"

@dataclass(frozen=True)
class CountryInfo:
    """Country-specific phone number information."""
    country_code: str
    trunk_rule: TrunkRule  # "drop_zero" | "keep_zero" | "france_10digit" | "none"
    national_destination_code: Dict[str, str]  # city_name -> area_code

# Consolidated country and city information: calling code + trunk rules + city area codes
COUNTRY_INFO: Dict[str, CountryInfo] = {
    "Australia": CountryInfo(
        country_code="61",
        trunk_rule=TrunkRule.DROP_ZERO,
        national_destination_code={
            "Chatswood": "2",
            "Glen Waverly": "3",
            "Melbourne": "3",
            "North Sydney": "2",
            "South Brisbane": "7",
        },
    ),
    "Austria": CountryInfo(
        country_code="43",
        trunk_rule=TrunkRule.DROP_ZERO,
        national_destination_code={
            "Graz": "316",
            "Salzburg": "662",
        },
    ),
    "Belgium": CountryInfo(
        country_code="32",
        trunk_rule=TrunkRule.DROP_ZERO,
        national_destination_code={
            "Bruxelles": "2",
            "Charleroi": "71",
        },
    ),
    "Canada": CountryInfo(
        country_code="1",
        trunk_rule=TrunkRule.NONE,
        national_destination_code={
            "Montreal": "514",
            "Tsawassen": "604",
            "Vancouver": "604",
        },
    ),
    "Denmark": CountryInfo(
        country_code="45",
        trunk_rule=TrunkRule.DROP_ZERO,
        national_destination_code={
            "Aaarhus": "86",
            "Kobenhavn": "35",
        },
    ),
    "Finland": CountryInfo(
        country_code="358",
        trunk_rule=TrunkRule.DROP_ZERO,
        national_destination_code={
            "Espoo": "9",
            "Helsinki": "9",
            "Oulu": "8",
        },
    ),
    "France": CountryInfo(
        country_code="33",
        trunk_rule=TrunkRule.FRANCE_10DIGIT,
        national_destination_code={
            "Lille": "3",
            "Lyon": "4",
            "Marseille": "4",
            "Nantes": "2",
            "Paris": "1",
            "Reims": "3",
            "Strasbourg": "3",
            "Toulouse": "5",
            "Versailles": "1",
        },
    ),
    "Germany": CountryInfo(
        country_code="49",
        trunk_rule=TrunkRule.DROP_ZERO,
        national_destination_code={
            "Frankfurt": "69",
            "Koln": "221",
            "Munich": "89",
        },
    ),
    "Ireland": CountryInfo(
        country_code="353",
        trunk_rule=TrunkRule.DROP_ZERO,
        national_destination_code={
            "Dublin": "1",
        },
    ),
    "Italy": CountryInfo(
        country_code="39",
        trunk_rule=TrunkRule.DROP_ZERO,
        national_destination_code={
            "Bergamo": "035",
            "Reggio Emilia": "0522",
            "Torino": "011",
        },
    ),
    "Japan": CountryInfo(
        country_code="81",
        trunk_rule=TrunkRule.DROP_ZERO,
        national_destination_code={
            "Minato-ku": "3",
            "Osaka": "6",
        },
    ),
    "Norway": CountryInfo(
        country_code="47",
        trunk_rule=TrunkRule.KEEP_ZERO,
        national_destination_code={
            "Bergen": "55",
            "Oslo": "22",
            "Stavern": "33",
        },
    ),
    "Philippines": CountryInfo(
        country_code="63",
        trunk_rule=TrunkRule.DROP_ZERO,
        national_destination_code={
            "Makati City": "2",
        },
    ),
    "Singapore": CountryInfo(
        country_code="65",
        trunk_rule=TrunkRule.KEEP_ZERO,
        national_destination_code={
            "Singapore": "",  # No area codes
        },
    ),
    "Spain": CountryInfo(
        country_code="34",
        trunk_rule=TrunkRule.DROP_ZERO,
        national_destination_code={
            "Barcelona": "93",
            "Madrid": "91",
            "Sevilla": "95",
        },
    ),
    "Sweden": CountryInfo(
        country_code="46",
        trunk_rule=TrunkRule.DROP_ZERO,
        national_destination_code={
            "Boras": "33",
            "Lule": "920",
        },
    ),
    "Switzerland": CountryInfo(
        country_code="41",
        trunk_rule=TrunkRule.DROP_ZERO,
        national_destination_code={
            "Gensve": "22",
        },
    ),
    "UK": CountryInfo(
        country_code="44",
        trunk_rule=TrunkRule.DROP_ZERO,
        national_destination_code={
            "Cowes": "1983",
            "Liverpool": "151",
            "London": "20",
            "Manchester": "161",
        },
    ),
    "USA": CountryInfo(
        country_code="1",
        trunk_rule=TrunkRule.NONE,
        national_destination_code={
            "Allentown": "610",
            "Boston": "617",
            "Brickhaven": "203",
            "Bridgewater": "508",
            "Brisbane": "415",
            "Burbank": "818",
            "Burlingame": "650",
            "Cambridge": "617",
            "Glendale": "818",
            "Las Vegas": "702",
            "Los Angeles": "213",
            "NYC": "212",
            "Nashua": "603",
            "New Bedford": "508",
            "New Haven": "203",
            "Newark": "973",
            "Pasadena": "626",
            "Philadelphia": "215",
            "San Diego": "619",
            "San Francisco": "415",
            "San Jose": "408",
            "San Rafael": "415",
            "White Plains": "914",
        },
    ),
}

# Derived: Country codes sorted by descending length order for international number parsing
KNOWN_COUNTRY_CODES_LONGEST_FIRST: List[str] = sorted(
    set(info.country_code for info in COUNTRY_INFO.values()),
    key=len,
    reverse=True,
)

def _strip_to_digits_with_optional_leading_plus(raw_phone: object) -> str:
    """Strip to digits only, preserving leading '+' if present."""
    if raw_phone is None or pd.isna(raw_phone):
        return ""

    raw_text = str(raw_phone).strip()

    if not raw_text or raw_text.lower() == "nan":
        return ""
    
    has_plus = raw_text[0] == "+"
    digits = "".join(string_number for string_number in raw_text if string_number.isdigit())
    
    return f"+{digits}" if has_plus and digits else digits

def _extract_country_code_from_international_number(
    cleaned_phone_with_plus: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Extract country code from international number starting with '+'. Returns (country_code, national_number)."""
    if len(cleaned_phone_with_plus) < 2:
        return None, None
    
    digits = cleaned_phone_with_plus[1:]

    # Try matching country codes
    for code in KNOWN_COUNTRY_CODES_LONGEST_FIRST:
        if digits.startswith(code):
            national = digits[len(code):]
            if not national:
                return None, None
            return code, national
        
    return None, None

def _apply_trunk_prefix_rules(
    domestic_digits: str,
    country_name: str,
) -> str:
    """Apply country-specific trunk prefix rules. Returns modified_digits."""
    if not domestic_digits:
        return domestic_digits

    # Get trunk rule from consolidated structure
    country_info = COUNTRY_INFO.get(country_name)
    if not country_info:
        return domestic_digits
    
    trunk_rule = country_info.trunk_rule
    
    if trunk_rule == TrunkRule.FRANCE_10DIGIT:
        if len(domestic_digits) == 10 and domestic_digits.startswith("0"):
            return domestic_digits[1:]
        return domestic_digits
    
    elif trunk_rule == TrunkRule.KEEP_ZERO:
        return domestic_digits
    
    elif trunk_rule == TrunkRule.DROP_ZERO:
        if domestic_digits.startswith("0"):
            return domestic_digits[1:]
        return domestic_digits
    
    return domestic_digits

def _validate_and_format_e164(
    country_code: str,
    national_number_digits: str,
) -> str:
    """Validate and format to E.164. Returns formatted phone or empty string if invalid."""
    # Validation checks
    if not country_code or not national_number_digits:
        return ""
    
    if not national_number_digits.isdigit():
        return ""
    
    # NANP special handling (country code 1)
    if country_code == "1":
        if len(national_number_digits) == 11 and national_number_digits.startswith("1"):
            national_number_digits = national_number_digits[1:]

        if len(national_number_digits) != 10:
            return ""

    # General length validation
    if len(national_number_digits) < 8:
        return ""

    return f"+{country_code}{national_number_digits}"

def normalize_phone_to_e164(raw_phone: object, country_name: object, city_name: object = None) -> str:
    """Normalize phone to E.164 format. Returns formatted phone or empty string if invalid."""
    cleaned_phone = _strip_to_digits_with_optional_leading_plus(raw_phone)
    if not cleaned_phone:
        return ""

    # Normalize inputs
    normalized_country = "" if country_name is None else str(country_name).strip()
    normalized_city = "" if city_name is None else str(city_name).strip()
    
    # Check if country is missing (but allow international format)
    if not normalized_country or normalized_country.lower() == "nan":
        if not cleaned_phone.startswith("+"):
            return ""

    # Handle international format (+...)
    if cleaned_phone.startswith("+"):
        country_code, national_number = _extract_country_code_from_international_number(cleaned_phone)
        if not country_code or not national_number:
            return ""
        return _validate_and_format_e164(country_code, national_number)

    # Handle domestic format
    if normalized_country not in COUNTRY_INFO:
        return ""

    country_info = COUNTRY_INFO[normalized_country]
    country_code = country_info.country_code
    domestic_digits = _apply_trunk_prefix_rules(cleaned_phone, normalized_country)
    
    # Prepend city area code if available and number appears local
    if normalized_city and len(domestic_digits) <= 8:
        area_code = country_info.national_destination_code.get(normalized_city)
        if area_code and not domestic_digits.startswith(area_code):
            domestic_digits = area_code + domestic_digits

    return _validate_and_format_e164(country_code, domestic_digits)