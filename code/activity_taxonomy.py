from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Iterable


ACTIVITY12_CATEGORIES = (
    "Food_Dining",
    "Nightlife",
    "Shopping_Service",
    "Work_Public",
    "Health_Care",
    "Education",
    "Residence",
    "Transport",
    "Lodging",
    "Arts_Culture_Entertainment",
    "Outdoor_Sport",
    "Religion",
)

ACTIVITY12_DESCRIPTIONS = {
    "Food_Dining": "venues primarily for meals or non-alcohol-focused drinks; excludes bars, pubs, clubs, and hotel accommodation",
    "Nightlife": "venues primarily for alcohol-focused or late-night socializing, such as bars, pubs, clubs, lounges, and breweries",
    "Shopping_Service": "retail and customer-facing commercial or personal services; excludes clinical health care and professional workplaces",
    "Work_Public": "offices, professional workplaces, government, civic administration, and public-service buildings; excludes schools, hospitals, and religious venues",
    "Health_Care": "clinical treatment and health care at hospitals, clinics, medical offices, emergency rooms, or pharmacies; excludes wellness and beauty services",
    "Education": "venues primarily for teaching, study, or academic administration, including schools, universities, classrooms, and libraries",
    "Residence": "long-term private or communal living places, including homes, apartments, housing, and residence halls; excludes temporary lodging",
    "Transport": "transport infrastructure or transfer points, including stations, airports, roads, bridges, parking, ferries, taxis, and travel terminals",
    "Lodging": "temporary overnight accommodation such as hotels, motels, resorts, and bed-and-breakfast venues",
    "Arts_Culture_Entertainment": "venues primarily for culture, performances, exhibitions, attractions, games, cinema, conventions, or spectator entertainment",
    "Outdoor_Sport": "indoor or outdoor recreation, exercise, sports, parks, nature, beaches, trails, gyms, courts, pools, and stadiums",
    "Religion": "venues primarily for worship, spiritual practice, funerals, burial, or memorial observance",
}


_GROUP_TO_FINE = {
    "Arts_Culture_Entertainment": frozenset({
        "Aquarium", "Arcade", "Art Gallery", "Art Museum",
        "Arts & Entertainment", "Casino", "College Theater", "Comedy Club",
        "Concert Hall", "Convention Center", "Dance Studio", "Event Space",
        "Fair", "General Entertainment", "Historic Site", "History Museum",
        "Jazz Club", "Monument / Landmark", "Movie Theater", "Multiplex",
        "Museum", "Music Venue", "Performing Arts Venue", "Public Art",
        "Rock Club", "Science Museum", "Sculpture Garden", "Theater",
        "Theme Park", "Zoo",
    }),
    "Education": frozenset({
        "College & University", "College Academic Building",
        "College Administrative Building", "College Arts Building",
        "College Auditorium", "College Classroom",
        "College Communications Building", "College Engineering Building",
        "College Lab", "College Library", "College Science Building",
        "College Technology Building", "Community College",
        "Elementary School", "General College & University", "High School",
        "Law School", "Library", "Medical School", "Middle School",
        "Music School", "Nursery School", "School", "Student Center",
        "Trade School", "University",
    }),
    "Food_Dining": frozenset({
        "African Restaurant", "Airport Food Court", "American Restaurant",
        "Arepa Restaurant", "Asian Restaurant", "Australian Restaurant",
        "BBQ Joint", "Bagel Shop", "Bakery", "Brazilian Restaurant",
        "Breakfast Spot", "Burger Joint", "Burrito Place",
        "Cafeteria", "Café", "Cajun / Creole Restaurant",
        "Caribbean Restaurant", "Chinese Restaurant", "Coffee Shop",
        "College Cafeteria", "Cuban Restaurant", "Cupcake Shop",
        "Deli / Bodega", "Dessert Shop", "Dim Sum Restaurant", "Diner",
        "Donut Shop", "Dumpling Restaurant", "Eastern European Restaurant",
        "Falafel Restaurant", "Fast Food Restaurant", "Filipino Restaurant",
        "Food", "Food Court", "Food Truck", "French Restaurant",
        "Fried Chicken Joint", "Frozen Yogurt", "German Restaurant",
        "Hot Dog Joint", "Ice Cream Shop", "Indian Restaurant",
        "Indonesian Restaurant", "Italian Restaurant", "Japanese Restaurant",
        "Juice Bar", "Korean Restaurant", "Latin American Restaurant",
        "Mac & Cheese Joint", "Malaysian Restaurant",
        "Mediterranean Restaurant", "Mexican Restaurant",
        "Middle Eastern Restaurant", "Molecular Gastronomy Restaurant",
        "Pizza Place", "Ramen /  Noodle House", "Restaurant", "Salad Place",
        "Sandwich Place", "Scandinavian Restaurant", "Seafood Restaurant",
        "Snack Place", "Soup Place", "South American Restaurant",
        "Southern / Soul Food Restaurant", "Spanish Restaurant", "Steakhouse",
        "Sushi Restaurant", "Swiss Restaurant", "Taco Place",
        "Tapas Restaurant", "Tea Room", "Thai Restaurant",
        "Vegetarian / Vegan Restaurant", "Vietnamese Restaurant", "Wings Joint",
    }),
    "Health_Care": frozenset({
        "Dentist's Office", "Doctor's Office", "Drugstore / Pharmacy",
        "Emergency Room", "Hospital", "Medical Center",
    }),
    "Lodging": frozenset({
        "Bed & Breakfast", "Hotel", "Motel", "Resort",
    }),
    "Nightlife": frozenset({
        "Bar", "Beer Garden", "Brewery", "Cocktail Bar", "Gastropub",
        "Gay Bar", "Hotel Bar", "Karaoke Bar", "Lounge", "Nightclub",
        "Other Nightlife", "Pub", "Sake Bar", "Sports Bar", "Strip Club",
        "Whisky Bar", "Wine Bar",
    }),
    "Outdoor_Sport": frozenset({
        "Athletic & Sport", "Baseball Stadium", "Basketball Court",
        "Basketball Stadium", "Beach", "Bowling Alley", "Campground",
        "College Basketball Court", "College Gym", "College Quad",
        "College Stadium", "Farm", "Field", "Garden", "Gym",
        "Gym / Fitness Center", "Gym Pool", "Hiking Trail", "Island", "Lake",
        "Lighthouse", "Martial Arts Dojo", "Mountain", "Other Great Outdoors",
        "Outdoors & Recreation", "Park", "Playground", "Plaza", "Pool",
        "Pool Hall", "Racetrack", "River", "Scenic Lookout", "Skate Park",
        "Skating Rink", "Soccer Field", "Soccer Stadium", "Stadium",
        "Roof Deck", "Surf Spot", "Tennis Court", "Track", "Trail", "Water Park",
        "Yoga Studio",
    }),
    "Religion": frozenset({
        "Cemetery", "Church", "Funeral Home", "Mosque", "Shrine",
        "Spiritual Center", "Synagogue", "Temple",
    }),
    "Residence": frozenset({
        "Assisted Living", "College Residence Hall", "Home (private)",
        "Housing Development", "Neighborhood",
        "Residential Building (Apartment / Condo)",
    }),
    "Shopping_Service": frozenset({
        "Accessories Store", "Antique Shop", "Arts & Crafts Store",
        "Automotive Shop", "Bank", "Bike Shop", "Board Shop", "Bookstore",
        "Boutique", "Bridal Shop", "Butcher", "Camera Store", "Candy Store",
        "Clothing Store", "Convenience Store", "Cosmetics Shop",
        "Department Store", "Electronics Store",
        "Fish Market", "Flea Market",
        "Food & Drink Shop", "Furniture / Home Store", "Gaming Cafe",
        "Garden Center", "Gift Shop", "Grocery Store", "Hardware Store",
        "Hobby Shop", "Internet Cafe", "Jewelry Store", "Laundry Service",
        "Mall", "Market", "Men's Store", "Miscellaneous Shop",
        "Mobile Phone Shop", "Music Store", "Newsstand",
        "Paper / Office Supplies Store", "Pet Store", "Record Shop",
        "Recycling Facility", "Salon / Barbershop", "Shoe Store", "Smoke Shop",
        "Spa / Massage",
        "Sporting Goods Shop", "Tanning Salon", "Tattoo Parlor",
        "Thrift / Vintage Store", "Toy / Game Store", "Video Game Store",
        "Video Store", "Women's Store",
    }),
    "Transport": frozenset({
        "Airport", "Airport Terminal", "Bridge", "Bus Line", "Bus Station",
        "Ferry", "Gas Station / Garage", "General Travel", "Harbor / Marina",
        "Light Rail", "Moving Target", "Parking", "Pier", "Platform",
        "Rest Area", "Road", "Subway", "Taxi", "Train", "Train Station",
        "Travel & Transport", "Travel Lounge",
    }),
    "Work_Public": frozenset({
        "Animal Shelter", "Building", "City", "City Hall", "Courthouse",
        "Coworking Space", "Design Studio", "Factory",
        "Financial or Legal Service", "Fire Station", "Government Building",
        "Military Base", "Office", "Police Station", "Post Office",
        "Professional & Other Places", "Tech Startup",
    }),
}


def _build_mapping() -> dict[str, str]:
    mapping: dict[str, str] = {}
    duplicates = []
    for activity, fine_categories in _GROUP_TO_FINE.items():
        if activity not in ACTIVITY12_CATEGORIES:
            raise RuntimeError(f"unknown activity category in mapping: {activity}")
        for fine_category in fine_categories:
            if fine_category in mapping:
                duplicates.append(fine_category)
            mapping[fine_category] = activity
    if duplicates:
        raise RuntimeError(f"fine categories mapped more than once: {sorted(duplicates)}")
    return mapping


FINE_TO_ACTIVITY12 = _build_mapping()

FINE_CATEGORY_ALIASES = {
    "Cafe": "Café",
    "Metro Station": "Subway",
    "Light Rail Station": "Light Rail",
    "Boat or Ferry": "Ferry",
    "Ramen / Noodle House": "Ramen /  Noodle House",
    "Athletics & Sports": "Athletic & Sport",
}


def normalize_fine_category(category: object) -> str:
    value = str(category).strip()
    return FINE_CATEGORY_ALIASES.get(value, value)


def fine_to_activity12(category: object) -> str:
    normalized = normalize_fine_category(category)
    try:
        return FINE_TO_ACTIVITY12[normalized]
    except KeyError as exc:
        raise KeyError(
            f"fine POI category {category!r} has no Activity12 mapping"
        ) from exc


def validate_activity12_coverage(categories: Iterable[object]) -> dict:
    normalized = [normalize_fine_category(category) for category in categories]
    unknown = sorted({category for category in normalized
                      if category not in FINE_TO_ACTIVITY12})
    if unknown:
        raise ValueError(
            f"activity12 mapping does not cover {len(unknown)} categories: {unknown}"
        )
    counts = Counter(FINE_TO_ACTIVITY12[category] for category in normalized)
    return {
        "fine_categories": len(set(normalized)),
        "activity_categories": len(counts),
        "activity_fine_category_counts": {
            activity: int(counts.get(activity, 0))
            for activity in ACTIVITY12_CATEGORIES
        },
        "mapping_sha256": activity12_mapping_sha256(),
    }


def activity12_mapping_sha256() -> str:
    payload = json.dumps(
        {
            "categories": ACTIVITY12_CATEGORIES,
            "descriptions": ACTIVITY12_DESCRIPTIONS,
            "mapping": FINE_TO_ACTIVITY12,
        },
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
