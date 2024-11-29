NON_BLOCK_LAND_USE = (
    'outside',
    'feasible',
    'road',
    'boundary'
)

BLOCK_LAND_USE = (
    'residential',
    'business',
    'wastemgmt',
    'green_l',
    'green_s',
    'school',
    'hospital_l',
    'hospital_s',
    'recreation',
    'office',
    'waterbody',
    'drainage'
)
DENSITY = {
    1:50,
    4: 200,  # Residential
    5: 75,   # Business
    6: 50,   # Office
    7: 50,    # GREEN_L (Large Green Areas)
    8: 20,    # GREEN_S (Small Green Areas)
    9: 100,   # School
    10: 100,  # HOSPITAL_L (Large Hospitals)
    11: 20,  # HOSPITAL_S (Small Hospitals)
    12: 5 ,   # Recreation
    14: 20,
    13: 150

}

LAND_USE = (NON_BLOCK_LAND_USE + BLOCK_LAND_USE)

OUTSIDE = 0
FEASIBLE = 1
ROAD = 2
BOUNDARY = 3
RESIDENTIAL = 4
BUSINESS = 5
WASTEMGMT = 6
GREEN_L = 7
GREEN_S = 8
SCHOOL = 9
HOSPITAL_L = 10
HOSPITAL_S = 11
RECREATION = 12
OFFICE = 13
INTERSECTION = 15
WATERBODY =14
DRAINAGE = 16
LAND_USE_ID = (
    OUTSIDE,
    FEASIBLE,
    ROAD,
    BOUNDARY,
    RESIDENTIAL,
    BUSINESS,
    WASTEMGMT,
    GREEN_L,
    GREEN_S,
    SCHOOL,
    HOSPITAL_L,
    HOSPITAL_S,
    RECREATION,
    OFFICE,
    WATERBODY,
    DRAINAGE
)

NUM_TYPES = len(LAND_USE_ID)+1

LAND_USE_ID_MAP = dict(
    zip(LAND_USE, LAND_USE_ID))

LAND_USE_ID_MAP_INV = dict(
    zip(LAND_USE_ID, LAND_USE))



PUBLIC_SERVICES_ID = (
    BUSINESS,
    WASTEMGMT,
    SCHOOL,
    (HOSPITAL_L, HOSPITAL_S),
    RECREATION,
    OFFICE
)

PUBLIC_SERVICES = (
    'shopping',
    'wastemgmt',
    'education',
    'medical care',
    'entertainment',
    'working'
)

GREEN_ID = (
    GREEN_L,
    GREEN_S
)
WASTEMGMT_ID = WASTEMGMT
GREEN_AREA_THRESHOLD = 1500
WASTEMGMT_AREA_THRESHOLD = 500

TYPE_COLOR_MAP = {
    'boundary': 'lightgreen',
    'business': 'fuchsia',
    'feasible': 'white',
    'green_l': 'green',
    'green_s': 'lightgreen',
    'hospital_l': 'purple',
    'hospital_s': 'cyan',
    'wastemgmt': 'gold',
    'outside': 'black',
    'residential': 'yellow',
    'road': 'red',
    'school': 'darkorange',
    'recreation': 'lavender',
    'office' : 'red',
    'waterbody':'blue',
    'drainage':'lightblue'
}
