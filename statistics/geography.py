from area import area as area_model
from math import radians, cos, sin, asin, sqrt
from enum import Enum


# mean earth radius - https://en.wikipedia.org/wiki/Earth_radius#Mean_radius
AVG_EARTH_RADIUS_KM = 6371.0088


class Unit(Enum):
    """
    Enumeration of supported units.
    """

    KILOMETERS = 'km'
    METERS = 'm'
    MILES = 'mi'


# Unit values taken from http://www.unitconversion.org/unit_converter/length.html
_CONVERSIONS_FROM_KM = {
    Unit.KILOMETERS: 1.0,
    Unit.METERS: 1000.0,
    Unit.MILES: 0.621371192
}


def distance(point1, point2, unit: Unit = Unit.KILOMETERS) -> float:
    """
    Calculate the great-circle distance between two points on the Earth surface.
    Takes two 2 array-like, containing the latitude and longitude of each point
    in decimal degrees, and, optionally, a unit of length.

    :param point1: array_like
        first point; array like of (latitude, longitude) in decimal degrees
    :param point2: array_like
        second point; array like of (latitude, longitude) in decimal degrees
    :param unit: str
        a member of Unit, or, equivalently, a string containing the
        initials of its corresponding unit of measurement
        default 'km' (kilometers).
    :return: Number
        the distance between the two points in the requested unit, as a float.
    """
    # unpack latitude/longitude
    lat1, lng1 = point1
    lat2, lng2 = point2

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1 = radians(lat1)
    lng1 = radians(lng1)
    lat2 = radians(lat2)
    lng2 = radians(lng2)

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = sin(lat * 0.5)**2 + cos(lat1) * cos(lat2) * sin(lng * 0.5)**2

    return 2 * (AVG_EARTH_RADIUS_KM * _CONVERSIONS_FROM_KM[Unit(unit)]) * asin(sqrt(d))


def area(*points, unit: str = 'km2') -> float:
    """
    Calculate area in earth
    :param points: array_like
        points of polygon likes (lat, lon)
    :param unit: str
        result unit -> 'km2' or 'm2'
    :return: Number
        area result
    """

    if unit == 'km2':
        scale = 1000 ** 2
    elif unit == 'm2':
        scale = 1
    else:
        raise ValueError('Unit must be one of `km` or `m`.')

    geo_json = {
        'type': 'Polygon',
        'coordinates': [list(map(lambda item: (item[1], item[0]), points))]
    }

    return area_model(geo_json) / scale
