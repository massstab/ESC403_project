import numpy as np

def lv95_latlong(E, N):
    """
    Converts lv95 to latitude and longitude according to the Federal Office of Topography swisstopo approximation.
    Described in 'data/raw/descriptions/projektion_lv95_wgs.pdf.
    Longitude is the measurement east or west of the prime meridian. So it corresponds to 'AccidentLocation_CHLV95_E'
    in our datasets and the swisstopo convention.

    :param E: AccidentLocation_CHLV95_E
    :param N: AccidentLocation_CHLV95_N
    :return: latitude, longitude in the DD format
    """
    y = (E - 2600000)/1000000
    x = (N - 1200000)/1000000

    longitude = 2.6779094 + 4.728982 * y + 0.791484 * y * x + 0.1306 * y * x**2 - 0.0436 * y**3
    latitude = 16.9023892 + 3.238272 * x - 0.270978 * y**2 - 0.002528 * x**2 - 0.0447 * y**2 * x - 0.0140 * x**3

    longitude = longitude * 100 / 36
    latitude = latitude * 100 / 36

    return np.round(longitude, 5), np.round(latitude, 5)

def gis_pixel(center, scale, dpi):
    """
    FUNKTIONIERT NOCH NICHT!
    Computes the boundaries of the map of an images printed from the gis browser (https://maps.zh.ch/).
    :param center: Printed in the bottom right corner. Coordinates in LV95.
    :param scale: Also in the bottom right corner of the print. Given in LV95 coordinates. Center of the map.
    :param dpi: The dpi of the image in adjustable in the print dialog.
    :return: The corner coordinates in LV95 (meters).
    """
    pixels = np.array([2168, 2750])
    true_distance = pixels * 0.0254 * scale / dpi
    left = center[0] - 0.5 * true_distance[0]
    right = center[0] + 0.5 * true_distance[0]
    bottom = center[1] - 0.5 * true_distance[1]
    top = center[1] + 0.5 * true_distance[1]
    return (left, right, bottom, top)