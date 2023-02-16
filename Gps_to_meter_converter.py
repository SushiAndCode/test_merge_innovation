import math
# Convertitore da gps a metri
def converter(lon1, lat1, lon2, lat2):
    # raggio della terra
    R = 6372795.477598
    lon1 = lon1*math.pi/180
    lat1 = lat1*math.pi/180
    lon2 = lon2*math.pi/180
    lat2 = lat2*math.pi/180

    latm = (lat1+lat2)/2
    lonm = (lon1+lon2)/2

    # deltax è la distanza in metri lungo l'asse x (longitudine), si calcola
    # utilizzando la stessa latitudine per i due punti e variando solo la
    # longitudine

    deltax = R*math.acos(math.sin(latm)*math.sin(latm) +
                         math.cos(latm)*math.cos(latm)*math.cos(lon1-lon2))
    deltay = R*math.acos(math.sin(lat1)*math.sin(lat2) +
                         math.cos(lat1)*math.cos(lat2)*math.cos(lonm-lonm))
    dist = math.sqrt(deltax ^ 2+deltay ^ 2)

    if lon2 < lon1:
        deltax = -deltax
    if lat2 < lat1:
        deltay = -deltay
    return dist
