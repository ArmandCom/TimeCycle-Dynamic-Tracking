import xml.etree.ElementTree as ET
import numpy as np

def get_polygons_from_xml(path='/data/Ponc/tracking/ground_truth/philadelphia_warmup.xml'):
    print("Loading data from xml...")
    tree = ET.parse(path)
    root = tree.getroot()
    pols = []
    num_images = 0
    for child in root:
        if(child.tag == 'image'):
            num_images = num_images + 1
            polygon_elem = child[0]
            points = polygon_elem.get('points')
            points = points.split(';')
            four_points = []
            for p in points:
                # p is x,y
                ps = p.split(',')
                px, py = int(float(ps[0])), int(float(ps[1]))
                four_points.append([px,py])
            pols.append(four_points)
    points_np = np.asarray(pols)
    return points_np
        

if __name__ == '__main__':
    x = get_polygons_from_xml()
    print(x.shape)
    print(x[0,:,:])