import math

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def matching_bounding_boxes(box1, box2, threshold=2.0):
    distances = []

    for point1, point2 in zip(box1, box2):
        distance = calculate_distance(point1, point2)
        distances.append(distance)
        
    average_distance = sum(distances) / len(distances)

    return average_distance <= threshold