import numpy as np
from rapidfuzz import fuzz, process  
import unidecode
import cv2
import string
import unidecode

def find_key(template_data, all_text_image, result):
    all_text_data = []
    key_matched_data = []

    # Prepare data
    json_keys = [pair['key'].strip() for pair in template_data.values()]
    # json_labels = [pair['label'].strip() for pair in template_data.values()]

    # Process each detected line of text
    for line in result:
        flag = 0
        text = line[1]
        punctuations = string.punctuation
        text = "".join([char for char in text if char not in punctuations])
        
        # Tính toán tọa độ x, y thấp nhất và cao nhất
        coordinates = line[0]
        x_min = min(coordinates, key=lambda point: point[0])[0]
        x_max = max(coordinates, key=lambda point: point[0])[0]
        y_min = min(coordinates, key=lambda point: point[1])[1]
        y_max = max(coordinates, key=lambda point: point[1])[1]

        # Thêm dữ liệu văn bản vào danh sách
        all_text_data.append({
            'text': text,
            'coordinate': [x_min, y_min],
            'width': x_max - x_min,
            'height': y_max - y_min
        })

        # Vẽ hình chữ nhật xung quanh văn bản
        all_text_image = cv2.rectangle(all_text_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 192, 203), 2)

        if text in json_keys:
            key_matched_data.append({
                'text': text,
                'text_key': text,
                'coordinate': [x_min, y_min],
                'width': x_max - x_min,
                'height': y_max - y_min,
                'flag': flag
            })
        else:
            def generate_overlapping_substrings(text, token_count):
                substrings = []
                tokens = text.split()
                for i in range(len(tokens) - token_count + 1):
                    substring = " ".join(tokens[i : i + token_count])
                    substrings.append(substring)
                return substrings
            
            text_uni = unidecode.unidecode(text.lower())
            for text_key in json_keys:
                text_key_uni = unidecode.unidecode(text_key.lower())
                substrings_with_same_tokens = generate_overlapping_substrings(text_uni, len(text_key_uni.split()))
                if len(substrings_with_same_tokens) > 1:
                    flag = 1
                elif len(substrings_with_same_tokens) == 1:
                    flag = 0
                best_match = process.extractOne(text_key_uni, substrings_with_same_tokens)
                
                if best_match is not None:
                    # matching_substring = best_match[0]  # Get the substring itself
                    # similarity_score = best_match[1] 

                    if best_match[1] > 90:
                        key_matched_data.append({
                            'text': text,
                            'text_key': text_key,
                            'coordinate': [x_min, y_min],
                            'width': x_max - x_min,
                            'height': y_max - y_min,
                            'flag': flag
                        })

                    # if json_labels[json_keys.index(text_key)] == "age" and best_match[1] == 86:  # You can adjust the similarity threshold
                    # # if len(best_match[0].split()) < len(text_uni.split()):
                    #     key_matched_data.append({
                    #         'text': text,
                    #         'text_key': text_key,
                    #         'coordinate': [x_min, y_min],
                    #         'width': x_max - x_min,
                    #         'height': y_max - y_min,
                    #         'flag': flag
                    #     })
                    # elif json_labels[json_keys.index(text_key)] == "age" and best_match[1] > 90:
                    #     key_matched_data.append({
                    #         'text': text,
                    #         'text_key': text_key,
                    #         'coordinate': [x_min, y_min],
                    #         'width': x_max - x_min,
                    #         'height': y_max - y_min,
                    #         'flag': flag
                    #     })
                    # elif best_match[1] > 90:
                    #     key_matched_data.append({
                    #         'text': text,
                    #         'text_key': text_key,
                    #         'coordinate': [x_min, y_min],
                    #         'width': x_max - x_min,
                    #         'height': y_max - y_min,
                    #         'flag': flag
                    #     })
                # else:
                #     matching_substring = 0  # or any default value you want to assign
                #     similarity_score = 0

            #     print("Best matching substring (overlapping, same token count):", matching_substring)
            #     print("Similarity score:", similarity_score)
            # print("key match data: ", self.key_matched_data)

    points_array = np.array([[item['coordinate'][0], item['coordinate'][1]] for item in all_text_data])
    return all_text_data, key_matched_data, points_array, json_keys


def find_key_value_from_large_set(points, pair, key_matched_data, all_text_data):
    """
    Function to find key-value pairs from a large set of points using template information.

    Parameters:
    - points: Array of coordinates representing a large set of points.
    - pair: Template information for a key-value pair.

    Finds the key and value coordinates in the set of points based on template information.
    """
    vectors_from_root = points - pair['root']
    magnitudes = np.linalg.norm(vectors_from_root, axis=1)
    angles = np.degrees(np.arccos(
        np.dot(vectors_from_root, pair['ox']) /
        (np.linalg.norm(vectors_from_root, axis=1) * np.linalg.norm(pair['ox']))
    ))

    # Find the indices with the closest magnitudes and angles to the KeyValuePair
    closest_index = np.argmin(
        np.abs(magnitudes - pair['magnitude_root_key']) + np.abs(angles - pair['angle_degrees_key'])
    )
    key = points[closest_index]

    # Find corresponding information in all_text_data
    key_info = next(item for item in all_text_data if np.array_equal(item['coordinate'], key))

    # Check if the text in the key bounding box matches any text in key_matched_data
    matching_key_data = next((item for item in key_matched_data if item['text'] == key_info['text']), None)
    # matching_key_data = next((item for item in key_matched_data if item['text_key'] == pair['key']), None)
    if matching_key_data is not None and matching_key_data['flag'] == 1:
        words = matching_key_data['text'].split()
        value_text =  " ".join(words[len(matching_key_data['text_key'].split()):])
        return {
            'key_coordinates': matching_key_data['coordinate'],
            'value_coordinates': matching_key_data['coordinate'],
            'key_text': matching_key_data['text_key'],
            'value_text': value_text,
            'key_width': pair['key_size'][0],
            'key_height': pair['key_size'][1],
            'value_width': matching_key_data['width'],
            'value_height': matching_key_data['height']
        }

    # If there is no match, update the key with the coordinates from key_matched_data having the same content as pair['key']
    else:
        for item in key_matched_data:
            text_key = unidecode.unidecode(item['text_key'].lower())
            key_con = unidecode.unidecode(pair['key'].lower())
            comparision = fuzz.ratio(text_key, key_con)
            if comparision >= 85 and item['flag'] == 0:
                matching_key_data = item
                break
            elif comparision >= 85 and item['flag'] == 1:
                words = item['text'].split()
                value_text =  " ".join(words[len(item['text_key'].split()):])
                return {
                    'key_coordinates': item['coordinate'],
                    'value_coordinates': item['coordinate'],
                    'key_text': item['text_key'],
                    'value_text': value_text,
                    'key_width': pair['key_size'][0],
                    'key_height': pair['key_size'][1],
                    'value_width': item['width'],
                    'value_height': item['height']
                }

    key = matching_key_data['coordinate'] if matching_key_data is not None else key
    # key = matching_key_data['coordinate'] 

    # Approach 1
    x_new = key[0] + (pair['magnitude_key_value'] * pair['cosine_angle_key_value'])
    y_new = key[1] + (pair['magnitude_key_value'] * pair['sine_angle_key_value'])
    value_new = np.array([x_new, y_new])
    distances = np.linalg.norm(points - value_new, axis=1)
    closest_index_2 = np.argmin(distances)

    # Approach 2
    closest_index_3 = np.argmin(
        np.abs(magnitudes - pair['magnitude_root_value']) + np.abs(angles - pair['angle_degrees_value'])
    )

    # Approach 3
    closest_index_4 = np.argmin(
        np.abs(np.linalg.norm(points - key, axis=1) - pair['magnitude_key_value'])
    )

    # Choose the value vector based on the comparison of three closest indices
    unique_indices, counts = np.unique(
        [closest_index_2, closest_index_3, closest_index_4], axis=0, return_counts=True
    )
    most_common_index = unique_indices[np.argmax(counts)]

    # Check if all indices are different and choose closest_index_2
    if len(set([closest_index_2, closest_index_3, closest_index_4])) == 3:
        most_common_index = closest_index_2

    value = points[most_common_index]

    # Find corresponding information in all_text_data
    key_info = next(item for item in all_text_data if np.array_equal(item['coordinate'], key))
    value_info = next(item for item in all_text_data if np.array_equal(item['coordinate'], value))

    return {
        'key_coordinates': key,
        'value_coordinates': value,
        'key_text': key_info.get('text', ''),
        'value_text': value_info.get('text', ''),
        'key_width': key_info.get('width', 0),
        'key_height': key_info.get('height', 0),
        'value_width': value_info.get('width', 0),
        'value_height': value_info.get('height', 0)
    }