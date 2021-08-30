import os
import const
import json


def import_data(df, img_list):

    # for i in range(1, 20):
    #   image_path = os.path.join(const.Config.IMG_FOLDER, const.Config.IMG_NAME + str(i).zfill(5) + const.Config.IMG_TYPE)
    #   img = cv2.imread(image_path)
    #   cv2.imshow('image'+str(i).zfill(5), img)
    #   cv2.waitKey(0)
    #   cv2.destroyAllWindows()
    #   image_list = image_list + img

    for i in range(1, 27639):
        path = os.path.join(const.Config.JSON_FOLDER, const.Config.JSON_NAME + str(i).zfill(5) + const.Config.JSON_TYPE)
        with open(path, encoding="utf8") as json_file:
            row = json.load(json_file)
            row['attributes_course'] = row['attributes']['course'] if 'course' in row['attributes'] else None
            row['attributes_cuisine'] = row['attributes']['cuisine'] if 'cuisine' in row['attributes'] else None
            row['attributes_holiday'] = row['attributes']['holiday'] if 'holiday' in row['attributes'] else None
            row['attribution_url'] = row['attribution']['url'] if 'url' in row['attribution'] else None
            row['attribution_text'] = row['attribution']['text'] if 'text' in row['attribution'] else None
            row['attribution_html'] = row['attribution']['html'] if 'html' in row['attribution'] else None
            row['attribution_logo'] = row['attribution']['logo'] if 'logo' in row['attribution'] else None
            for j in range(len(row['nutritionEstimates'])):
                row['n_est_att_' + str(j)] = row['nutritionEstimates'][j]['attribute'] if 'attribute' in row['nutritionEstimates'][j] else None
                row['n_est_val_' + str(j)] = row['nutritionEstimates'][j]['value'] if 'value' in row['nutritionEstimates'][j] else None
                row['nutr_est_' + row['n_est_att_' + str(j)]] = row['n_est_val_' + str(j)]
                row.pop('n_est_att_' + str(j))
                row.pop('n_est_val_' + str(j))
            row['flavor_' + str(j)] = [row['flavors'][j] for j in row['flavors']]
            df = df.append(row, ignore_index=True).drop(columns=['attributes', 'attribution', 'images', 'source', 'nutritionEstimates', 'flavors'])

    return df, img_list


def feature_selection():
    return None


def univariant_analysis():
    return None


def cross_validation():
    return None


def random_forest():
    return None


def support_vector_machine():
    return None


def naive_bayes():
    return None
