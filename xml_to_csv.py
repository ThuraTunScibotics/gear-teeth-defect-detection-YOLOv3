import os
import glob
import argparse
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def generate_csv_file(path_to_images, path_to_output_csv_file):
    
    xml_df = xml_to_csv(path_to_images)
    xml_df.to_csv(path_to_output_csv_file, index=None)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generating csv file from xml files")
    parser.add_argument("--path_to_xml", type=str, help="folder that contains xml files",
	                    default="/home/thura/Desktop/gear_teeth_detection/img_xml_data/train")
    parser.add_argument("--path_to_csv", type=str, help="full path to annotations csv file",
	                    default="/home/thura/Desktop/gear_teeth_detection/img_xml_data/train/annotation.csv")
    args = parser.parse_args()

    path_to_images = args.path_to_xml
    path_to_csv_file = args.path_to_csv
    generate_csv_file(path_to_images, path_to_csv_file)
    print("Successfully generated csv file....")
    
