import os
import xml.etree.ElementTree as ET
import glob
def xml_to_txt(path):
    num = 1
    for xml_file in glob.glob(path + '/*.xml'):
        xml_name = xml_file[:-4]
        if not os.path.exists(xml_name + '.jpg'):
#             os.remove(xml_file)
            print(xml_name)
            continue
        tree = ET.parse(xml_file)
        root = tree.getroot()
        temp_list = []
        xml_list = []
        for member in root.findall('object'):
            bbox = member.find('bndbox')
            value = (
                     int(bbox.find('xmin').text),
                     int(bbox.find('ymin').text),
                     int(bbox.find('xmax').text),
                     int(bbox.find('ymax').text),
                     0,
                     )
            temp_list.append(value)
        xml_list.append(temp_list)
        txt_path = path + 'all_train.txt'
        with open(txt_path, 'a') as fp:
            for xml_value in xml_list:
                fp.write(xml_name + '.jpg')
                fp.write(' ')
                for temp in xml_value:
                    for i in range(len(temp)):
                        if (i < len(temp) - 1):
                            fp.write(str(temp[i])+',')
                        else:
                            fp.write(str(temp[i]))
                    fp.write(' ')
                fp.write('\n')
        num = num + 1


xml_to_txt('/home/lcx/room_dataset/img/img')