## -*- coding: utf-8 -*-
import os
import csv
import json
from bs4 import BeautifulSoup
from collections import OrderedDict
import urllib.request, urllib.parse


def write_txt(dict_to_write, path, file_name):
    cntr = 0
    with open(path+ "/" + file_name + ".txt", "w") as fp:
        for k, v in dict_to_write.items():
            try:
                fp.write(''.join(v + "--" + file_name + '--'))
                fp.write('\n')
            except Exception as error:
                cntr +=1
                # print(error)
        print("missed docs", cntr)

    return None

def write_json(dict_to_write, path, file_name):
    """
    this function muct be called when the all images of one class are download to save the text in the same folder.
    """
    with open(path + "/" + file_name + ".json", "w") as fp:
        json.dump(dict_to_write, fp)
    return None

def write_all_in_one_csv(dict_to_write, path, file_name):
    """
    path must be created manually. c:/user/srsha/.../name_of_the_csv_file
    this function must be call at the end of the manuscript.
    """
    with open(path + file_name + '.csv', 'w', newline='') as fp:
        cntr = 0
        fp.write("PAGE TITLE,PAGE URL,PARENT NAME,IMAGE NUMBER,IMAGE URL, \n")
        spamwriter = csv.writer(fp, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for image_name_num, description in dict_to_write.items():
            # print("description:", description)
            try:
                spamwriter.writerow([str(description[1])] +
                                    [str(description[-2])] +
                                    [str(description[-1])] + [image_name_num] +
                                    [description[0]]  #
                                    )
            except Exception as error:
                cntr += 1
                print("missed docs:", cntr)
    return None

def write_all_in_one_txt(dict_to_write, path, ):
    with open("C:/Users/srsha/image_text_classifier/text_analysis/vgg_mc_sl.txt", "w") as fp:
        for k, v in all_in_one_data.items():
            print("v[1]:", str(v[1]), '--' + k.split("_")[0] + '--')
            try:
                fp.write(v[1] + '--' + k.split("_")[0] + '--' + '\n')
            except Exception as error:
                print("error:", error)
    return None

def read_json(dict_to_read, path):
    with open(dict_to_read +".json", "r") as fp:
        loaded_json = json.load(fp)
    return loaded_json

def get_soup(url,header):
    return BeautifulSoup(urllib.request.urlopen(urllib.request.Request(url, headers=header,)),'html.parser')

queries_list = ['black dress', 'black scarf', 'black watch', 'others', 'red dress', 'red scarf', 'red watch',
                'white dress', 'white watch',]
adjectives = ['good', 'bad', 'better', 'worse', 'best', 'worst', 'small', 'medium', 'large', 'new', 'old',]
all_in_one_data_path = os.path.dirname('img_downloader')

all_in_one_data = OrderedDict([])
text_data_all = OrderedDict([])
query_dict = OrderedDict([])

for q in queries_list:
    query_dict[str(q)] = [str(q) + " " + str(adj) for adj in adjectives]
    query_dict[str(q)].insert(0, q)
print("query_dict:", query_dict.items())

for parent, children in query_dict.items():
    text_data = OrderedDict([])
    print("parent:", parent)
    print(" ")
    print("children:", children)
    print(" ")
    google_img_counter = 0
    for query in children:
        # add the directory for your image here
        DIR = str(parent)
        if not os.path.exists(DIR):
            os.mkdir(DIR)

        image_name = query
        print("image_name:", image_name)
        query = query.split()
        query ='+'.join(query)
        print("query:", query)

        google_url = "https://www.google.com/search?tbm=isch&q=" + query

        print("google url:", google_url)

        header = {'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}

        google_soup = get_soup(google_url, header)

        google_img_urls = [] #contains the google_img_urls for Large original images, type of image
        google_page_urls = [] #contains the page_urls and the corresponding title for future usage

        for a in google_soup.find_all("div",{"class":"rg_meta"}):
            page_url, page_title, img_url, img_type = json.loads(a.text)["ru"], json.loads(a.text)["pt"], \
                                                      json.loads(a.text)["ou"], json.loads(a.text)["ity"]
            google_img_urls.append((img_type, img_url))
            google_page_urls.append((page_title, page_url))

        for i, (img_type, img_url) in enumerate(google_img_urls):
            print("google i:", i)
            try:
                # raw_img = urllib.request.urlopen(img_url).read()
                #
                # if len(img_type) == 0:
                #     f = open(os.path.join(DIR, image_name + "_" + str(google_img_counter)+".jpg"), 'wb')
                # else :
                #     f = open(os.path.join(DIR , image_name + "_" + str(google_img_counter)+"."+ img_type), 'wb')

                key = str(image_name+"_"+str(google_img_counter))
                all_in_one_data[key] = (img_url, google_page_urls[i][0], google_page_urls[i][-1], str(parent))
                text_data[google_img_counter] = google_page_urls[i][0]
                google_img_counter += 1
                print("google img_counter:", google_img_counter)

                # f.write(raw_img)
                # f.close()

            except Exception as err:
                print("could not load: " + img_url)
                print("err:", err)
        write_json(dict_to_write=text_data, path=DIR, file_name=parent)
        write_txt(dict_to_write=text_data, path=DIR, file_name=parent)

write_all_in_one_csv(dict_to_write=all_in_one_data, path=all_in_one_data_path, file_name='vgg_mc_ml_text')