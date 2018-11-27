from bs4 import BeautifulSoup
import urllib.request, urllib.parse
import os
import json
import csv
from collections import OrderedDict


def get_soup(url,header):
    return BeautifulSoup(urllib.request.urlopen(urllib.request.Request(url, headers=header)),'html.parser')


adjectives = ['good', 'bad', 'best', 'worst', 'small', 'medium', 'large', 'new', 'old',] # 'better', 'worse', 'white', 'black', 'red', 'blue', 'easy', 'hard',

queries_list = ['black watch', 'white watch', 'red watch', 'black dress', 'white dress', 'red dress', 'black scarf',
              'white scarf', 'red scarf', 'black jeans', ] #'white jean', 'red jean'

# multi labels
query_dict = OrderedDict([])
for q in queries_list:
    query_dict[str(q)] = [str(q) + " " + str(adj) for adj in adjectives]
    query_dict[str(q)].insert(0, q)
print("query_dict:", query_dict.items())

all_in_one_data = {}

# for key_word in query_list:
for parent, children in query_dict.items():
    print("parent:", parent)
    print(" ")
    print("children:", children)
    print(" ")
    for key_word in children:
        # add the directory for your image here
        DIR = str(parent)
        if not os.path.exists(DIR):
            os.mkdir(DIR)
        DIR = os.path.join(DIR,) #key_words[0]

        if not os.path.exists(DIR):
            os.mkdir(DIR)
        print("key word:", key_word)
        query = key_word
        print("query:", query)
        image_name = query
        query = query.split()
        query ='+'.join(query)

        google_url = "https://www.google.com/search?tbm=isch&q=" + query

        print("google url:", google_url)
        google_img_counter = 0

        header = {'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}

        google_soup = get_soup(google_url, header)
        # print("soup:", google_soup)

        google_img_urls = [] #contains the google_img_urls for Large original images, type of image
        google_page_urls = [] #contains the page_urls and the corresponding title for future usage

        for a in google_soup.find_all("div",{"class":"rg_meta"}):
            page_url, page_title, img_url, img_type = json.loads(a.text)["ru"], json.loads(a.text)["pt"], \
                                                      json.loads(a.text)["ou"], json.loads(a.text)["ity"]
            google_img_urls.append((img_type, img_url))
            google_page_urls.append((page_title, page_url))


        for i, (img_type, img_link) in enumerate(google_img_urls):
            print("google i:", i)
            try:
                raw_img = urllib.request.urlopen(img_link).read()

                if len(img_type) == 0:
                    f = open(os.path.join(DIR, image_name + "_"+ str(google_img_counter)+".jpg"), 'wb')
                else :
                    f = open(os.path.join(DIR , image_name + "_"+ str(google_img_counter)+"."+ img_type), 'wb')

                key = str(image_name+"_"+str(google_img_counter))
                all_in_one_data[key] = (img_link, google_page_urls[i][0], google_page_urls[i][-1], str(parent))
                # print("desc:", (img_link, go/ogle_page_urls[i][0], google_page_urls[i][-1], str(parent)))
                google_img_counter += 1
                print("google img_counter:", google_img_counter)

                f.write(raw_img)
                f.close()

            except Exception as err:
                print("could not load: " + img_link)
                print("err:", err)

with open("all_in_one_data_data.json", "w") as fp:
    json.dump(all_in_one_data, fp)


# with open("all_in_one_data_data.json", "r") as fp:
#     all_in_one_data = json.load(fp)

with open('all_in_one_data_data.csv', 'w', newline='') as fp:
    fp.write("PAGE TITLE, PAGE URL, PARENT NAME, IMAGE LABEL,IMAGE NAME, IMAGE NUMBER, IMAGE URL, \n")
    spamwriter = csv.writer(fp, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for image_name_num, description in all_in_one_data.items():
        spamwriter.writerow( [description[1].encode("utf-8")] + [description[-1].encode("utf-8")] +
                            [str(description[-1])] + [image_name_num] +
                            [(image_name_num).split("-")[0]] + [(image_name_num).split("-")[-1]]
                            + [description[0].encode("utf-8")]
                            )
