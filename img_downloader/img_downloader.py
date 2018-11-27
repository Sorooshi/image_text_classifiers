from bs4 import BeautifulSoup
import urllib.request, urllib.parse
import os
import json
import csv
from collections import OrderedDict


def get_soup(url,header):
    return BeautifulSoup(urllib.request.urlopen(urllib.request.Request(url, headers=header)),'html.parser')

query_dict = {
    "tools": (
        ["hammer", "good_hammer", "hammer_set", "hammer_types", "hammer_tool","hammer as a tool",
                         "hammer and nails" ],
        ["screwdriver", "good_screwdriver" ,"screwdriver_set", "screwdriver_types",
                         "screwdriver_tool","screwdriver and screw", "screwdriver as a tool",
                         "screwdriver as a tool lowes"],
        ["pliers", "good_pliers", "pliers_set", "pliers_types", "pliers_tool",
                         "pliers as a tool" ],
        ["drill", "good_drill", "drill_set", "drill_type", "drill_tool", "drill heavy duty",
                        "hammer drill", "handheld jack hammer"]),

    "medical": (
        ["doctor", "general_doctor", "young_doctor", "experienced_doctor", "doctor_of_medicine"
                              , "doctor male", "surgical doctor"],
        ["nurse", "good_nurse", "young_nurse","experienced_nurse", "nurse hospital",
                           "nurse with hat", "nurse uniform", "nurse uniform pink", "old fashioned nurse"],
        ["dentist", "good_dentist", "young_dentist", "experienced_dentist", "dentist oral surgery",
                           "orthodontist dentist", "dental hygienist"])}

# for key_word in query_list:
for parent, children in query_dict.items():
    print("parent:", parent)
    print(" ")
    print("children:", children)
    print(" ")
    for key_words in children:
        # add the directory for your image here
        DIR = str(parent)
        if not os.path.exists(DIR):
            os.mkdir(DIR)
        DIR = os.path.join(DIR, key_words[0]) #.split()[0])

        if not os.path.exists(DIR):
            os.mkdir(DIR)

        for key_word in key_words:
            print("keywords:", key_words)
            print("key word:", key_word)
            query = key_word
            print("query:", query)
            image_name = query
            query = query.split()
            query ='+'.join(query)

            google_url = "https://www.google.com/search?tbm=isch&q=" + query
            bing_url = "https://www.bing.com/images/search?q=" + query + "&FORM=HDRSC2"
            # url_yandex = "https://yandex.ru/images/search?text="+query

            # print("google url:", google_url)
            # print("bing url:", bing_url)
            google_img_counter = 0
            bing_img_counter = 0

            header = {'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 "
                             "(KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}

            google_soup = get_soup(google_url, header)
            print("soup:", google_soup)

            bing_soup = get_soup(bing_url, header)
            print("bing_soup:", bing_soup)

            google_img_urls = [] #contains the google_img_urls for Large original images, type of image
            google_page_urls = [] #contains the page_urls and the corresponding title for future usage

            bing_img_urls = []  # contains the google_img_urls for Large original images, type of image
            bing_page_urls = []  # contains the page_urls and the corresponding title for future usage

            for a in google_soup.find_all("div",{"class":"rg_meta"}):
                page_url, page_title, img_url, img_type = json.loads(a.text)["ru"], json.loads(a.text)["pt"], \
                                                          json.loads(a.text)["ou"], json.loads(a.text)["ity"]
                google_img_urls.append((img_type, img_url))
                google_page_urls.append((page_title, page_url))

            for a in bing_soup.find_all("a", {"class": "iusc"}):
                mad, m = json.loads(a["mad"]), json.loads(a["m"])
                turl, murl = mad["turl"], m["murl"] #png, #jgp
                page_url, page_title, bing_image_name = m["purl"], str("not available yet!"), \
                                                        urllib.parse.urlsplit(murl).path.split("/")[-1]
                bing_img_urls.append((bing_image_name, turl, murl))
                bing_page_urls.append((page_title, page_url))


            ##
            # inja
            ##

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

            bing_img_counter = google_img_counter + bing_img_counter+1
            print("bing_img_counter:", bing_img_counter)

            for i, (bing_image_name, turl, murl) in enumerate(bing_img_urls):
                print("bing i:", i)
                try:
                    raw_img = urllib.request.urlopen(murl).read()
                    # bing_counter = len([i for i in os.listdir(DIR)]) + 1  # if image_name in i
                    print("inja:", image_name)
                    print("bing_img_counter:", bing_img_counter)

                    f = open(os.path.join(DIR, image_name+str(bing_img_counter)+".jpg"), 'wb')

                    key = str(image_name + "_" + str(bing_img_counter))
                    print("key:", key)
                    all_in_one_data[key] = (murl, bing_page_urls[i][0], bing_page_urls[i][-1], str(parent))
                    print("all in one inja:", all_in_one_data.items())
                    bing_img_counter += 1
                    print("bing img_counter:", bing_img_counter)

                    f.write(raw_img)
                    f.close()

                except Exception as err:
                    print("bing could not load:" + murl)
                    print("bing err:", err)

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
