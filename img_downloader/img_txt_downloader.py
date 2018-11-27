from bs4 import BeautifulSoup
import urllib.request, urllib.parse
import os
import json
import csv
from collections import OrderedDict


def get_soup(url,header):
    return BeautifulSoup(urllib.request.urlopen(urllib.request.Request(url, headers=header,)),'html.parser')

queries_list = ["hammer",]#"screwdriver", "pliers","drill",]# "doctor", "nurse", "dentist", "other"]
adjectives = ['good',]# 'bad', 'better', 'worse', 'best', 'worst', 'small', 'medium', 'large', 'new', 'old',]

all_in_one_data = OrderedDict([])
query_dict = OrderedDict([])

for q in queries_list:
    query_dict[str(q)] = [str(q) + " " + str(adj) for adj in adjectives]
    query_dict[str(q)].insert(0, q)
print("query_dict:", query_dict.items())

for parent, children in query_dict.items():
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

        image_name = query.split()[0]
        query = query.split()
        query ='+'.join(query)

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
                google_img_counter += 1
                print("google img_counter:", google_img_counter)

                # f.write(raw_img)
                # f.close()

            except Exception as err:
                print("could not load: " + img_url)
                print("err:", err)

# with open("vgg_mc_sl.json", "w") as fp:
#     json.dump(all_in_one_data, fp)

with open("C:/Users/srsha/image_text_classifier/text_analysis/vgg_mc_sl.txt", "w") as fp:
    for k, v in all_in_one_data.items():
        print("v[1]:", str(v[1]), '--'+ k.split("_")[0]+'--')
        try:
            fp.write(v[1] +'--'+ k.split("_")[0]+'--'+'\n')
        except Exception as error:
            print("error:", error)

# with open("all_in_one_data_data.json", "r") as fp:
#     all_in_one_data = json.load(fp)

''' 
with open('vgg_mc_sl.csv', 'w', newline='') as fp:
    fp.write("PAGE TITLE,PAGE URL,PARENT NAME,IMAGE LABEL,IMAGE URL, \n")
    spamwriter = csv.writer(fp, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for image_name_num, description in all_in_one_data.items():
        print("description:", description)

        # print("description[1].strip():", description[1].encode("utf-8"))
        # print("description[1]:", description[1])

        spamwriter.writerow([str(description[1].ecnode("utf-8"))] + #.encode("utf-8")
                            [str(description[-2].encode("utf-8"))] + #.encode("utf-8")
                            [description[-1].encode("utf-8")] + [image_name_num.encode("utf-8")] +
                            [description[0].encode("utf-8")] #
                            )
'''