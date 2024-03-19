from bs4 import BeautifulSoup as bs
from PIL import Image

with open('annotations.xml', 'r') as f:
    xml_file = f.read()

soup = bs(xml_file, 'xml')
count = 0
for image in soup.find_all('image'):
    name = image['name']

    soup = bs(str(image), 'xml')

    for box in soup.find_all('box'):
        box_name = f"{box['label']}-{count}"
        im = Image.open(f'imgs/img/{name}')
        im_crop = im.crop((float(box['xtl']), float(box['ytl']), float(box['xbr']), float(box['ybr'])))
        im_crop.save(f'imgs/refactor_img/{box_name}.png', quality=95)
        count += 1
        print(f"{box_name} saved")

