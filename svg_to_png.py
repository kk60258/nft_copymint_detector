import urllib
import urllib.request as request
import cairosvg
import os
DIR = 'temp_0704'
import shutil


base = os.listdir(DIR)
base = [f for f in base if f.endswith('svg')]
print(f'svg len {len(base)}')
for f in base:
    alt = f[:-4]
    print(alt)
    try:
        cairosvg.svg2png(url=os.path.join(DIR, f'{alt}.svg'), write_to=os.path.join(DIR, f'{alt}.png'))
    except:
        src = os.path.join(DIR, f'{alt}.svg')
        src_png = os.path.join(DIR, f'{alt}.png')
        dst = os.path.join(DIR, 'temp.svg')
        dst_png = os.path.join(DIR, 'temp.png')
        shutil.copyfile(src, dst)

        cairosvg.svg2png(url=dst, write_to=dst_png)
        shutil.copyfile(dst_png, src_png)
        os.remove(dst)
        os.remove(dst_png)


# req = urllib.request.Request(src, headers={'User-Agent': 'Mozilla/5.0'})
# r = request.urlopen(req)
# with open(f'temp/{alt}.svg', "wb") as f:
#     f.write(r.read())
# r.close()



# example_url = 'http://cdn.nba.net/assets/logos/teams/secondary/web/PHI.svg'
#
# img = urllib.request.urlretrieve(example_url, "PHI.svg")
#
# s = cairosvg.svg2png(url=img)
