import urllib
import urllib.request as request
import cairosvg
src = 'https://openseauserdata.com/files/f999b4b1aedc879614e960cd8faacf07.svg'
# src = 'http://cdn.nba.net/assets/logos/teams/secondary/web/PHI.svg'
alt = 'The Laborers#3534'

opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
img = urllib.request.urlretrieve(src, filename=f'temp_0704/{alt}.svg')

s = cairosvg.svg2png(url=f'temp/{alt}.svg', write_to=f'temp/{alt}.png')

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
