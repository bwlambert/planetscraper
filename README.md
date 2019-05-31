# planetscraper

This repository contains scripts that assist in automating the ordering of imagery from Planet Labs.

Typical usage will follow this pattern:
```
python order_api.py points.csv 20-Aug-18 30-Aug-18 orderName
```
In the preceding command, columns named 'lat' and 'long' must exist in the 'points.csv' file.

If the script times out or crashes for any reason, the image activation process will continue on Planet's servers, and any imagery that has been ordered but not downloaded can be acquired by starting the script with the 'resume' option:
```
python order_api.py resume
```

Remaining monthly quota can be checked with:
```
python order_api.py quota
```

Presently, the script expects to find the user's Planet API key in a file named: 'key.txt'
