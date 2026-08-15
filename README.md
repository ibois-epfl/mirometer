# mirometer
Mirometer is a tool that measures the area of a colored patch in a picture, using fiducial markers. 
In the case of IBOIS it is used to measure the area of joint faces made in irregular timber pieces.

# Usage
To use this code, we rely on [uv](https://docs.astral.sh/uv/getting-started/installation/), so you will need that installed.

With uv installed, run:
```bash
uv run main.py -i <path-of-image> -c (r,g,b) 
```
You should have as result a .txt file in the same folder as the picture given, with the area of the patch in it, in mm2