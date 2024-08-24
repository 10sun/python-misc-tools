'''
Author: J , jwsun1987@gmail.com
Date: 2024-02-09 17:59:24
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


try:
    from pptx.util import (Cm, Pt)
    from pptx.dml.color import RGBColor
except ImportError:
    import sys
    sys.path.append(r"\\merlin\lib_isg\\28.Alternative Data\Code\python-quant\libs\common\reporting\slides\python-pptx-master")
    from pptx.util import (Cm, Pt)
    from pptx.dml.color import RGBColor
#from common.reporting.style.lo_color import *
from common.reporting.style.lo_style import *

##### font params
FONT_NAME = 'Arial Narrow'
FONT_COLOR = RGBColor(61, 49, 46)

# Table font params
TABLE_FONT_COLOR = RGBColor(255, 255, 255)
TABLE_TITLE_FONT_SIZE = Pt(12)
TABLE_HEADER_FONT_SIZE = Pt(10)
TABLE_BODY_FONT_SIZE = Pt(10)

# Chart font params
CHART_NARRATIVE_FONT_SIZE = Pt(16)
CHART_CAPTION_FONT_SIZE = Pt(14)
CHART_TITLE_FONT_SIZE = Pt(14)
CHART_INFO_FONT_SIZE = Pt(12)
CHART_HEADER_FONT_SIZE = Pt(10)
CHART_BODY_FONT_SIZE = Pt(9)
CHART_FONT_COLOR = FONT_COLOR

# Footnote font params
FOOTNOTE_FONT_SIZE = Pt(8)
FOOTNOTE_FONT_COLOR = FONT_COLOR

##### Line params
# Chart line params
CHART_LINE_WIDTH = Pt(1.5)
CHART_BOLD_LINE_WIDTH = Pt(2.25)
CHART_EX_BOLD_LINE_WIDTH = Pt(2.5)


# Textbox line params
TEXTBOX_LINE_WIDTH = Pt(0.25)

##### Layout params
# Page dimensions
PAGE_WIDTH = Cm(24.19)
PAGE_HEIGHT = Cm(15.3)

# Title dimensions
TITLE_X = Cm(0.61)
TITLE_Y = Cm(1.53)
TITLE_WIDTH = Cm(24.19)
TITLE_HEIGHT = Cm(1.49)

# Body dimensions
BODY_WIDTH = Cm(24.19)
BODY_HEIGHT = Cm(12.3)
BODY_X = Cm(0.61)
BODY_Y = Cm(4.53)

# Caption dimensions
CAPTION_HEIGHT = Cm(1)

# Spacing parmas
MARGIN_VER = Cm(0.3)
MARGIN_HOR = Cm(0.3)



LO_DIMGRAY = RGBColor(57,46,44) # '#392E2C' #'rgb(57, 46, 44)'
LO_GRAY = RGBColor(146, 132, 122) #'#92847A'#'rgb(146, 132, 122)'
LO_TAN = RGBColor(204, 194, 187) #'#CCC2BB' #'rgb(204, 194, 187)'
LO_LIGHTTAN =RGBColor(245, 243, 241) # '#F5F3F1' #'rgb(245, 243, 241)'
LO_BROWN = RGBColor(142, 101, 76) #'#8E654C' #'rgb(142, 101, 76)'
LO_NAVY = RGBColor(0, 44, 75) #'#002C4B' #'rgb(0, 44, 75)'
LO_LIGHTGRAY = RGBColor(217, 217, 217) #'#D9D9D9' #'rgb(217, 217, 217)'
LO_GRAYBLUE = RGBColor(107, 135, 157) #'#6B879D' #'rgb(107, 135, 157)'
LO_DARKGRAYBLUE = RGBColor(79, 101, 119) #'#4F6577' #'rgb(79, 101, 119)'
LO_BLUE = RGBColor(0, 108, 184) #'#006CB8' #'rgb(0, 108, 184)'
LO_SKYBLUE = RGBColor(38, 165, 255) #'#26A5FF' #'rgb(38, 165, 255)'
LO_DARKRED = RGBColor(150, 0, 33) #'#960021' #'rgb(150, 0, 33)'
LO_PALEPINK = RGBColor(192, 102, 122) #'#C0667A' #'rgb(192, 102, 122)'
LO_GREEN = RGBColor(0, 117, 73) #'#007549' #'rgb(0, 117, 73)'
LO_PALEGREEN = RGBColor(127, 186, 164) #'#7FBAA4' #'rgb(127, 186, 164)'
LO_ORANGE = RGBColor(189, 73, 23) #'#BD4917' #'rgb(189, 73, 23)'
LO_LIGHTSALMON = RGBColor(222, 164, 139) #'#DEA48B' #'rgb(222, 164, 139)'
LO_GINGER = RGBColor(222, 164, 139) #'#DCAA00' #'rgb(220, 170, 0)'
LO_PALEYELLOW = RGBColor(237, 212, 127) #'#EDD47F' #'rgb(237, 212, 127)'
LO_PURPLE = RGBColor(151, 120, 211) #'#9778D3' #'rgb(151, 120, 211)'
LO_VIOLET = RGBColor(203, 187, 233) #'#CBBBE9' #'rgb(203, 187, 233)'

LO_COLOR_PALETTE = [
    LO_DIMGRAY,
    LO_TAN,
    LO_GRAY,
    LO_BROWN,
    LO_NAVY,
    LO_GRAYBLUE,
    LO_BLUE,
    LO_DARKRED,
    LO_PALEPINK,
    LO_GREEN,
    LO_ORANGE,
    LO_LIGHTSALMON,
    LO_GINGER,
    LO_LIGHTGRAY,
    LO_SKYBLUE,
    LO_PALEGREEN,
    LO_PALEYELLOW,
    LO_PURPLE,
    LO_VIOLET,
]


LO_DEFAULT_FONT_COLOR = RGBColor(57, 46, 44)

LO_HEATMAP_COLORSCALE = []

"""
COUNTRY_COLOR = {
    "US": 'rgb(150, 0, 33)',
    "China": 'rgb(204, 194, 187)',
    "EU": 'rgb(107, 135, 157)',
    "UK": 'rgb(0, 70, 119)',
    "Germany": 'rgb(80, 65, 62)',
    "France": 'rgb(192, 102, 122)',
    "Italy": 'rgb(0, 108, 184)',
    "Spain": 'rgb(255, 192, 0)',
    "Brazil": 'rgb(204, 194, 187)',
    "Russia": 'rgb(30, 30, 30)',
    "India": 'rgb(153, 171, 183)',
    "Mexico": 'rgb(142, 101, 76)',
    "Indonesia": 'rgb(0, 32, 96)',
    "Poland": 'rgb(142, 101, 76)',
}
"""