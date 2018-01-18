import pandas as pd
import jieba
import re
a = ['️', '[', '心', ']']
b = '胸怀决定你的格局[good]噢对[dhsf]了'
c = re.split('\[.*?\]', b, re.S)
print(c)
