import cv2
import pytesseract
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r'C:\OCR\tesseract'
config = ('-l kor --oem 3 --psm 7')


img = cv2.imread('number.jpg', cv2.IMREAD_COLOR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dst = cv2.GaussianBlur(gray, (5, 5), 1)

canny1 = 30

canny2 = 200

dst1 = cv2.Canny(dst, canny1, canny2)
    
contours, _ = cv2.findContours(dst1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

cnts = sorted(contours, key = cv2.contourArea, reverse =  True)[:1000]

for n in cnts:

        approx = cv2.approxPolyDP(n, cv2.arcLength(n, True)*0.02, True)
        if len(approx) == 4:
            
               x,y,w,h = cv2.boundingRect(n) #finds co-ordinates of the plate
               new_img=gray[y:y+h,x:x+w]
               rect = n
                
               break 
        
new_img = cv2.GaussianBlur(new_img, (1, 1), 1)

_, new_img = cv2.threshold(new_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

new_img = cv2.copyMakeBorder(new_img, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
"""
new_img1 = getSubImage(cv2.minAreaRect(n), gray)

new_img1 = cv2.GaussianBlur(new_img1, (1, 1), 1)

_, new_img1 = cv2.threshold(new_img1, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

new_img1 = cv2.copyMakeBorder(new_img1, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
"""

chars = pytesseract.image_to_string(new_img, config=config)
    
result_chars = ''
has_digit = False
for c in chars:
    if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
        if c.isdigit():
            has_digit = True
        result_chars += c
    
print(result_chars)


plt.imshow(new_img, cmap='gray')