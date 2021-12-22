import cv2
import pytesseract
import matplotlib.pyplot as plt

# Tesseract의 실행 파일 경로를 지정합니다.

pytesseract.pytesseract.tesseract_cmd = r'C:\OCR\tesseract'

# 언어로 한글, 레거시 + LSTM 모드, 번호판 인식에 최적화된 psm 7 모드.

config = ('-l kor --oem 3 --psm 7')

"""
def getSubImage(rect, src):
    # rect 배열은 중앙점, 가로 세로 길이, 그리고 각도를 담고 있습니다.
    
    center, size, theta = rect[0], rect[1], rect[2]
    
    # Opencv에서 사용할 수 있도록 int로 변환합니다. 
    
    center, size = tuple(map(int, center)), tuple(map(int, size))
    
    # 원본 이미지의 세로  가로 길이를 저장합니다.
    
    height, width = src.shape[0], src.shape[1]
    
    # 사각형의 회전 행렬을 만듭니다.
    
    M = cv2.getRotationMatrix2D( center, theta, 1)
    
    # 원본 이미지를 회전 행렬을 통해 회전한 이미지를 만듭니다.
    # 회전하면서 원본 크기를 동일하게 유지하는 경우 잘리는 현상 때문에 크기를 2배로 늘립니다.
    
    dst = cv2.warpAffine(src, M, (2* width, 2* height))
    
    # 회전한 이미지에서 rect 정보를 이용해서 영수증 사각형을 잘라냅니다.
    
    out = cv2.getRectSubPix(dst, size, center)
    
    return out

"""

# 이미지 입력

img = cv2.imread('number.jpg', cv2.IMREAD_COLOR)

# 이미지 그레이스케일 변환

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 가우시안 블러 적용

dst = cv2.GaussianBlur(gray, (5, 5), 1)

canny1 = 30

canny2 = 200

# Canny 필터 적용함

dst1 = cv2.Canny(dst, canny1, canny2)
    
# 필터 적용된 이미지에서 계층 구조와 무관하게 모든 윤곽선을 검출하며, 모든 점을 반환합니다.  

contours, _ = cv2.findContours(dst1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# contours 배열에서 contourArea를 key로 삼아서 내림차순으로 정렬

cnts = sorted(contours, key = cv2.contourArea, reverse =  True)[:1000]

for n in cnts:
    
# 폐곡선 n의 전체 길이의 2%로 근사 정확도를 정하고, 꼭짓점의 개수를 줄여서 새로운 다각형을 만듭니다.
       
        approx = cv2.approxPolyDP(n, cv2.arcLength(n, True)*0.02, True)

# 꼭짓점 개수가 4개 (사각형)인지 확인합니다.
        
        if len(approx) == 4:

# 외곽선을 좌표와 가로 세로 길이로 변환합니다.

               x,y,w,h = cv2.boundingRect(n) #finds co-ordinates of the plate
               
# 그레이스케일 이미지에서 잘라내서 새로운 이미지를 만듭니다.
               
               new_img = gray[y:y+h,x:x+w]
               
# 이미지 회전을 위한 예비 변수입니다.
                
               rect = n
                
               break 

# 잘라낸 이미지에 새로 가우시안 블러를 적용합니다.

new_img = cv2.GaussianBlur(new_img, (1, 1), 1)

# 이미지에 Threshold를 적용합니다.

_, new_img = cv2.threshold(new_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 이미지에 외곽 경계선을 추가합니다.

new_img = cv2.copyMakeBorder(new_img, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))



"""

# 회전된 이미지를 다시 수평으로 맞춥니다.

new_img1 = getSubImage(cv2.minAreaRect(n), gray)

new_img1 = cv2.GaussianBlur(new_img1, (1, 1), 1)

_, new_img1 = cv2.threshold(new_img1, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

new_img1 = cv2.copyMakeBorder(new_img1, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
"""

# 콘솔에 번호를 출력합니다.

print(pytesseract.image_to_string(new_img, config=config))

# Tesseract에 이미지를 입력하여 결과를 chars로 받습니다.

chars = pytesseract.image_to_string(new_img, config=config)
    
result_chars = ''
for c in chars:

# 한글 완성형의 가 ~ 힣 과 숫자만을 추가합니다.    

    if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
        result_chars += c
    
print(result_chars)

# IDE의 Plot에 이미지를 출력합니다.

plt.imshow(new_img, cmap='gray')