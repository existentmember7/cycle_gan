import cv2

imgo = cv2.imread('/media/han/D/cy/dataset/images/A/12-6/228.png', cv2.IMREAD_UNCHANGED)
img = cv2.resize(imgo,(512,512))

# 顯示圖片
cv2.imshow('My Image', imgo)

# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()

# 顯示圖片
cv2.imshow('My Image', img)

# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()