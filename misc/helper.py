import cv2


vid = cv2.VideoCapture('http://localhost:8554/', cv2.CAP_FFMPEG)

while(True):

    ret, frame = vid.read()
    if ret is False:
        print('failed')
        break

    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
