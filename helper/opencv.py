import cv2


def print_fourcc(cap) -> None:
    fourcc_code = cap.get(cv2.CAP_PROP_FOURCC)
    fourcc_int = int(fourcc_code)
    fourcc_str = chr(fourcc_int&0xff) + chr((fourcc_int>>8)&0xff) + chr((fourcc_int>>16)&0xff) + chr((fourcc_int>>24)&0xff) 
    print(f'fourcc: {fourcc_str}')

cap = cv2.VideoCapture('/dev/video0')

print_fourcc(cap)

while(True):

    ret, frame = cap.read()
    if ret is False:
        print('failed')
        break 
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
