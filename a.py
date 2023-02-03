
from MMEdu import MMPose
import shutil,mmcv,cv2,time
import numpy as np
font=cv2.FONT_ITALIC
fps = 10    #FPS
size=(544,960)    #图片、视频尺寸
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter('demo3.mp4',fourcc,fps,size)
video = mmcv.VideoReader('tes2.mp4')
model = MMPose(backbone='SCNet')
laslk,lasrk=[0,0,0],[0,0,0]
cnt=0
status=0
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle
all=0
for i in enumerate(video):
    all+=1
for i, frame in enumerate(video):
    print (str(i)+'in'+str(all))
    if i % 3 !=0 : continue
    result = model.inference(img=frame,device='cpu',show=False,save=True,work_dir='./save/',name='pic2'+str(i)) # 使用MMPose进行人体姿态估计
    lis=result[0]['keypoints']
    knee_angle=calculate_angle(lis[13],lis[11],lis[15])
    hip_angle=180-calculate_angle(lis[5],lis[11],lis[13])
    if hip_angle>120 and status==1: 
        cnt+=1
        status=0
    elif hip_angle<120 :
        status=1
    print(knee_angle,hip_angle)
    laslk,lasrk=result[0]['keypoints'][13],result[0]['keypoints'][14]
    print(laslk,lasrk,cnt)
    res=cv2.imread('./save/pic2'+str(i)+'.png')
    cv2.putText(res, "hip_angle:"+str(hip_angle), (30,30), font,1.3, (255, 255, 255),3)
    cv2.putText(res, "knee_angle:"+str(knee_angle), (30,80), font,1.3, (255, 255, 255),3)
    cv2.putText(res, "cnt:"+str(cnt), (30,130), font,1.3, (255, 255, 255),3)
    # cv2.imshow("res", res)
    # cv2.waitKey()
    videoWriter.write(res)
videoWriter.release()
