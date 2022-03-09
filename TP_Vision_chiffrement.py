import cv2
import numpy as np


def convertentier(b): #fonction qui convert binaire en entier
  n = 0
  for  i in b:
    n = (2 * n) + i    
  return n

image=input("donner le chemin dune image ")
img = cv2.imread(str(image),0)
if img is None : 
    print('image vide')
    exit(0)
else :
    liste=[]
    text=input('le message a chiffrer : ') 
    texte0=""
    text2=text.split()
    for j in text2:
        texte0=texte0+j+"|"
        
    texte1=texte0+"@"  #condition d'arrete

    mon_texte = bytearray(str(texte1), "utf8") # //transformeation en binaire
    for j in mon_texte:
        maliste_texte = list(bin(j)[2:])  		#//transformation en liste binaire
       
        for i in range(len(maliste_texte)):
            maliste_texte[i]=int(maliste_texte[i])    #// transformation les caractérs binaire en entier
            liste.append(maliste_texte[i])			      
    cmp=0
    h,w = img.shape
    
    imageCode= np.zeros((h,w),np.uint8)
    for a in range(h):
        for b in range(w):
            imageCode[a,b]=img[a,b]

    for y in range(h):
        for x in range(w):
            liste1=[]
            pixel= img[y,x] 
            newpixel=list(bin(pixel)[2:])
            for i in range(len(newpixel)):
                 newpixel[i]=int(newpixel[i])    
                 liste1.append(newpixel[i])	
    
            liste1[-1]=liste[cmp]
            
            liste1=convertentier(liste1)
            
            imageCode[y,x]=liste1
            cmp+=1
            if cmp==len(liste):
                break
      
        if cmp==len(liste):
            break      


    cv2.imwrite('imagecode.png',imageCode)
    

