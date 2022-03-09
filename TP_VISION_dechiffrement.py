import cv2


def convertentier(b): #fonction qui convert binaire en entier
  n = 0
  for  i in b:
    n = (2 * n) + i    
  return n


image=input("donner le chemin de l image ")
img = cv2.imread(str(image),0)
if img is None : 
    print('image vide')
    exit(0)
else :
    
    cmp=0
    h,w = img.shape
    liste=[]      
    texte=''
    for y in range(h):
        for x in range(w):
            pixel= img[y,x] 
            maliste1=list(bin(pixel)[2:])
            liste.append(maliste1[-1])
   
   
    t=""
    i=0
    while(i<len(liste)):   
        d=i
        dliste=[]
        dliste1=[]
        while(d<i+7):
            dliste.append(liste[d])
            d+=1
        for a in range(len(dliste)):
            dliste[a]=int(dliste[a])    
            dliste1.append(dliste[a])
        dliste1=convertentier(dliste1)
        #print("---")
        t=chr(dliste1)
        if(t=="@"):      
            break
        if(t=="|"):
            t=" "
        texte=texte+t
        
           
        i+=7

    print(texte)
        
         
    
